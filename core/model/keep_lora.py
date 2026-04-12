import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .finetune import Finetune
import math



class KeepLoRA(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        if 'num_classes' in kwargs:
            num_class = kwargs.pop('num_classes')
        super().__init__(backbone, feat_dim, num_class, **kwargs)


        self.epsilon_w = kwargs.get('epsilon_w', 0.85)
        self.epsilon_f = kwargs.get('epsilon_f', 0.99)
        self.lora_rank = kwargs.get('lora_rank', 8)
        self.lora_alpha = kwargs.get('lora_alpha', 16)

        self.W_p = None
        self.M = None

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

        self.hook_handles = []


    def _init_pretrained_subspace(self, device):
        """
        初始化预训练权重的主空间 W_p (对应论文公式 2)
        """
        print(f"[{self.__class__.__name__}] 正在提取预训练模型的主空间 (W_p)...")
        self.W_p = {}

        # 遍历骨干网络的所有参数
        for name, param in self.backbone.named_parameters():
            # 只处理二位以上的权重矩阵
            if 'weight' not in name or param.dim() < 2:
                continue

            # 1. 形状统一化：
            # 如果是卷积层 (例如 shape 为 [64, 3, 3, 3])，需要展平成二维矩阵 [64, 27]
            # 如果是全连接层，它本来就是 [out_features, in_features]
            W_mat = param.data.view(param.data.size(0), -1).to(device)

            # 2. 奇异值分解 (SVD)
            # U: 左奇异向量, S: 奇异值 (一维 Tensor), Vh: 右奇异向量的转置
            U, S, Vh = torch.linalg.svd(W_mat, full_matrices=False)

            # 3. 计算“能量”并截断 (对应论文 Eq. 2)
            # 矩阵的 Frobenius 范数平方，等于其所有奇异值的平方和
            S_sq = S ** 2
            total_energy = torch.sum(S_sq)

            # 计算累计能量占比
            cum_energy_ratio = torch.cumsum(S_sq, dim=0) / total_energy

            # 找到刚刚满足 cumulative energy >= epsilon_w 的那个索引 (即截断点 p)
            p_indices = (cum_energy_ratio >= self.epsilon_w).nonzero()

            # 如果所有的能量加起来都达不到要求（虽然极少发生），就全保留
            if len(p_indices) == 0:
                p = len(S)
            else:
                p = p_indices[0].item() + 1

            # 4. 提取并保存主空间
            # 取 U 的前 p 列作为主空间 W_p
            self.W_p[name] = Vh[:p, :].t().clone()

    def _get_initial_gradient(self, train_loader, device):
        """
        通过一次模拟的前向和反向传播，获取模型在当前任务上的初始梯度 G_t。
        """
        print(f"[{self.__class__.__name__}] 正在获取当前任务的初始梯度 (G_t)...")
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone.train()
        self.zero_grad()

        # 2. 抽取数据：从 DataLoader 里拿一个 Batch 的真实数据
        try:
            batch_data = next(iter(train_loader))
            # 兼容不同的数据返回格式
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)
            else:
                images, labels = batch_data[0].to(device), batch_data[1].to(device)
        except StopIteration:
            raise ValueError("train_loader为空")

        # 3. 前向传播 (Forward)
        # 获取骨干网络提取的特征
        features = self.backbone(images)

        if isinstance(features, dict):
            features = features['features']

        # 将特征送入分类头计算预测 logits (得分)
        logits = self.classifier(features)

        # 计算损失 (使用交叉熵)
        loss = F.cross_entropy(logits, labels)

        # 4.反向传播
        loss.backward()

        # 5.收集梯度
        grad_dict = {}
        for name, param in self.backbone.named_parameters():
            # 只关心挂了LoRA的二位以上权重矩阵
            if 'weight' in name and param.dim() >= 2:
                if param.grad is not None:
                    grad_dict[name] = param.grad.clone()
                else:
                    grad_dict[name] = torch.zeros_like(param.data)

        # 6.清空反向传播痕迹
        self.zero_grad()

        # 关掉backbone的梯度，因为后续只有LoRA_B需要梯度
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"[{self.__class__.__name__}] G_t 获取成功！准备进入残差空间投影...")
        return grad_dict

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        print(f"[{self.__class__.__name__}] 正在为 Task {task_idx} 初始化 KeepLoRA...")

        # 0. 准备工作：确保拿到设备信息，并把模型切换到训练模式
        device = next(self.backbone.parameters()).device
        self.train()

        # 只在第一次任务初始化预训练主子空间W_p
        if task_idx == 0 and self.W_p is None:
            self._init_pretrained_subspace(device)

        # 1. 获取针对当前任务的“理想梯度” G_t
        # 注意：这里需要你封装一个小函数，它需要：
        # - 从 train_loader 里拿一个 batch 的数据
        # - 让 backbone 开启 requires_grad=True
        # - 算 loss 并 .backward()
        # - 收集并返回每一层的梯度字典
        grad_dict = self._get_initial_gradient(train_loader, device)

        for name, param in self.backbone.named_parameters():
            if 'weight' not in name or param.dim() < 2:
                continue

            layer_grad = grad_dict[name]

            layer_grad_2d = layer_grad.view(layer_grad.size(0),-1)
            # (1) 残差空间投影 (对应论文 Eq. 5)
            G_hat = layer_grad_2d.clone()
            # 剔除预训练主空间方向
            if self.W_p is not None and name in self.W_p:
                W_p_layer = self.W_p[name]
                # G_hat = G_t - W_p * W_p^T * G_t
                G_hat = G_hat - torch.mm(layer_grad_2d,torch.mm(W_p_layer,W_p_layer.t()))

            # 剔除历史任务主导方向
            if self.M is not None and name in self.M:
                M_layer = self.M[name]
                # G_hat = G_hat - M_{t-1} * M_{t-1}^T * G_t
                G_hat = G_hat - torch.mm(layer_grad_2d,torch.mm(M_layer,M_layer.t()))

            # (2) SVD 分解 (对应论文 Eq. 6)
            # PyTorch 的 torch.linalg.svd 返回的是 U, S, Vh
            U, S, Vh = torch.linalg.svd(G_hat, full_matrices=False)

            # 截取前r个方向
            r = min(self.lora_rank, U.size(1))
            # A取U的前r列
            A_init = U[:, :r]
            # B需要结合奇异值S和Vh
            B_init = torch.mm(torch.diag(S[:r]), Vh[:r, :])

            # 将变量名中的.改为_
            safe_name = name.replace('.','_')

            # 把初始化好的 A 和 B 存入ParameterDict 里
            self.lora_A[safe_name] = nn.Parameter(A_init.to(device))
            self.lora_B[safe_name] = nn.Parameter(B_init.to(device))

            # (3) 冻结 A (KeepLoRA 的关键设计)
            self.lora_A[safe_name].requires_grad = False

            # (4) 反向补偿骨干网络 (Modified W)
            # W' = W - (alpha / r) * A * B
            with torch.no_grad():
                compensation = (self.lora_alpha / r) * torch.mm(self.lora_A[safe_name], self.lora_B[safe_name])
                param.sub_(compensation.view(param.shape))

            # 确保骨干网络本身不参与后续的梯度更新
            param.requires_grad = False

        self._register_lora_hook()
        print(f"[{self.__class__.__name__}] Task {task_idx} 的KeepLoRA初始化完成")




    def observe(self, data):
        """
        执行一次训练步（Forward + Loss 计算）。
        反向传播 (backward) 和优化器更新 (step) 会由 Trainer 在外部自动调用。
        """

        # 1. 解析数据并放到显卡上
        device = next(self.backbone.parameters()).device
        images = data['image'].to(device)
        labels = data['label'].to(device)

        # 2.前向传播
        features = self.backbone(images)

        if isinstance(features, dict):
            features = features['features']

        logits = self.classifier(features)

        # 3.计算损失（交叉熵）
        loss = F.cross_entropy(logits, labels)

        # 计算当前batch的准确率
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).item()

        # 按照 LibContinual 父类的契约，严格返回这三个值
        return preds, acc / images.size(0), loss

    def get_parameters(self, config):
        """
        在每个 Task 训练前被 Trainer 调用。
        告诉优化器：“这个任务里，你只被允许更新这些参数”。
        对应论文中“只优化 KeepLoRA 参数 B_t”的设定。
        """

        print(f"[{self.__class__.__name__}] 正在向优化器注册可训练参数 (lora_B 和 classifier)...")

        params_to_update = []

        # (1) 将我们所有的 LoRA B 矩阵加入优化列表
        for name, param in self.lora_B.items():
            params_to_update.append({'params': param})

        # (2) 分类头 (Classifier) 需要适应新任务类别，必须加入优化列表
        params_to_update.append({'params': self.classifier.parameters()})

        return params_to_update

    def _register_lora_hook(self):
        """
        利用 PyTorch Hook 机制，将 LoRA 的计算动态注入到 Backbone 中。
        """
        print(f"[{self.__class__.__name__}] 正在将 KeepLoRA 挂载到 Backbone 的前向传播中...")

        # 遍历之前在before_task里建好的LoRA字典
        for name , param in self.backbone.named_parameters():
            safe_name = name.replace('.','_')

            if safe_name not in self.lora_A:
                continue

            module_name = name.rsplit('.',1)[0]
            module = self.backbone.get_submodule(module_name)

            def make_hook(safe_layer_name):
                def lora_hook(module, inputs, output):
                    x = inputs[0]
                    A = self.lora_A[safe_layer_name]
                    B = self.lora_B[safe_layer_name]
                    r = A.size(1)

                    if isinstance(module, nn.Linear):
                        lora_out = F.linear(x, B)
                        lora_out = F.linear(lora_out, A)
                    elif isinstance(module, nn.Conv2d):
                        # (1) 还原 B 的形状：将其变回一个真正的卷积核
                        # 形状: [rank, in_channels_per_group, kernel_h, kernel_w]
                        b_shape = (r, module.in_channels // module.groups, *module.kernel_size)
                        B_reshaped = B.view(b_shape)

                        # (2) 还原 A 的形状：将其变成一个 1x1 卷积核用于升维
                        # 形状: [out_channels, rank, 1, 1]
                        a_shape = (module.out_channels, r, 1, 1)
                        A_reshaped = A.view(a_shape)

                        # (3) 第一次卷积：用 B 提取特征
                        # 必须完美继承原厂卷积层的步长(stride)、填充(padding)等属性！
                        lora_out = F.conv2d(x, B_reshaped,
                                            stride=module.stride,
                                            padding=module.padding,
                                            dilation=module.dilation,
                                            groups=module.groups)

                        # (4) 第二次卷积：用 A 进行 1x1 升维整合
                        # 此时不需要 padding 和 stride 了
                        lora_out = F.conv2d(lora_out, A_reshaped)
                    else:
                        return output

                    lora_out = lora_out * (self.lora_alpha / r)

                    return output + lora_out

                return lora_hook

            handle = module.register_forward_hook(make_hook(safe_name))
            self.hook_handles.append(handle)

        print(f"[{self.__class__.__name__}] 所有 Hook 挂载完毕，前向传播已接管！")

    def _get_layer_inputs(self, train_loader, device, max_samples=256):
        """
        利用前向传播钩子 (Forward Hook) 悄悄收集各个指定层的输入特征 X_t。
        """
        print(f"[{self.__class__.__name__}] 正在收集各层的输入特征 (最多 {max_samples} 个样本)...")

        X_dict_list = {}
        hook_handles = []

        # 1.定义“间谍”钩子
        def get_activation(layer_name, module):
            def hook(model, inputs, output):
                x = inputs[0].detach()

                # --核心数学对齐：维度重构--
                if isinstance(module, nn.Conv2d):
                    x_unfold = F.unfold(x, kernel_size=module.kernel_size, dilation=module.dilation,
                                        padding=module.padding, stride=module.stride)
                    x_flat = x_unfold.transpose(1, 2).reshape(-1, x_unfold.size(1))
                elif isinstance(module, nn.Linear):
                    x_flat = x.reshape(-1, x.size(-1))
                else:
                    return

                # 防止显存崩溃
                if layer_name not in X_dict_list:
                    X_dict_list[layer_name] = []
                X_dict_list[layer_name].append(x_flat.cpu())

            return hook

        # 2.挂载钩子
        for name, module in self.backbone.named_modules():
            # 我们之前存在 W_p 和 M 里的 key 是带有 ".weight" 后缀的
            weight_name = name + ".weight"

            # 只在我们重点关注的层（有预训练主空间的层）上挂钩子
            if self.W_p is not None and weight_name in self.W_p:
                handle = module.register_forward_hook(get_activation(weight_name, module))
                hook_handles.append(handle)

        # 3.模拟前向传播，收集数据
        self.backbone.eval()
        collected_samples = 0
        with torch.no_grad():
            for batch_data in train_loader:
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(device)
                else:
                    images = batch_data[0].to(device)

                # 前向传播
                _ = self.backbone(images)
                collected_samples += images.size(0)
                if collected_samples >= max_samples:
                    break  # 收集够了就停止，SVD只需要几百个样本就足以提取主方向

        # 4.卸载钩子，拼接Tensor
        for handle in hook_handles:
            handle.remove()

        X_dict = {}
        for name, tensor_list in X_dict_list.items():
            # 把CPU上多个batch的特征拼起来
            X_concat = torch.cat(tensor_list, dim=0)
            # 最后再统一搬回显卡，交给外面的函数做SVD分解
            X_dict[name] = X_concat.to(device)

        print(f"[{self.__class__.__name__}] 特征收集完毕！")
        return X_dict

    def _update_feature_space(self, train_loader, device):
        """
        提取当前任务的特征主方向，追加到记忆矩阵 M 中。
        """
        print(f"[{self.__class__.__name__}] 正在提取当前任务特征主方向 (M_t)...")
        if self.M is None:
            self.M = {}

        # 1.收集当前任务的输入特征X_t
        X_dict = self._get_layer_inputs(train_loader, device)

        for name, X_t in X_dict.items():
            # (2) 投影到残差空间 (对应论文 Eq. 3: X_hat = X_t - W_p W_p^T X_t - M M^T X_t)
            X_hat = X_t.clone()
            if self.W_p is not None and name in self.W_p:
                W_p_layer = self.W_p[name]
                # X_t 乘法维度处理 (根据右乘原理)
                X_hat = X_hat - torch.mm(X_t, torch.mm(W_p_layer, W_p_layer.t()))

            if name in self.M:
                M_layer = self.M[name]
                X_hat = X_hat - torch.mm(X_t, torch.mm(M_layer, M_layer.t()))

            # (3) 对残差特征 X_hat 做 SVD 分解
            U, S, Vh = torch.linalg.svd(X_hat, full_matrices=False)

            # (4) 根据能量阈值 epsilon_f 决定保留多少个主方向 (对应论文 Eq. 4)
            S_sq = S ** 2
            total_energy = torch.sum(S_sq)
            cum_energy_ratio = torch.cumsum(S_sq, dim=0) / total_energy

            k_indices = (cum_energy_ratio >= self.epsilon_f).nonzero()
            k = k_indices[0].item() + 1 if len(k_indices) > 0 else len(S)

            # 提取主方向 (右奇异向量 Vh 的前 k 行，并转置回列向量形式)
            V_k = Vh[:k, :].t()

            # (5) 将新方向追加到历史记忆 M 中
            if name not in self.M:
                self.M[name] = V_k
            else:
                self.M[name] = torch.cat([self.M[name], V_k], dim=1)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        """
        在每个任务训练结束后调用，负责知识的融合与记忆空间的更新。
        """
        print(f"[{self.__class__.__name__}] Task {task_idx} 训练结束，进入记忆巩固阶段...")
        device = next(self.backbone.parameters()).device

        # 1. 卸载所有的前向传播钩子 (Hooks)
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []

        # 2. 权重合并 (Merge) [cite: 191]
        for name, param in self.backbone.named_parameters():
            safe_name = name.replace('.','_')
            if safe_name in self.lora_A and safe_name in self.lora_B:
                A = self.lora_A[safe_name]
                B = self.lora_B[safe_name]
                r = A.size(1)

                with torch.no_grad():
                    # A 和 B 相乘后，恢复成原生参数的形状 (不管是 Conv 还是 Linear)
                    delta_W = torch.mm(A, B).view(param.shape) * (self.lora_alpha / r)
                    # 把学到的新知识正式刻入骨干网络
                    param.add_(delta_W)

        # 3.提取并更新历史任务主方向
        self._update_feature_space(train_loader, device)

        # 4.清理lora_A,lora_B
        self.lora_A.clear()
        self.lora_B.clear()

        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"[{self.__class__.__name__}] Task {task_idx} 完美收官！")
