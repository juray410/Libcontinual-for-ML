import os
import torchvision
from tqdm import tqdm


def export_cifar10(root='./data/cifar10'):
    # 1. 利用 torchvision 自动从镜像站下载原始数据
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    classes = train_set.classes  # ['airplane', 'automobile', ...]

    # 2. 遍历并保存为图片文件夹格式
    for name, dataset in [('train', train_set), ('test', test_set)]:
        print(f"正在转换 {name} 数据集...")
        for i, (img, target) in enumerate(tqdm(dataset)):
            # 建立路径: ./data/cifar10/train/airplane/
            cls_dir = os.path.join(root, name, classes[target])
            os.makedirs(cls_dir, exist_ok=True)

            # 保存图片
            img.save(os.path.join(cls_dir, f"{i}.png"))


if __name__ == "__main__":
    export_cifar10()