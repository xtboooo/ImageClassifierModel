"""数据增强和预处理"""
import torchvision.transforms as T


def get_train_transforms(img_size=224):
    """
    训练集数据增强（激进策略应对小数据集）

    Args:
        img_size: 目标图像尺寸

    Returns:
        torchvision.transforms.Compose: 数据增强pipeline
    """
    return T.Compose([
        # 几何变换
        T.Resize((256, 256)),  # 先放大
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # 随机裁剪
        T.RandomHorizontalFlip(p=0.5),  # 水平翻转
        T.RandomRotation(degrees=10),  # 随机旋转±10度

        # 颜色变换
        T.ColorJitter(
            brightness=0.3,  # 亮度
            contrast=0.3,    # 对比度
            saturation=0.3,  # 饱和度
            hue=0.1          # 色调
        ),
        T.RandomGrayscale(p=0.1),  # 随机转灰度

        # 模糊和噪声
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),

        # 转换为Tensor并归一化（ImageNet统计值）
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # 随机擦除（防止过拟合）
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size=224):
    """
    验证/测试集增强（仅必要操作）

    Args:
        img_size: 目标图像尺寸

    Returns:
        torchvision.transforms.Compose: 数据增强pipeline
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(img_size=224):
    """
    推理时的数据预处理（与验证集相同）

    Args:
        img_size: 目标图像尺寸

    Returns:
        torchvision.transforms.Compose: 数据预处理pipeline
    """
    return get_val_transforms(img_size)
