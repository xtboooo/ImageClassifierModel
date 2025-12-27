"""DataLoader 创建和管理"""
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import ScreenshotDataset
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(data_root, batch_size=16, num_workers=4, img_size=224):
    """
    创建训练/验证/测试 DataLoader

    Args:
        data_root: 数据根目录（包含train/val/test子目录）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)

    # 创建训练集Dataset
    train_dataset = ScreenshotDataset(
        data_root / 'train',
        transform=get_train_transforms(img_size)
    )

    # 创建验证集Dataset
    val_dataset = ScreenshotDataset(
        data_root / 'val',
        transform=get_val_transforms(img_size)
    )

    # 创建测试集Dataset
    test_dataset = ScreenshotDataset(
        data_root / 'test',
        transform=get_val_transforms(img_size)
    )

    # 打印数据集信息
    print("\n" + "="*50)
    print("数据集信息:")
    print("="*50)
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"  类别分布: {train_dataset.get_class_distribution()}")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"  类别分布: {val_dataset.get_class_distribution()}")
    print(f"测试集: {len(test_dataset)} 张图片")
    print(f"  类别分布: {test_dataset.get_class_distribution()}")
    print(f"类别: {train_dataset.classes}")
    print("="*50 + "\n")

    # 创建DataLoader
    # 训练集: shuffle=True, drop_last=True（丢弃最后一个不完整batch）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # 加速数据传输到GPU/MPS
        drop_last=True    # 丢弃最后不完整的batch（保持batch大小一致）
    )

    # 验证集和测试集: shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_single_dataloader(data_dir, batch_size=16, num_workers=4,
                             img_size=224, is_train=False):
    """
    创建单个DataLoader（用于推理或特殊场景）

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        is_train: 是否使用训练增强

    Returns:
        DataLoader: 数据加载器
    """
    transform = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)

    dataset = ScreenshotDataset(data_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    return loader
