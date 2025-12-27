"""数据集划分脚本 - 将原始数据划分为训练集/验证集/测试集"""
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.training_config import BaseConfig


def split_dataset(config=None):
    """
    将数据集划分为训练集、验证集和测试集

    Args:
        config: BaseConfig 配置对象，如果为None则使用默认配置
    """
    if config is None:
        config = BaseConfig()

    # 设置随机种子以保证可复现性
    random.seed(config.random_seed)

    # 检查源数据目录是否存在
    if not config.raw_data_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在: {config.raw_data_dir}")

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        split_dir = config.processed_data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("开始数据划分")
    print("="*60)
    print(f"源数据目录: {config.raw_data_dir}")
    print(f"输出目录: {config.processed_data_dir}")
    print(f"划分比例: Train={config.train_ratio}, Val={config.val_ratio}, Test={config.test_ratio}")
    print(f"随机种子: {config.random_seed}")
    print("="*60 + "\n")

    # 获取所有类别
    classes = [d.name for d in config.raw_data_dir.iterdir() if d.is_dir()]
    classes = sorted(classes)  # 确保顺序一致

    total_stats = {'train': 0, 'val': 0, 'test': 0}
    class_stats = {}

    # 对每个类别独立进行分层采样
    for class_name in classes:
        print(f"\n处理类别: {class_name}")
        print("-"*40)

        class_dir = config.raw_data_dir / class_name

        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(ext)))

        if len(image_files) == 0:
            print(f"  警告: 类别 {class_name} 中没有找到图片，跳过")
            continue

        print(f"  总共找到 {len(image_files)} 张图片")

        # 第一次划分: 分离出测试集
        train_val_files, test_files = train_test_split(
            image_files,
            test_size=config.test_ratio,
            random_state=config.random_seed
        )

        # 第二次划分: 从训练+验证集中分离出验证集
        # 计算验证集在剩余数据中的比例
        val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)

        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_ratio_adjusted,
            random_state=config.random_seed
        )

        # 统计
        class_stats[class_name] = {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'total': len(image_files)
        }

        print(f"  划分结果:")
        print(f"    训练集: {len(train_files)} 张")
        print(f"    验证集: {len(val_files)} 张")
        print(f"    测试集: {len(test_files)} 张")

        # 复制文件到对应目录
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            # 创建类别子目录
            split_class_dir = config.processed_data_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            # 复制文件
            for img_file in files:
                dst = split_class_dir / img_file.name
                shutil.copy2(img_file, dst)

            total_stats[split_name] += len(files)

        print(f"  ✓ 文件已复制到目标目录")

    # 打印总结
    print("\n" + "="*60)
    print("数据划分完成！")
    print("="*60)
    print("\n总体统计:")
    print(f"  训练集: {total_stats['train']} 张")
    print(f"  验证集: {total_stats['val']} 张")
    print(f"  测试集: {total_stats['test']} 张")
    print(f"  总计: {sum(total_stats.values())} 张")

    print("\n各类别统计:")
    print(f"{'类别':<15} {'训练集':>8} {'验证集':>8} {'测试集':>8} {'总计':>8}")
    print("-"*60)
    for class_name, stats in class_stats.items():
        print(f"{class_name:<15} {stats['train']:>8} {stats['val']:>8} "
              f"{stats['test']:>8} {stats['total']:>8}")

    print("\n" + "="*60)
    print(f"数据已保存到: {config.processed_data_dir}")
    print("="*60)


if __name__ == '__main__':
    # 运行数据划分
    split_dataset()
