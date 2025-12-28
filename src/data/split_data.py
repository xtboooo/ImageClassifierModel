"""数据集划分脚本 - 将原始数据划分为训练集/验证集/测试集"""
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.training_config import BaseConfig
from src.utils.logger import logger
from src.utils.rich_console import print_header, print_table


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

    print_header("数据集划分", "将原始数据划分为训练集/验证集/测试集")
    logger.info("数据划分配置",
                raw_data_dir=str(config.raw_data_dir),
                output_dir=str(config.processed_data_dir),
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                random_seed=config.random_seed)

    # 递归查找所有包含图片的文件夹，按文件夹名称分类
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    # 收集所有图片，按文件夹名称（类别）分组
    class_images = {}

    # 递归遍历所有子目录
    for root_path in config.raw_data_dir.rglob('*'):
        if root_path.is_file() and root_path.suffix in image_extensions:
            # 使用图片所在的直接父文件夹名称作为类别
            class_name = root_path.parent.name

            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(root_path)

    # 获取所有类别并排序
    classes = sorted(class_images.keys())

    if not classes:
        raise ValueError(f"在 {config.raw_data_dir} 下没有找到任何图片文件")

    logger.info(f"找到 {len(classes)} 个类别", classes=', '.join(classes))

    total_stats = {'train': 0, 'val': 0, 'test': 0}
    class_stats = {}

    # 对每个类别独立进行分层采样
    for class_name in classes:
        logger.info(f"处理类别: {class_name}")

        # 获取该类别的所有图片文件
        image_files = class_images[class_name]

        if len(image_files) == 0:
            logger.warning(f"类别 {class_name} 中没有找到图片，跳过")
            continue

        logger.info(f"类别 {class_name} 总共找到 {len(image_files)} 张图片")

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

        logger.info(f"类别 {class_name} 划分结果",
                    train=len(train_files),
                    val=len(val_files),
                    test=len(test_files))

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

        logger.success(f"类别 {class_name} 文件已复制到目标目录")

    # 打印总结
    logger.success("数据划分完成",
                   train=total_stats['train'],
                   val=total_stats['val'],
                   test=total_stats['test'],
                   total=sum(total_stats.values()))

    # 总体统计表格
    print_table(
        title="总体统计",
        headers=["数据集", "图片数量"],
        rows=[
            ["训练集", total_stats['train']],
            ["验证集", total_stats['val']],
            ["测试集", total_stats['test']]
        ],
        caption=f"总计: {sum(total_stats.values())} 张图片"
    )

    # 各类别统计表格
    class_rows = [
        [class_name, stats['train'], stats['val'], stats['test'], stats['total']]
        for class_name, stats in class_stats.items()
    ]
    print_table(
        title="各类别统计",
        headers=["类别", "训练集", "验证集", "测试集", "总计"],
        rows=class_rows
    )

    logger.success(f"数据已保存到: {config.processed_data_dir}")


if __name__ == '__main__':
    # 运行数据划分
    split_dataset()
