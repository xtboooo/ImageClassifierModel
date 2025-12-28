"""训练配置类"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """训练超参数配置"""

    # 模型配置
    model_name: str = 'mobilenet_v2'
    num_classes: int = 3
    pretrained: bool = True

    # 训练超参数
    num_epochs: int = 30
    batch_size: int = 16  # 小数据集使用小batch
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 优化器和调度器
    optimizer: str = 'adamw'  # AdamW对小数据集效果好
    scheduler: str = 'cosine'  # 余弦退火
    warmup_epochs: int = 3

    # 数据增强
    img_size: int = 224

    # 正则化
    dropout: float = 0.3
    label_smoothing: float = 0.1  # 标签平滑防止过拟合

    # 早停
    early_stopping_patience: int = 10

    # 设备
    device: str = 'auto'  # auto/mps/cuda/cpu
    num_workers: int = 4

    # 两阶段训练
    stage1_epochs: int = 10  # 阶段1：冻结主干
    stage2_epochs: int = 20  # 阶段2：微调
    stage2_lr: float = 1e-4  # 阶段2降低学习率
    unfreeze_from_layer: int = 14  # 从第14层开始解冻

    # 路径
    data_root: Path = Path('data/processed')
    output_dir: Path = Path('data/output')
    checkpoint_dir: Path = Path('data/output/checkpoints')
    log_dir: Path = Path('data/output/logs')

    # 类别名称
    class_names: list = None

    def __post_init__(self):
        """初始化后处理"""
        # 确保路径是Path对象
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

        # 设置默认类别名称
        if self.class_names is None:
            self.class_names = ['Failure', 'Loading', 'Success']

        # 创建必要的目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BaseConfig:
    """基础配置（项目级别）"""

    project_name: str = "ImageClassifierModel"
    version: str = "0.1.0"
    random_seed: int = 42

    # 数据集路径
    raw_data_dir: Path = Path('data/input')
    processed_data_dir: Path = Path('data/processed')

    # 数据划分比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self):
        """初始化后处理"""
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)

        # 验证比例和为1
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "数据划分比例之和必须为1"
