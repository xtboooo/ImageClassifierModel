"""模型工厂 - 统一的模型创建接口"""
from .mobilenet_v2 import MobileNetV2Classifier
from ..utils.logger import logger
from ..utils.rich_console import print_table


def create_model(model_name='mobilenet_v2', num_classes=3, pretrained=True, **kwargs):
    """
    创建模型

    Args:
        model_name: 模型名称 ('mobilenet_v2', 'efficientnet_lite0' 等)
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        **kwargs: 其他模型特定参数

    Returns:
        model: 创建的模型实例

    Raises:
        ValueError: 如果模型名称不支持
    """
    model_name = model_name.lower()

    if model_name == 'mobilenet_v2':
        model = MobileNetV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.3)
        )
    else:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"支持的模型: ['mobilenet_v2']"
        )

    # 打印模型信息
    params_info = model.count_parameters()

    print_table(
        title=f"模型信息: {model_name}",
        headers=["配置项", "值"],
        rows=[
            ["总参数", f"{params_info['total']:,}"],
            ["可训练参数", f"{params_info['trainable']:,} ({params_info['trainable_percentage']:.1f}%)"],
            ["冻结参数", f"{params_info['frozen']:,}"],
            ["预训练", "是" if pretrained else "否"],
            ["分类数量", str(num_classes)]
        ]
    )

    logger.info("模型创建完成", model=model_name, params=params_info['total'])

    return model


def load_model_from_checkpoint(checkpoint_path, model_name='mobilenet_v2', num_classes=3):
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        model_name: 模型名称
        num_classes: 分类数量

    Returns:
        model: 加载了权重的模型
        checkpoint: 检查点数据（包含其他训练信息）
    """
    import torch

    # 创建模型（不使用预训练权重，因为要加载自己的权重）
    model = create_model(model_name, num_classes, pretrained=False)

    # 加载检查点（weights_only=False 因为我们信任自己训练的检查点）
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.success("从检查点加载模型", path=str(checkpoint_path))

    # 打印检查点信息
    if 'history' in checkpoint:
        history = checkpoint['history']
        if 'val_acc' in history and len(history['val_acc']) > 0:
            best_acc = max(history['val_acc'])
            logger.info("检查点信息", best_val_acc=f"{best_acc:.2f}%")

    return model, checkpoint
