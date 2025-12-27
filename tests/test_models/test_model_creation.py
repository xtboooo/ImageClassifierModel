"""测试模型创建"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.model_factory import create_model


def test_mobilenet_creation():
    """测试 MobileNetV2 模型创建"""
    model = create_model('mobilenet_v2', num_classes=3, pretrained=False)

    # 测试模型可以创建
    assert model is not None

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    # 检查输出形状
    assert output.shape == (2, 3), f"Expected shape (2, 3), got {output.shape}"

    print("✓ MobileNetV2 模型创建测试通过")


def test_model_parameters():
    """测试模型参数统计"""
    model = create_model('mobilenet_v2', num_classes=3, pretrained=False)

    params_info = model.count_parameters()

    # 检查参数统计
    assert 'total' in params_info
    assert 'trainable' in params_info
    assert 'frozen' in params_info
    assert params_info['total'] > 0
    assert params_info['trainable'] > 0

    print("✓ 模型参数统计测试通过")


def test_freeze_unfreeze():
    """测试冻结/解冻功能"""
    model = create_model('mobilenet_v2', num_classes=3, pretrained=False)

    # 初始状态（主干已冻结）
    initial_params = model.count_parameters()
    assert initial_params['frozen'] > 0

    # 解冻主干
    model.unfreeze_backbone()
    unfrozen_params = model.count_parameters()
    assert unfrozen_params['trainable'] > initial_params['trainable']

    # 重新冻结
    model.freeze_backbone()
    frozen_params = model.count_parameters()
    assert frozen_params['frozen'] > 0

    print("✓ 冻结/解冻功能测试通过")


if __name__ == '__main__':
    test_mobilenet_creation()
    test_model_parameters()
    test_freeze_unfreeze()
    print("\n✅ 所有模型测试通过！")
