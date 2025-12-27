"""MobileNetV2 图像分类器"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    """
    基于 MobileNetV2 的三分类器

    特性:
    - ImageNet 预训练权重
    - 支持两阶段训练（冻结/解冻主干）
    - 针对小数据集优化（Dropout防止过拟合）
    """

    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: 分类数量
            pretrained: 是否使用预训练权重
            dropout: Dropout比例
        """
        super().__init__()

        # 加载预训练的 MobileNetV2
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        # 冻结主干网络（阶段1训练策略）
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # 替换分类头
        # MobileNetV2 的 classifier 是一个 Sequential，最后一层是 Linear(1280, 1000)
        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 tensor, shape (batch_size, 3, H, W)

        Returns:
            输出 logits, shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def unfreeze_backbone(self, unfreeze_from_layer=14):
        """
        解冻主干网络的后几层（阶段2训练策略）

        Args:
            unfreeze_from_layer: 从第几层开始解冻（默认14层）
                                MobileNetV2 的 features 总共有 18 层
        """
        print(f"\n解冻主干网络从第 {unfreeze_from_layer} 层开始...")

        for i, child in enumerate(self.backbone.features.children()):
            if i >= unfreeze_from_layer:
                for param in child.parameters():
                    param.requires_grad = True
                print(f"  - 层 {i}: 已解冻")

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n可训练参数: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)\n")

    def freeze_backbone(self):
        """冻结整个主干网络（仅训练分类头）"""
        print("\n冻结主干网络...")

        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"可训练参数: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)\n")

    def get_trainable_params(self):
        """获取所有可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """
        统计模型参数

        Returns:
            dict: 包含总参数、可训练参数、冻结参数的字典
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_percentage': 100 * trainable / total if total > 0 else 0
        }
