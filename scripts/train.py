"""训练主脚本"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.config.training_config import TrainingConfig
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.utils.device import print_device_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练图像分类模型')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练 epoch 数量 (默认: 30)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减 (默认: 1e-4)')

    # 模型参数
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2'],
                        help='模型架构 (默认: mobilenet_v2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='不使用预训练权重')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比例 (默认: 0.3)')

    # 数据参数
    parser.add_argument('--data-root', type=str, default='data/processed',
                        help='数据根目录 (默认: data/processed)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='图像尺寸 (默认: 224)')

    # 两阶段训练
    parser.add_argument('--two-stage', action='store_true', default=False,
                        help='使用两阶段训练（冻结主干 → 微调）')
    parser.add_argument('--stage1-epochs', type=int, default=10,
                        help='阶段1 epoch 数量 (默认: 10)')
    parser.add_argument('--stage2-epochs', type=int, default=20,
                        help='阶段2 epoch 数量 (默认: 20)')
    parser.add_argument('--stage2-lr', type=float, default=1e-4,
                        help='阶段2学习率 (默认: 1e-4)')
    parser.add_argument('--unfreeze-from', type=int, default=14,
                        help='从第几层开始解冻 (默认: 14)')

    # 其他
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='训练设备 (默认: auto)')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值 (默认: 10)')

    return parser.parse_args()


def train_single_stage(args):
    """单阶段训练（标准流程）"""
    # 创建配置
    config = TrainingConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        img_size=args.img_size,
        device=args.device,
        num_workers=args.num_workers,
        early_stopping_patience=args.patience,
        data_root=Path(args.data_root),
        pretrained=args.pretrained
    )

    # 打印设备信息
    print_device_info()

    # 创建数据加载器
    print("加载数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=config.img_size
    )

    # 创建模型
    model = create_model(
        config.model_name,
        num_classes=config.num_classes,
        pretrained=args.pretrained,
        dropout=config.dropout
    )

    # 创建训练器
    trainer = Trainer(model, config, train_loader, val_loader)

    # 开始训练
    history = trainer.train()

    return trainer, history


def train_two_stage(args):
    """两阶段训练（冻结主干 → 微调）"""
    # ========== 阶段 1: 冻结主干，仅训练分类头 ==========
    print("\n" + "🔥"*35)
    print("阶段 1: 冻结主干网络，训练分类头")
    print("🔥"*35 + "\n")

    # 创建配置（阶段1）
    config_stage1 = TrainingConfig(
        model_name=args.model,
        num_epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        img_size=args.img_size,
        device=args.device,
        num_workers=args.num_workers,
        early_stopping_patience=args.patience,
        data_root=Path(args.data_root),
        pretrained=args.pretrained
    )

    # 打印设备信息
    print_device_info()

    # 加载数据
    print("加载数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        config_stage1.data_root,
        batch_size=config_stage1.batch_size,
        num_workers=config_stage1.num_workers,
        img_size=config_stage1.img_size
    )

    # 创建模型（主干已冻结）
    model = create_model(
        config_stage1.model_name,
        num_classes=config_stage1.num_classes,
        pretrained=args.pretrained,
        dropout=config_stage1.dropout
    )

    # 训练阶段1
    trainer_stage1 = Trainer(model, config_stage1, train_loader, val_loader)
    history_stage1 = trainer_stage1.train()

    # ========== 阶段 2: 解冻主干，微调 ==========
    print("\n" + "🚀"*35)
    print("阶段 2: 解冻主干网络，微调模型")
    print("🚀"*35 + "\n")

    # 解冻主干网络
    model.unfreeze_backbone(unfreeze_from_layer=args.unfreeze_from)

    # 创建配置（阶段2，降低学习率）
    config_stage2 = TrainingConfig(
        model_name=args.model,
        num_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        learning_rate=args.stage2_lr,  # 降低学习率
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        img_size=args.img_size,
        device=args.device,
        num_workers=args.num_workers,
        early_stopping_patience=args.patience,
        data_root=Path(args.data_root),
        pretrained=False  # 不重新加载预训练权重
    )

    # 训练阶段2（使用解冻后的模型）
    trainer_stage2 = Trainer(model, config_stage2, train_loader, val_loader)

    # 继承阶段1的历史
    trainer_stage2.history['train_loss'] = history_stage1['train_loss']
    trainer_stage2.history['val_loss'] = history_stage1['val_loss']
    trainer_stage2.history['val_acc'] = history_stage1['val_acc']
    trainer_stage2.history['learning_rate'] = history_stage1['learning_rate']

    # 继续训练
    history_stage2 = trainer_stage2.train()

    return trainer_stage2, history_stage2


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*70)
    print("ImageClassifierModel - 模型训练")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"Epochs: {args.epochs if not args.two_stage else f'{args.stage1_epochs} + {args.stage2_epochs}'}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"预训练: {'是' if args.pretrained else '否'}")
    print(f"两阶段训练: {'是' if args.two_stage else '否'}")
    print("="*70 + "\n")

    try:
        if args.two_stage:
            trainer, history = train_two_stage(args)
        else:
            trainer, history = train_single_stage(args)

        print("\n✅ 训练成功完成!")
        print(f"最佳模型已保存到: {trainer.config.checkpoint_dir / 'best_model.pth'}")

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
