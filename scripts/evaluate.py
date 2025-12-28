"""评估模型性能"""
import argparse
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm

from src.config.training_config import TrainingConfig
from src.data.dataloader import create_dataloaders
from src.models.model_factory import load_model_from_checkpoint
from src.training.metrics import MetricsCalculator
from src.utils.visualization import Visualizer
from src.utils.device import get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估训练好的模型')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data-root', type=str, default='data/training_data',
                        help='数据根目录')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='评估设备')

    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试集 DataLoader
        device: 设备

    Returns:
        tuple: (y_true, y_pred, y_probs)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    print("\n开始评估...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # 预测
            _, predicted = outputs.max(1)

            # 收集结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*70)
    print("模型评估")
    print("="*70)
    print(f"检查点: {args.checkpoint}")
    print(f"数据目录: {args.data_root}")
    print("="*70 + "\n")

    # 设备
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
        print(f"使用设备: {device}\n")

    # 加载模型
    print("加载模型...")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint)

    # 加载数据
    print("\n加载测试集...")
    _, _, test_loader = create_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=224
    )

    # 评估
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)

    # 计算指标
    print("\n计算评估指标...")
    class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']
    metrics_calculator = MetricsCalculator(class_names)
    metrics = metrics_calculator.compute_metrics(y_true, y_pred)

    # 打印指标
    metrics_calculator.print_metrics(metrics)

    # 保存结果
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / 'metrics'
    vis_dir = output_dir / 'visualizations'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标到JSON
    metrics_json = {k: v for k, v in metrics.items() if k != 'confusion_matrix_array'}
    metrics_file = metrics_dir / 'test_metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f"指标已保存到: {metrics_file}")

    # 保存详细报告
    report_file = metrics_dir / 'classification_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        report = metrics_calculator.get_classification_report(y_true, y_pred)
        f.write(report)
    print(f"分类报告已保存到: {report_file}")

    # 生成可视化
    print("\n生成可视化图表...")

    # 混淆矩阵
    Visualizer.plot_confusion_matrix(
        metrics['confusion_matrix_array'],
        class_names,
        save_path=vis_dir / 'test_confusion_matrix.png'
    )

    # 每类指标
    Visualizer.plot_per_class_metrics(
        metrics,
        save_path=vis_dir / 'test_per_class_metrics.png'
    )

    # 如果有训练历史，也绘制训练曲线
    if 'history' in checkpoint:
        Visualizer.plot_training_history(
            checkpoint['history'],
            save_path=vis_dir / 'training_history.png'
        )

    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)
    print(f"总体准确率: {metrics['accuracy']*100:.2f}%")
    print(f"宏平均 F1: {metrics['macro_avg']['f1_score']*100:.2f}%")
    print(f"结果已保存到: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
