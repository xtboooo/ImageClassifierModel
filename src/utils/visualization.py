"""训练可视化工具"""
import matplotlib
# 使用非交互式后端，避免弹出窗口和卡顿
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from src.utils.logger import logger

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class Visualizer:
    """训练和评估可视化工具"""

    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        绘制训练历史曲线

        Args:
            history: 训练历史字典（包含train_loss, val_loss, val_acc等）
            save_path: 保存路径，如果为None则仅显示
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 损失曲线
        ax1 = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2 = axes[1]
        ax2.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 学习率曲线
        ax3 = axes[2]
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            ax3.plot(epochs, history['learning_rate'], 'm-', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练曲线已保存到: {save_path}")

        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None, normalize=True):
        """
        绘制混淆矩阵

        Args:
            cm: 混淆矩阵（numpy array）
            class_names: 类别名称列表
            save_path: 保存路径
            normalize: 是否归一化
        """
        plt.figure(figsize=(10, 8))

        # 归一化
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Confusion Matrix (Normalized)'
        else:
            cm_normalized = cm
            fmt = 'd'
            title = 'Confusion Matrix'

        # 绘制热图
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            square=True,
            linewidths=1,
            linecolor='gray'
        )

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")

        plt.close()

    @staticmethod
    def plot_per_class_metrics(metrics, save_path=None):
        """
        绘制每类指标对比图

        Args:
            metrics: 指标字典（来自MetricsCalculator）
            save_path: 保存路径
        """
        class_names = list(metrics['per_class'].keys())
        precision = [metrics['per_class'][cls]['precision'] for cls in class_names]
        recall = [metrics['per_class'][cls]['recall'] for cls in class_names]
        f1 = [metrics['per_class'][cls]['f1_score'] for cls in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon', edgecolor='black')

        # 添加数值标签
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=10)

        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)

        ax.set_xlabel('Class', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Per-Class Metrics Comparison', fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"每类指标图已保存到: {save_path}")

        plt.close()

    @staticmethod
    def plot_all(history, metrics, output_dir):
        """
        生成所有可视化图表

        Args:
            history: 训练历史
            metrics: 评估指标
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("生成可视化图表...")

        # 训练历史
        Visualizer.plot_training_history(
            history,
            save_path=output_dir / 'training_history.png'
        )

        # 混淆矩阵
        if 'confusion_matrix_array' in metrics:
            class_names = list(metrics['per_class'].keys())
            Visualizer.plot_confusion_matrix(
                metrics['confusion_matrix_array'],
                class_names,
                save_path=output_dir / 'confusion_matrix.png'
            )

        # 每类指标
        Visualizer.plot_per_class_metrics(
            metrics,
            save_path=output_dir / 'per_class_metrics.png'
        )

        logger.info(f"所有图表已保存到: {output_dir}")
