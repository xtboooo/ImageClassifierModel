"""评估指标计算"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, class_names=None):
        """
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names or ['Failure', 'Loading', 'Success']

    def compute_metrics(self, y_true, y_pred):
        """
        计算所有评估指标

        Args:
            y_true: 真实标签 (numpy array)
            y_pred: 预测标签 (numpy array)

        Returns:
            dict: 包含所有指标的字典
        """
        # 确保是numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 总体准确率
        accuracy = accuracy_score(y_true, y_pred)

        # 每类的精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names)), zero_division=0
        )

        # 宏平均（Macro average）- 每类等权重
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # 加权平均（Weighted average）- 按样本数量加权
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))

        # 组装结果
        metrics = {
            'accuracy': accuracy,
            'per_class': {},
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            },
            'confusion_matrix': cm.tolist(),  # 转为列表便于JSON序列化
            'confusion_matrix_array': cm  # numpy数组用于可视化
        }

        # 每类指标
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        return metrics

    def print_metrics(self, metrics):
        """
        打印指标报告

        Args:
            metrics: compute_metrics 返回的指标字典
        """
        print("\n" + "="*70)
        print("评估指标报告")
        print("="*70)

        # 总体准确率
        print(f"\n总体准确率: {metrics['accuracy']*100:.2f}%")

        # 每类指标
        print(f"\n{'类别':<15} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>10}")
        print("-"*70)

        for class_name in self.class_names:
            if class_name in metrics['per_class']:
                stats = metrics['per_class'][class_name]
                print(f"{class_name:<15} "
                      f"{stats['precision']*100:>9.2f}% "
                      f"{stats['recall']*100:>9.2f}% "
                      f"{stats['f1_score']*100:>9.2f}% "
                      f"{stats['support']:>10}")

        # 平均指标
        print("-"*70)
        print(f"{'宏平均':<15} "
              f"{metrics['macro_avg']['precision']*100:>9.2f}% "
              f"{metrics['macro_avg']['recall']*100:>9.2f}% "
              f"{metrics['macro_avg']['f1_score']*100:>9.2f}%")

        print(f"{'加权平均':<15} "
              f"{metrics['weighted_avg']['precision']*100:>9.2f}% "
              f"{metrics['weighted_avg']['recall']*100:>9.2f}% "
              f"{metrics['weighted_avg']['f1_score']*100:>9.2f}%")

        print("="*70 + "\n")

    def get_classification_report(self, y_true, y_pred):
        """
        获取详细的分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            str: 格式化的分类报告
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
