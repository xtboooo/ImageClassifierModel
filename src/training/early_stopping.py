"""早停机制"""


class EarlyStopping:
    """
    早停回调 - 在验证损失不再改善时停止训练

    当验证损失在 patience 个 epoch 内没有改善至少 min_delta 时触发早停
    """

    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        """
        Args:
            patience: 容忍的 epoch 数量
            min_delta: 最小改善量（小于此值视为没有改善）
            mode: 'min' 表示监控指标越小越好（如loss），'max' 表示越大越好（如accuracy）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value):
        """
        检查是否应该早停

        Args:
            current_value: 当前监控的指标值（如验证损失）

        Returns:
            bool: 如果应该早停返回 True，否则返回 False
        """
        if self.best_value is None:
            # 第一次调用，直接保存
            self.best_value = current_value
            return False

        # 判断是否改善
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:  # mode == 'max'
            improved = current_value > self.best_value + self.min_delta

        if improved:
            # 有改善，重置计数器并更新最佳值
            self.best_value = current_value
            self.counter = 0
            self.early_stop = False
        else:
            # 没有改善，计数器+1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def state_dict(self):
        """返回状态字典（用于保存）"""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'counter': self.counter,
            'best_value': self.best_value,
            'early_stop': self.early_stop
        }

    def load_state_dict(self, state_dict):
        """从状态字典恢复（用于加载）"""
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        self.counter = state_dict['counter']
        self.best_value = state_dict['best_value']
        self.early_stop = state_dict['early_stop']
