"""训练器 - 核心训练逻辑"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

from .early_stopping import EarlyStopping
from ..utils.device import get_device
from ..utils.logger import logger
from ..utils.rich_console import RichProgressManager


class Trainer:
    """
    训练器主类

    负责:
    - 训练循环
    - 验证
    - 学习率调度
    - 早停
    - 模型保存
    - 训练历史记录
    """

    def __init__(self, model, config, train_loader, val_loader):
        """
        Args:
            model: PyTorch 模型
            config: TrainingConfig 配置对象
            train_loader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 设备设置（自动检测 MPS/CUDA/CPU）
        if config.device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(config.device)
        self.model.to(self.device)

        # 损失函数（带标签平滑）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器（余弦退火）
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 初始周期
            T_mult=2  # 周期倍增因子
        )

        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode='min'  # 监控验证损失（越小越好）
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # 最佳指标
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self, epoch_num=None):
        """
        训练一个 epoch

        Args:
            epoch_num: 当前 epoch 编号（用于进度条显示）

        Returns:
            tuple: (平均训练损失, 平均训练准确率)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        desc = f"训练 Epoch {epoch_num}" if epoch_num else "训练"

        with RichProgressManager() as progress:
            task = progress.add_task(desc, total=len(self.train_loader))

            for images, labels in self.train_loader:
                # 数据移到设备
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 更新参数
                self.optimizer.step()

                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                progress.update(desc, advance=1)

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def validate(self):
        """
        验证模型

        Returns:
            tuple: (验证损失, 验证准确率)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            with RichProgressManager() as progress:
                task = progress.add_task("验证", total=len(self.val_loader))

                for images, labels in self.val_loader:
                    # 数据移到设备
                    images, labels = images.to(self.device), labels.to(self.device)

                    # 前向传播
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # 统计
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # 更新进度条
                    progress.update("验证", advance=1)

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, num_epochs=None):
        """
        完整训练流程

        Args:
            num_epochs: epoch 数量，如果为None则使用config中的值

        Returns:
            dict: 训练历史
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        logger.info("开始训练",
                    epochs=num_epochs,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    device=str(self.device),
                    early_stopping_patience=self.config.early_stopping_patience)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch(epoch_num=epoch+1)

            # 验证
            val_loss, val_acc = self.validate()

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # 记录指标
            logger.info("Epoch 完成",
                        epoch=f"{epoch+1}/{num_epochs}",
                        train_loss=f"{train_loss:.4f}",
                        train_acc=f"{train_acc:.2f}%",
                        val_loss=f"{val_loss:.4f}",
                        val_acc=f"{val_acc:.2f}%",
                        learning_rate=f"{current_lr:.6f}")

            # 保存最佳模型（基于验证准确率）
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                logger.success(f"保存最佳模型 (验证准确率: {val_acc:.2f}%)")

            # 早停检查
            if self.early_stopping(val_loss):
                logger.warning(
                    f"早停触发！验证损失在 {self.config.early_stopping_patience} 个 epoch 内未改善",
                    best_val_acc=f"{self.best_val_acc:.2f}%"
                )
                break

        logger.success("训练完成",
                       best_val_acc=f"{self.best_val_acc:.2f}%",
                       best_val_loss=f"{self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename):
        """
        保存模型检查点

        Args:
            filename: 文件名
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'early_stopping_state': self.early_stopping.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点继续训练

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.early_stopping.load_state_dict(checkpoint['early_stopping_state'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.success("已从检查点加载",
                       checkpoint_path=str(checkpoint_path),
                       best_val_acc=f"{self.best_val_acc:.2f}%")
