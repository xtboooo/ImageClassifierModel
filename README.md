# ImageClassifierModel

移动端截图分类系统 - 将手机截图自动分类为 Failure（失败）、Loading（加载中）、Success（成功）三种状态。

## 项目概述

本项目使用 PyTorch 训练一个轻量级的图像分类模型，专门用于识别移动应用的不同状态。模型基于 MobileNetV2 架构，经过优化后可部署到 iOS 和 Android 设备。

**特点**:
- 🚀 轻量级模型（< 10 MB）
- 📱 支持 iOS (CoreML) 和 Android (TFLite) 部署
- 🎯 针对小数据集优化（256 张训练图片）
- ⚡ 使用 Apple Silicon MPS 加速训练
- 🔧 完整的训练、评估和导出流程

## 快速开始

### 环境要求

- Python 3.10-3.12
- uv (推荐) 或 pip
- macOS (支持 MPS) / Linux / Windows

### 安装

```bash
# 克隆仓库
cd ImageClassifierModel

# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

### 数据准备

将数据放置在 `data/input/Data1227-2029/` 目录下，按以下结构组织：

```
data/input/Data1227-2029/
├── Failure/    # 失败状态截图
├── Loading/    # 加载状态截图
└── Success/    # 成功状态截图
```

运行数据划分脚本（将数据划分为 70% 训练 / 15% 验证 / 15% 测试）：

```bash
uv run python src/data/split_data.py
```

划分后的数据将保存到 `data/processed/` 目录。

### 训练模型

```bash
# 基础训练（使用默认参数）
uv run python scripts/train.py

# 自定义参数
uv run python scripts/train.py --epochs 30 --batch-size 16

# 两阶段训练（推荐：先冻结主干，再微调）
uv run python scripts/train.py --two-stage --stage1-epochs 10 --stage2-epochs 20
```

训练过程中会自动：
- 保存最佳模型到 `data/output/checkpoints/best_model.pth`
- 使用早停防止过拟合
- 在验证集上评估性能
- 支持 Apple Silicon MPS 加速

### 评估模型

```bash
uv run python scripts/evaluate.py --checkpoint data/output/checkpoints/best_model.pth
```

### 导出模型

```bash
# 导出所有格式（ONNX, CoreML, TFLite）
uv run python scripts/export.py --checkpoint best_model.pth --formats onnx coreml

# 仅导出 ONNX
uv run python scripts/export.py --checkpoint best_model.pth --formats onnx
```

## 项目结构

```
ImageClassifierModel/
├── src/
│   ├── config/         # 配置文件
│   ├── data/           # 数据加载和处理
│   ├── models/         # 模型定义
│   ├── training/       # 训练逻辑
│   ├── export/         # 模型导出
│   └── utils/          # 工具函数
├── scripts/            # 可执行脚本
├── tests/              # 单元测试
├── data/
│   ├── input/          # 原始数据
│   ├── processed/      # 划分后的数据
│   └── output/         # 训练输出
└── pyproject.toml      # 项目配置
```

## 模型性能

| 指标 | 目标 |
|------|------|
| 验证集准确率 | 90%+ |
| 模型大小 | < 10 MB |
| 推理延迟 (MPS) | < 30ms |

## 技术栈

- **深度学习**: PyTorch, TorchVision
- **模型架构**: MobileNetV2 (ImageNet 预训练)
- **数据增强**: 激进策略应对小数据集
- **优化器**: AdamW + 余弦退火调度
- **导出格式**: ONNX, CoreML, TensorFlow Lite

## 开发

```bash
# 运行测试
uv run pytest tests/ --cov=src

# 代码格式化
uv run black src/ tests/ scripts/

# 代码检查
uv run ruff check src/ tests/ scripts/
```

## License

MIT

## 作者

ImageClassifierModel Project
