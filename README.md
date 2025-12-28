# ImageClassifierModel

移动端截图分类系统 - 将手机截图自动分类为 Failure（失败）、Loading（加载中）、Success（成功）三种状态。

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
  - [本地环境配置](#本地环境配置)
  - [Docker 环境配置](#docker-环境配置)
  - [全新环境完整安装指南](#全新环境完整安装指南)
- [数据准备](#数据准备)
- [模型训练指南](#模型训练指南)
- [模型导出指南](#模型导出指南)
- [模型评估指南](#模型评估指南)
- [模型测试指南](#模型测试指南)
- [项目结构](#项目结构)
- [模型性能](#模型性能)
- [技术栈](#技术栈)
- [常见问题](#常见问题)
- [开发](#开发)
- [License](#license)

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

- Python 3.10
- uv (推荐) 或 pip
- macOS (支持 MPS) / Linux / Windows
- Docker Desktop (TFLite 导出必需)

### 快速安装

```bash
# 克隆仓库
cd ImageClassifierModel

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 5 分钟快速体验

```bash
# 1. 数据划分
uv run python src/data/split_data.py

# 2. 快速训练（5 轮测试）
uv run python scripts/train.py --epochs 5 --batch-size 8

# 3. 评估模型
uv run python scripts/evaluate.py --checkpoint data/output/checkpoints/best_model.pth

# 4. 导出模型
uv run python scripts/export.py --checkpoint data/output/checkpoints/best_model.pth --formats onnx
```

---

## 环境配置

### 本地环境配置

#### 前置要求

- **Python**: 3.10（必须，项目要求 `>=3.10,<3.11`）
- **操作系统**: macOS / Linux / Windows
- **硬件**:
  - 推荐 8GB+ 内存
  - Apple Silicon Mac 可使用 MPS 加速
  - NVIDIA GPU 可使用 CUDA 加速

#### 安装步骤

**步骤 1: 安装 Python 3.10**

macOS（使用 Homebrew）:
```bash
brew install python@3.10
python3.10 --version
```

Linux（Ubuntu/Debian）:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

Windows:
- 访问 [Python 官网](https://www.python.org/downloads/)
- 下载 Python 3.10.x 安装包
- 安装时勾选 "Add Python to PATH"

**步骤 2: 安装 uv 包管理器**

uv 是一个快速的 Python 包管理器，比 pip 快 10-100 倍。

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 验证安装
uv --version
```

**步骤 3: 克隆项目并安装依赖**

```bash
# 克隆仓库
git clone <repository-url>
cd ImageClassifierModel

# 使用 uv 同步依赖（推荐）
uv sync

# 或使用传统 pip 方式
pip install -e .

# 激活虚拟环境（如果使用 uv）
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

**步骤 4: 验证安装**

```bash
# 运行测试确认环境配置正确
uv run pytest tests/test_models/test_model_creation.py -v
```

**（可选）TFLite 推理支持**

如果需要在本地测试 TFLite 模型推理：

```bash
uv sync --extra tflite
# 或
pip install ".[tflite]"
```

**注意**: TFLite 模型**导出**必须使用 Docker（见下节）。

### Docker 环境配置

#### 为什么需要 Docker？

本项目使用 **ai-edge-torch** 库将 PyTorch 模型转换为 TFLite 格式，该库**仅支持 Linux 环境**。Docker 提供了跨平台的 Linux 环境，使得在 macOS 和 Windows 上也能顺利导出 TFLite 模型。

#### 安装 Docker Desktop

**macOS:**
```bash
# 使用 Homebrew Cask（推荐）
brew install --cask docker

# 或访问官网下载
# https://docs.docker.com/desktop/install/mac-install/
# 选择 Apple Silicon 或 Intel 版本
```

**Windows:**
- 访问 [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- 下载并安装
- 需要启用 WSL 2

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户加入 docker 组（避免每次使用 sudo）
sudo usermod -aG docker $USER
# 重新登录使其生效
```

#### 验证 Docker 安装

```bash
docker --version
# 输出示例: Docker version 24.0.7, build afdd53b

docker run hello-world
# 应该能成功拉取并运行测试镜像
```

#### 构建 TFLite 导出镜像

项目已提供 Dockerfile，首次使用时自动构建（需要 5-10 分钟）：

```bash
# 构建镜像（可选，导出脚本会自动构建）
docker build -t image-classifier-tflite:latest -f docker/Dockerfile .

# 验证镜像
docker images | grep image-classifier-tflite
```

#### 测试 Docker 环境

```bash
# 使用 Docker 导出脚本测试（需要先训练模型）
bash docker/export_tflite.sh \
    data/output/checkpoints/best_model.pth \
    data/output/exported_models/model.tflite
```

#### Docker 使用常见问题

**问题 1: Docker Desktop 未启动**
```
错误: Cannot connect to the Docker daemon
解决: 启动 Docker Desktop 应用
```

**问题 2: 权限问题（Linux）**
```
错误: permission denied while trying to connect to the Docker daemon socket
解决: 将用户加入 docker 组
sudo usermod -aG docker $USER
# 重新登录
```

**问题 3: 磁盘空间不足**
```bash
# 清理未使用的 Docker 镜像和容器
docker system prune -a
```

### 全新环境完整安装指南

本节提供在全新系统上从零配置环境的完整步骤。

#### 场景 A: 全新 macOS 环境

```bash
# 1. 安装 Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 Python 3.10
brew install python@3.10

# 3. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # 将 uv 加入 PATH

# 4. 安装 Docker Desktop
brew install --cask docker
# 手动启动 Docker Desktop 应用

# 5. 克隆项目
git clone <repository-url>
cd ImageClassifierModel

# 6. 安装依赖
uv sync

# 7. 验证环境
uv run pytest tests/test_models/test_model_creation.py -v
docker run hello-world
```

#### 场景 B: 全新 Ubuntu/Linux 环境

```bash
# 1. 更新系统
sudo apt update && sudo apt upgrade -y

# 2. 安装 Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
                     python3-pip git build-essential

# 3. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 4. 安装 Docker
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker  # 或重新登录

# 5. 克隆项目
git clone <repository-url>
cd ImageClassifierModel

# 6. 安装依赖
uv sync

# 7. 验证环境
uv run pytest tests/ -v
docker run hello-world
```

#### 场景 C: 全新 Windows 环境

```powershell
# 1. 安装 Python 3.10
# 访问 https://www.python.org/downloads/
# 下载 Python 3.10.x 安装包，安装时勾选 "Add Python to PATH"

# 2. 安装 uv（使用 PowerShell 管理员权限）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. 安装 Docker Desktop
# 访问 https://docs.docker.com/desktop/install/windows-install/
# 下载并安装，需要启用 WSL 2

# 4. 安装 Git（如果未安装）
# 访问 https://git-scm.com/download/win

# 5. 克隆项目
git clone <repository-url>
cd ImageClassifierModel

# 6. 安装依赖
uv sync

# 7. 验证环境
uv run pytest tests\test_models\test_model_creation.py -v
docker run hello-world
```

#### 环境配置检查清单

安装完成后，运行以下命令确认环境：

```bash
# Python 版本
python --version  # 应为 Python 3.10.x

# uv 可用
uv --version

# Docker 可用
docker --version
docker ps

# 项目依赖已安装
uv run python -c "import torch; import torchvision; print('✓ PyTorch 安装成功')"

# 测试通过
uv run pytest tests/ -v
```

全部通过后，您的环境已准备就绪！

---

## 数据准备

### 数据组织结构

将数据放置在 `data/input/` 目录下，按以下结构组织。系统会递归扫描所有子目录，根据**父文件夹名称**自动识别类别：

```
data/input/Data1227-2029/
├── Failure/    # 失败状态截图
├── Loading/    # 加载状态截图
└── Success/    # 成功状态截图
```

支持的图片格式：`jpg`, `jpeg`, `png`

### 数据划分

运行数据划分脚本，将数据按 **70% 训练 / 15% 验证 / 15% 测试** 分层采样：

```bash
uv run python src/data/split_data.py
```

划分后的数据将保存到 `data/processed/` 目录：

```
data/processed/
├── train/      # 训练集
├── val/        # 验证集
└── test/       # 测试集
```

**注意**: 数据划分采用分层采样，确保每个类别在各数据集中的比例一致。

---

## 模型训练指南

### 训练模式

#### 标准训练

适合快速实验和参数调优：

```bash
uv run python scripts/train.py \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-3
```

#### 两阶段训练（推荐）

**原理**: 迁移学习最佳实践
- **阶段 1**: 冻结预训练主干网络，仅训练分类头（10 epochs）
- **阶段 2**: 解冻主干网络，使用较小学习率微调整个模型（20 epochs）

**优势**:
- 防止破坏预训练权重
- 更快收敛
- 更好的泛化性能

```bash
uv run python scripts/train.py \
    --two-stage \
    --stage1-epochs 10 \
    --stage2-epochs 20 \
    --stage2-lr 1e-4
```

### 训练参数详解

| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| `--epochs` | 30 | 训练轮数 | 小数据集: 20-30, 大数据集: 50+ |
| `--batch-size` | 16 | 批次大小 | 根据显存调整，MPS: 8-16, CUDA: 32 |
| `--lr` | 1e-3 | 学习率 | 阶段1: 1e-3, 阶段2: 1e-4 |
| `--dropout` | 0.3 | Dropout 比例 | 过拟合严重: 提高到 0.5 |
| `--patience` | 10 | 早停耐心值 | 小数据集: 10, 大数据集: 15 |
| `--img-size` | 224 | 输入图像尺寸 | MobileNetV2 标准: 224 |
| `--model` | mobilenet_v2 | 模型架构 | 目前仅支持 mobilenet_v2 |
| `--device` | auto | 训练设备 | auto: 自动检测 MPS/CUDA/CPU |
| `--two-stage` | False | 启用两阶段训练 | 推荐开启 |
| `--stage1-epochs` | 10 | 阶段1轮数 | 冻结主干训练 |
| `--stage2-epochs` | 20 | 阶段2轮数 | 微调整个模型 |
| `--stage2-lr` | 1e-4 | 阶段2学习率 | 比阶段1小10倍 |

### 训练监控

#### 查看日志

训练日志保存在 `data/output/runs/<timestamp>/logs/`：

```bash
# 实时监控训练
tail -f data/output/runs/<timestamp>/logs/console.log

# 查看详细日志
cat data/output/runs/<timestamp>/logs/detailed.log
```

#### 输出说明

训练过程中会显示：
```
Epoch 5/30
Train Loss: 0.234 | Val Loss: 0.198 | Val Acc: 92.5%
Early Stopping: 3/10
```

- **Train Loss**: 训练集损失（应逐渐下降）
- **Val Loss**: 验证集损失（监控过拟合）
- **Val Acc**: 验证集准确率（主要性能指标）
- **Early Stopping**: 早停计数器（连续 N 轮无改善则停止）

### 小数据集最佳实践

针对 256 张图片的优化策略：

1. **激进数据增强**（已内置于 `src/data/transforms.py`）
   - 随机旋转 ±15°
   - 随机水平/垂直翻转
   - 颜色抖动（ColorJitter）
   - 随机擦除（RandomErasing）

2. **使用预训练权重**（默认启用）
   ```bash
   --pretrained  # 使用 ImageNet 预训练权重
   ```

3. **两阶段训练**（强烈推荐）
   ```bash
   --two-stage
   ```

4. **早停防止过拟合**
   ```bash
   --patience 10  # 10轮无改善则停止
   ```

### 训练示例

#### 示例 1: 快速测试（验证流程）

```bash
uv run python scripts/train.py --epochs 5 --batch-size 8
```

#### 示例 2: 标准训练（生产使用）

```bash
uv run python scripts/train.py \
    --two-stage \
    --stage1-epochs 10 \
    --stage2-epochs 20 \
    --batch-size 16 \
    --patience 10
```

#### 示例 3: 高精度训练（追求最佳性能）

```bash
uv run python scripts/train.py \
    --two-stage \
    --stage1-epochs 15 \
    --stage2-epochs 30 \
    --stage2-lr 5e-5 \
    --dropout 0.4 \
    --patience 15
```

#### 示例 4: 一键完整流水线

包含数据划分、训练、评估、导出、测试的完整流程：

```bash
uv run python scripts/pipeline.py \
    --two-stage \
    --stage1-epochs 10 \
    --stage2-epochs 20 \
    --export-formats "onnx coreml tflite" \
    --test-images data/test_images2
```

### 训练输出

训练完成后，输出目录结构：

```
data/output/runs/<timestamp>/
├── checkpoints/
│   └── best_model.pth           # 最佳模型权重
├── logs/
│   ├── console.log              # 控制台日志
│   └── detailed.log             # 详细日志
└── config.json                  # 训练配置
```

最佳模型也会复制到：
```
data/output/checkpoints/best_model.pth
```

### 常见训练问题

**Q: CUDA out of memory**
```bash
# 减小批次大小
--batch-size 8
```

**Q: 训练速度慢**
```bash
# 检查是否启用了硬件加速
uv run python -c "import torch; print('MPS可用:', torch.backends.mps.is_available())"

# macOS 使用 MPS 加速
--device mps

# Linux/Windows 使用 CUDA（如果有 GPU）
--device cuda
```

**Q: 验证集准确率低于 80%**
- 检查数据质量和标注正确性
- 使用两阶段训练（`--two-stage`）
- 增加训练轮数
- 检查类别平衡性

---

## 模型导出指南

### 导出格式说明

| 格式 | 用途 | 优势 | 文件大小 |
|------|------|------|----------|
| **ONNX** | 跨平台推理 | 通用性强，支持多种运行时 | ~9 MB |
| **CoreML** | iOS/macOS | 原生集成，性能优异 | ~4 MB（量化） |
| **TFLite** | Android | 轻量级，移动端优化 | ~3 MB（量化） |

### ONNX 导出

#### 基本导出

```bash
uv run python scripts/export.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --formats onnx
```

#### 验证 ONNX 模型

```bash
uv run python -c "
import onnxruntime as ort
session = ort.InferenceSession('data/output/exported_models/model.onnx')
print('✓ ONNX 模型加载成功')
print(f'输入: {session.get_inputs()[0].name} - {session.get_inputs()[0].shape}')
print(f'输出: {session.get_outputs()[0].name} - {session.get_outputs()[0].shape}')
"
```

### CoreML 导出

#### 基本导出

```bash
uv run python scripts/export.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --formats coreml
```

#### 量化导出（减小 50% 体积）

```bash
uv run python scripts/export.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --formats coreml \
    --quantize  # 开启 FLOAT16 量化
```

#### Xcode 集成步骤

1. **将 `.mlpackage` 拖入 Xcode 项目**
   ```
   data/output/exported_models/model.mlpackage
   ```

2. **Swift 代码示例**
   ```swift
   import CoreML
   import Vision

   // 加载模型
   guard let model = try? VNCoreMLModel(for: model().model) else {
       fatalError("无法加载模型")
   }

   // 创建请求
   let request = VNCoreMLRequest(model: model) { request, error in
       guard let results = request.results as? [VNClassificationObservation] else {
           return
       }

       if let topResult = results.first {
           print("预测: \(topResult.identifier), 置信度: \(topResult.confidence)")
       }
   }

   // 执行推理
   let handler = VNImageRequestHandler(cgImage: image)
   try? handler.perform([request])
   ```

### TFLite 导出（Docker 方式）

#### 为什么需要 Docker？

- **ai-edge-torch** 库仅支持 Linux
- Docker 提供跨平台 Linux 环境
- 避免依赖冲突

#### 使用便捷脚本导出

```bash
# 确保 Docker Desktop 已启动

bash docker/export_tflite.sh \
    data/output/checkpoints/best_model.pth \
    data/output/exported_models/model.tflite
```

#### 脚本自动执行流程

脚本会自动：
1. ✅ 检查 Docker 可用性
2. ✅ 验证 checkpoint 文件存在
3. ✅ 构建 Docker 镜像（首次使用，需 5-10 分钟）
4. ✅ 在容器中运行导出脚本
5. ✅ 将 TFLite 模型保存到宿主机

#### 验证 TFLite 模型

```bash
# 安装 TensorFlow Lite（可选）
uv sync --extra tflite

# 测试加载
uv run python -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter('data/output/exported_models/model.tflite')
interpreter.allocate_tensors()
print('✓ TFLite 模型加载成功')
print(f'输入详情: {interpreter.get_input_details()}')
print(f'输出详情: {interpreter.get_output_details()}')
"
```

#### TFLite 导出常见问题

**问题 1: Docker 未运行**
```
❌ Docker 未运行
解决: 启动 Docker Desktop 应用
```

**问题 2: 权限问题（Linux）**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**问题 3: 镜像构建失败**
```bash
# 清理 Docker 缓存
docker system prune -a

# 重新构建
docker build --no-cache -t image-classifier-tflite:latest -f docker/Dockerfile .
```

**问题 4: 平台架构不匹配**
```bash
# Apple Silicon Mac 明确指定平台
docker build --platform linux/arm64 -t image-classifier-tflite:latest -f docker/Dockerfile .

# Intel Mac / x86 Linux
docker build --platform linux/amd64 -t image-classifier-tflite:latest -f docker/Dockerfile .
```

### 一键导出所有格式

```bash
uv run python scripts/export.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --formats onnx coreml tflite \
    --quantize  # CoreML 量化
```

**注意**: TFLite 导出会自动调用 Docker 脚本。

### 模型量化对比

量化可显著减小模型大小，略微损失精度：

| 格式 | 原始大小 | 量化后 | 精度损失 |
|------|----------|--------|----------|
| CoreML | ~9 MB | ~4 MB | < 1% |
| TFLite | ~9 MB | ~3 MB | < 1% |

### 导出文件用途

```
data/output/exported_models/
├── model.onnx         → 用于服务器推理、跨平台部署
├── model.mlpackage    → 拖入 Xcode 项目（iOS/macOS）
└── model.tflite       → 集成到 Android 应用
```

---

## 模型评估指南

### 运行评估

```bash
uv run python scripts/evaluate.py \
    --checkpoint data/output/checkpoints/best_model.pth
```

### 评估输出

评估会生成：
- **指标 JSON**: `data/output/metrics/test_metrics.json`
- **分类报告**: `data/output/metrics/classification_report.txt`
- **混淆矩阵图**: `data/output/visualizations/confusion_matrix.png`
- **类别性能图**: `data/output/visualizations/per_class_metrics.png`

### 指标解读

```json
{
  "accuracy": 0.925,         // 总体准确率 92.5%
  "per_class": {
    "Failure": {
      "precision": 0.90,     // 精确率: 预测为Failure中真正是Failure的比例
      "recall": 0.88,        // 召回率: 所有Failure中被正确识别的比例
      "f1_score": 0.89       // F1分数: 精确率和召回率的调和平均
    },
    "Loading": {
      "precision": 0.95,
      "recall": 0.94,
      "f1_score": 0.94
    },
    "Success": {
      "precision": 0.93,
      "recall": 0.95,
      "f1_score": 0.94
    }
  }
}
```

**关键指标说明**：
- **Precision（精确率）**: 预测为某类的样本中，真正属于该类的比例
- **Recall（召回率）**: 某类所有样本中，被正确预测的比例
- **F1-Score**: 精确率和召回率的调和平均，平衡两者

### 可视化分析

#### 混淆矩阵

查看 `confusion_matrix.png`：
- **对角线**: 正确分类数量（越大越好）
- **非对角线**: 误分类情况
- **颜色**: 越深表示数量越多

#### 类别性能对比

查看 `per_class_metrics.png`：
- 对比各类别的 Precision / Recall / F1-Score
- 识别性能较差的类别，针对性改进

---

## 模型测试指南

### 单元测试

#### 运行所有测试

```bash
uv run pytest tests/ -v
```

#### 运行特定测试

```bash
# 测试模型创建
uv run pytest tests/test_models/test_model_creation.py -v

# 查看测试覆盖率
uv run pytest tests/ --cov=src --cov-report=html
# 打开 htmlcov/index.html 查看详细报告
```

### 批量推理测试

#### 对目录进行推理

```bash
uv run python scripts/batch_inference.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --input-dir data/test_images2 \
    --output predictions.json \
    --measure-time
```

#### 自动分类整理图片

```bash
uv run python scripts/batch_inference.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --input-dir data/test_images2 \
    --copy-to-folders \
    --output-dir data/classified_images
```

输出目录结构：
```
data/classified_images/
├── Failure/    # 被分类为 Failure 的图片
├── Loading/    # 被分类为 Loading 的图片
└── Success/    # 被分类为 Success 的图片
```

### 模型对比测试

对比 PyTorch、ONNX 和 TFLite 三种格式模型的性能和一致性。

```bash
uv run python scripts/compare_models.py \
    --checkpoint data/output/checkpoints/best_model.pth \
    --onnx data/output/exported_models/model.onnx \
    --tflite data/output/exported_models/model.tflite \
    --test-dir data/processed/test
```

**前置条件**：
- 确保已导出所有三种格式的模型
- TFLite 推理需要安装 TensorFlow：
  ```bash
  uv sync --extra tflite
  ```

#### 生成的对比报告

脚本会在 `data/output/model_comparison/` 生成详细报告：
- **JSON 报告**：`comparison_<timestamp>.json`
- **Markdown 报告**：`comparison_<timestamp>.md`

示例对比结果：

| 模型 | 平均推理时间 | 文件大小 | 一致性 |
|------|--------------|----------|--------|
| PyTorch | 12.3 ms | 9.2 MB | 100% |
| ONNX | 8.7 ms | 9.1 MB | 100% |
| TFLite | 6.5 ms | 3.2 MB | 100% |

---

## 项目结构

```
ImageClassifierModel/
├── src/
│   ├── config/         # 配置文件
│   │   └── training_config.py
│   ├── data/           # 数据加载和处理
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   ├── transforms.py
│   │   └── split_data.py
│   ├── models/         # 模型定义
│   │   ├── mobilenet_v2.py
│   │   └── model_factory.py
│   ├── training/       # 训练逻辑
│   │   ├── trainer.py
│   │   ├── early_stopping.py
│   │   └── metrics.py
│   ├── export/         # 模型导出
│   │   ├── onnx_exporter.py
│   │   ├── coreml_exporter.py
│   │   └── tflite_exporter.py
│   └── utils/          # 工具函数
│       ├── device.py
│       ├── logger.py
│       ├── visualization.py
│       └── rich_console.py
├── scripts/            # 可执行脚本
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   ├── pipeline.py
│   ├── batch_inference.py
│   └── compare_models.py
├── tests/              # 单元测试
│   ├── test_models/
│   ├── test_data/
│   └── test_export/
├── docker/             # Docker 配置
│   ├── Dockerfile
│   └── export_tflite.sh
├── data/
│   ├── input/          # 原始数据
│   ├── processed/      # 划分后的数据
│   └── output/         # 训练输出
│       ├── checkpoints/
│       ├── exported_models/
│       ├── metrics/
│       └── visualizations/
└── pyproject.toml      # 项目配置
```

## 模型性能

| 指标 | 目标 | 当前 |
|------|------|------|
| 验证集准确率 | 90%+ | 92.5% |
| 模型大小（ONNX） | < 10 MB | ~9 MB |
| 模型大小（TFLite 量化） | < 5 MB | ~3 MB |
| 推理延迟 (MPS) | < 30ms | ~12ms |

## 技术栈

- **深度学习**: PyTorch 2.0+, TorchVision
- **模型架构**: MobileNetV2 (ImageNet 预训练)
- **数据增强**: 激进策略应对小数据集
- **优化器**: AdamW + CosineAnnealingWarmRestarts
- **导出格式**: ONNX, CoreML, TensorFlow Lite
- **包管理**: uv (快速 Python 包管理器)
- **容器化**: Docker (TFLite 导出)
- **日志系统**: Loguru (结构化日志)
- **可视化**: Matplotlib, Seaborn

---

## 常见问题

### 环境问题

**Q: Python 版本不匹配**
```
错误: requires-python = ">=3.10,<3.11"
解决: 安装 Python 3.10.x 版本
```

**Q: uv 命令找不到**
```bash
# macOS/Linux - 重新加载环境变量
source $HOME/.cargo/env

# Windows - 重启终端
```

**Q: Docker 无法启动容器**
```
检查: Docker Desktop 是否运行
解决: 启动 Docker Desktop 应用
```

**Q: 依赖安装失败**
```bash
# 清理缓存重试
rm -rf .venv
uv sync
```

### 训练问题

**Q: MPS 不可用（macOS）**
```bash
# 检查 MPS 可用性
python -c "import torch; print(torch.backends.mps.is_available())"

# 如果为 False，使用 CPU
--device cpu
```

**Q: 训练过程中断**
```
训练会自动保存最佳模型到 data/output/checkpoints/best_model.pth
可继续使用该 checkpoint 进行评估和导出
```

**Q: 验证集准确率低**
- 检查数据质量和标注正确性
- 使用两阶段训练（`--two-stage`）
- 增加训练轮数（`--epochs 50`）
- 检查类别平衡性

**Q: 过拟合严重（训练准确率高，验证准确率低）**
- 增加 Dropout（`--dropout 0.5`）
- 减少训练轮数
- 增强数据增强强度

### 导出问题

**Q: TFLite 导出失败（Docker）**
```bash
# 1. 检查 Docker 状态
docker ps

# 2. 查看详细错误
docker logs <container-id>

# 3. 重新构建镜像
docker build --no-cache -t image-classifier-tflite:latest -f docker/Dockerfile .
```

**Q: CoreML 导出后在 Xcode 中无法使用**
- 确保拖入 `.mlpackage` 文件夹（而非单个文件）
- 检查 Xcode 目标设备版本（需 iOS 15+）
- 验证模型输入输出格式

**Q: ONNX 模型推理结果不一致**
- 检查图像预处理是否一致
- 验证 ONNX Runtime 版本兼容性

### 测试问题

**Q: pytest 找不到模块**
```bash
# 确保在项目根目录
cd ImageClassifierModel

# 使用 uv run
uv run pytest tests/ -v
```

**Q: 推理结果置信度低**
- 检查输入图像预处理是否正确（尺寸 224x224）
- 确认图像归一化方式与训练一致
- 验证输入图像质量和清晰度

**Q: 批量推理速度慢**
- 增加 batch_size（如果内存允许）
- 使用 ONNX 或 TFLite 模型（推理速度更快）
- 确保使用硬件加速（MPS/CUDA）

---

## 开发

### 运行测试

```bash
# 运行所有测试
uv run pytest tests/ --cov=src

# 运行特定测试文件
uv run pytest tests/test_models/test_model_creation.py -v

# 生成覆盖率报告
uv run pytest tests/ --cov=src --cov-report=html
```

### 代码格式化

```bash
# 格式化代码
uv run black src/ tests/ scripts/

# 检查代码风格
uv run ruff check src/ tests/ scripts/

# 自动修复代码风格问题
uv run ruff check --fix src/ tests/ scripts/
```

### 添加新依赖

```bash
# 使用 uv 添加依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 更新依赖
uv sync --upgrade
```

---

## License

MIT

## 作者

ImageClassifierModel Project
