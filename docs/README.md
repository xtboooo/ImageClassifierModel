# ImageClassifierModel 文档中心

欢迎使用ImageClassifierModel！本项目提供完整的移动端图片分类解决方案，支持Android和iOS平台。

## 📚 文档导航

### 1. [部署指南 (DEPLOYMENT_GUIDE.md)](./DEPLOYMENT_GUIDE.md)
**适合移动端开发者**

详细介绍如何在Android和iOS上集成和使用模型：
- ✅ Android使用ONNX Runtime（推荐）
- ✅ Android使用TensorFlow Lite
- ✅ iOS使用CoreML（原生支持）
- ✅ 完整的代码示例（Kotlin/Swift/Objective-C）
- ✅ 性能优化技巧
- ✅ 常见问题解答

### 2. [模型API规格 (MODEL_API.md)](./MODEL_API.md)
**适合算法工程师和集成开发者**

完整的模型技术规格说明：
- 📊 输入输出格式详解
- 🔄 预处理流程（ImageNet归一化）
- 🎯 后处理流程（Softmax）
- 💻 Python/NumPy代码示例
- ⚡ 性能测试结果

### 3. [项目说明 (../CLAUDE.md)](../CLAUDE.md)
**项目概览和开发指南**

- 项目架构
- 开发环境设置
- 训练命令
- 数据集信息

---

## 🚀 快速开始

### 一、模型训练（数据科学家）

#### 1. 完整训练流程（一键执行）

```bash
# 推荐：使用两阶段训练 + 模型对比
uv run python scripts/pipeline.py \
  --two-stage \
  --stage1-epochs 15 \
  --stage2-epochs 25 \
  --compare-models

# 输出目录：data/output/runs/YYYYMMDD_HHMMSS/
```

#### 2. 使用已有模型（跳过训练）

```bash
uv run python scripts/pipeline.py \
  --skip-train \
  --checkpoint data/output/checkpoints/best_model.pth \
  --compare-models
```

#### 3. 自定义运行名称

```bash
uv run python scripts/pipeline.py \
  --run-name "mobilenet_v2_final_v1" \
  --two-stage
```

### 二、模型部署（移动端开发者）

#### Android集成（推荐：ONNX）

```kotlin
// 1. 添加依赖到 build.gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'

// 2. 加载模型并推理
val classifier = ImageClassifier(context)
val (className, confidence) = classifier.classify(bitmap)

// 结果
Log.d("AI", "预测: $className, 置信度: ${confidence * 100}%")
```

完整代码请参考：[DEPLOYMENT_GUIDE.md - Android部署](./DEPLOYMENT_GUIDE.md#android部署)

#### iOS集成（推荐：CoreML）

```swift
// 1. 将 model.mlpackage 拖入Xcode项目

// 2. 推理代码
let classifier = ImageClassifier()
classifier.classify(image: uiImage) { className, confidence in
    print("预测: \(className), 置信度: \(confidence * 100)%")
}
```

完整代码请参考：[DEPLOYMENT_GUIDE.md - iOS部署](./DEPLOYMENT_GUIDE.md#ios部署)

---

## 📁 输出文件结构

运行一次完整流程后，所有相关文件会按时间戳组织在一起：

```
data/output/runs/20251227_135042/
├── config.json                        # 运行配置
├── checkpoints/
│   └── best_model.pth                 # PyTorch模型 (29.1 MB)
├── exported_models/
│   ├── model.onnx                     # ONNX模型 (11.3 MB) ⚡
│   ├── model.onnx.data
│   └── model.mlpackage/               # CoreML模型 (10.8 MB)
├── metrics/
│   ├── test_metrics.json              # 评估指标
│   └── classification_report.txt      # 分类报告
├── visualizations/
│   ├── confusion_matrix.png           # 混淆矩阵
│   ├── per_class_metrics.png          # 各类别指标
│   └── training_history.png           # 训练曲线
├── test_results/
│   ├── predictions.json               # 测试图片预测结果
│   ├── predictions.csv
│   └── summary.txt                    # 分类统计摘要
├── model_comparison/
│   ├── comparison.json                # 模型对比数据
│   └── comparison.md                  # 对比报告
└── run_summary.md                     # 📄 完整运行总结报告
```

---

## 🎯 模型性能

### 评估指标（测试集）

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| **Failure** | 100.00% | 86.67% | 92.86% |
| **Loading** | 82.35% | 100.00% | 90.32% |
| **Success** | 100.00% | 92.86% | 96.30% |
| **总体准确率** | - | - | **93.02%** |

### 推理性能（43张测试图片）

| 模型格式 | 文件大小 | 平均耗时 | 速度提升 | 预测一致性 |
|---------|---------|---------|---------|-----------|
| PyTorch | 29.1 MB | 44.37 ms | 基准 | - |
| **ONNX** | 11.3 MB | **2.75 ms** | **16.1倍** | 100% |
| CoreML | 10.8 MB | ~3.1 ms | 14.3倍 | 100% |

**关键亮点**：
- ✅ ONNX模型推理速度提升**16倍**
- ✅ 文件大小减少**62%**
- ✅ 与原始模型保持**100%预测一致性**

---

## 🔧 常用命令

### 训练相关

```bash
# 标准训练
uv run python scripts/train.py --epochs 30

# 两阶段训练（推荐）
uv run python scripts/train.py --two-stage \
  --stage1-epochs 15 \
  --stage2-epochs 25

# 继续训练
uv run python scripts/train.py --resume data/output/checkpoints/best_model.pth
```

### 评估相关

```bash
# 评估模型
uv run python scripts/evaluate.py \
  --checkpoint data/output/checkpoints/best_model.pth

# 批量推理
uv run python scripts/batch_inference.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --input-dir data/test_images/ \
  --measure-time
```

### 导出相关

```bash
# 导出ONNX（Android推荐）
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --format onnx

# 导出CoreML（iOS推荐）
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --format coreml

# 导出TFLite
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --format tflite \
  --quantize  # 可选：INT8量化
```

### 对比相关

```bash
# 对比不同格式模型性能
uv run python scripts/compare_models.py \
  --pytorch-checkpoint data/output/checkpoints/best_model.pth \
  --onnx-model data/output/exported_models/model.onnx \
  --test-dir data/processed/test/
```

---

## 💡 使用建议

### 对于数据科学家

1. **首次训练**：使用 `pipeline.py --two-stage` 获得最佳效果
2. **实验对比**：使用 `--run-name` 给每次实验命名
3. **性能分析**：查看 `run_summary.md` 获取完整报告
4. **模型选择**：查看 `model_comparison/` 对比不同格式

### 对于Android开发者

1. **推荐格式**：ONNX（速度最快，集成简单）
2. **依赖管理**：`com.microsoft.onnxruntime:onnxruntime-android`
3. **加速优化**：启用NNAPI加速
4. **内存优化**：在后台线程初始化模型

### 对于iOS开发者

1. **推荐格式**：CoreML（Apple原生支持）
2. **集成方式**：直接将 `.mlpackage` 拖入Xcode
3. **自动优化**：CoreML自动选择Neural Engine/GPU/CPU
4. **异步推理**：使用GCD避免阻塞主线程

---

## 📊 数据集信息

### 训练数据

- **路径**: `data/input/data1226/`
- **总图片**: 256张已标注移动端截图
- **类别分布**:
  - Failure: 95张（操作失败状态）
  - Loading: 68张（加载/等待状态）
  - Success: 93张（操作成功状态）

### 测试数据

- **路径**: `data/test_images/`
- **总图片**: 275张未分类移动端截图
- **用途**: 批量推理测试

### 数据划分

流水线脚本会自动划分数据（如果不存在）：
- **训练集**: 70% (190张)
- **验证集**: 15% (42张)
- **测试集**: 15% (43张)

---

## 🐛 故障排查

### 问题1：模型加载失败

```python
# 确保使用正确的加载方式
from src.models.model_factory import load_model_from_checkpoint
model, checkpoint = load_model_from_checkpoint('path/to/checkpoint.pth')
```

### 问题2：ONNX推理结果不一致

```python
# 检查预处理是否正确
# 必须使用ImageNet归一化参数
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 问题3：Android内存溢出

```kotlin
// 在Application初始化时加载模型
class MyApp : Application() {
    lateinit var classifier: ImageClassifier

    override fun onCreate() {
        super.onCreate()
        // 后台线程初始化
        Thread {
            classifier = ImageClassifier(this)
        }.start()
    }
}
```

### 问题4：iOS推理速度慢

```swift
// 确保在后台线程执行推理
DispatchQueue.global(qos: .userInitiated).async {
    let result = classifier.classify(image: image)
    DispatchQueue.main.async {
        // 更新UI
    }
}
```

---

## 📮 技术支持

### 报告问题

如遇到问题，请提供以下信息：
- 操作系统和版本
- Python版本（`python --version`）
- PyTorch版本（`python -c "import torch; print(torch.__version__)"`）
- 完整错误日志
- 复现步骤

### 联系方式

- **Issue**: 在项目仓库提交Issue
- **Email**: [联系邮箱]
- **文档**: 查看相关文档章节

---

## 📖 进阶阅读

### 模型架构

- 基础架构：MobileNetV2
- 预训练：ImageNet-1K
- 迁移学习：冻结主干 + 微调全模型（两阶段）
- 分类头：全连接层 (1280 → 3)

### 训练策略

1. **阶段1**：冻结预训练主干，只训练分类头
   - Epochs: 15
   - 学习率: 1e-3
   - 优化器: Adam

2. **阶段2**：解冻所有层，端到端微调
   - Epochs: 25
   - 学习率: 1e-4（降低10倍）
   - 早停: patience=10

### 数据增强

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

---

## 🎓 学习资源

### PyTorch官方文档
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Mobile Deployment](https://pytorch.org/mobile/home/)

### ONNX资源
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [ONNX Model Zoo](https://github.com/onnx/models)

### CoreML资源
- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Converting Models to CoreML](https://coremltools.readme.io/)

---

## 📝 更新日志

### 2025-12-27
- ✅ 创建一键训练流水线脚本
- ✅ 实现8阶段自动化流程
- ✅ 添加模型对比功能
- ✅ 创建完整部署文档
- ✅ 优化输出目录结构

---

**最后更新**: 2025-12-27
**文档版本**: 1.0
