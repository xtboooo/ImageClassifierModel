# ImageClassifierModel 完整操作指南

本指南提供从零开始训练、导出和使用图像分类模型的详细步骤。

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [模型训练](#3-模型训练)
4. [模型评估](#4-模型评估)
5. [模型导出](#5-模型导出)
6. [模型使用](#6-模型使用)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 安装依赖

项目使用 `uv` 进行环境和依赖管理：

```bash
# 首次使用：安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖
cd ImageClassifierModel
uv sync

# 激活虚拟环境（可选，uv run 会自动处理）
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows
```

### 1.2 验证环境

```bash
# 检查 Python 版本
uv run python --version  # 应显示 Python 3.12+

# 检查 PyTorch 安装
uv run python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 检查 GPU/MPS 支持
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

---

## 2. 数据准备

### 2.1 数据集结构

训练数据应按以下结构组织：

```
data/input/data1226/
├── Failure/        # 失败状态截图
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
├── Loading/        # 加载状态截图
│   ├── img_001.png
│   └── ...
└── Success/        # 成功状态截图
    ├── img_001.png
    └── ...
```

### 2.2 数据划分

将数据分为训练集/验证集/测试集（70%/15%/15%）：

```bash
# 使用内置脚本自动划分数据
uv run python -c "
from pathlib import Path
from src.data.split_data import split_dataset

split_dataset(
    input_dir=Path('data/input/data1226'),
    output_dir=Path('data/processed'),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)
print('✓ 数据集划分完成！')
"
```

划分后的结构：

```
data/processed/
├── train/
│   ├── Failure/
│   ├── Loading/
│   └── Success/
├── val/
│   ├── Failure/
│   ├── Loading/
│   └── Success/
└── test/
    ├── Failure/
    ├── Loading/
    └── Success/
```

---

## 3. 模型训练

### 3.1 快速开始（推荐配置）

使用默认配置训练模型：

```bash
uv run python scripts/train.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-3 \
  --pretrained
```

**参数说明：**
- `--epochs 30`: 训练 30 轮
- `--batch-size 16`: 批次大小 16
- `--lr 1e-3`: 学习率 0.001
- `--pretrained`: 使用 ImageNet 预训练权重

### 3.2 两阶段训练（更好的效果）

先冻结主干网络训练分类头，再微调整个网络：

```bash
uv run python scripts/train.py \
  --two-stage \
  --stage1-epochs 10 \
  --stage2-epochs 20 \
  --stage2-lr 1e-4 \
  --batch-size 16 \
  --pretrained
```

**训练流程：**
1. **阶段 1** (10 epochs): 冻结 MobileNetV2 主干，仅训练分类头
2. **阶段 2** (20 epochs): 解冻部分层，微调整个网络

### 3.3 完整参数列表

```bash
uv run python scripts/train.py --help
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--dropout` | 0.3 | Dropout 比例 |
| `--img-size` | 224 | 输入图像尺寸 |
| `--pretrained` | True | 使用预训练权重 |
| `--patience` | 10 | 早停耐心值 |
| `--device` | auto | 设备选择 (auto/mps/cuda/cpu) |
| `--data-root` | data/processed | 数据根目录 |

### 3.4 训练输出

训练过程会自动保存：

```
data/output/
├── checkpoints/
│   └── best_model.pth           # 最佳模型检查点
├── logs/
│   └── training_YYYYMMDD_HHMMSS.log  # 训练日志
└── visualizations/
    └── training_history.png     # 训练曲线
```

### 3.5 监控训练过程

训练时会实时显示进度：

```
Epoch 10/30
Train Loss: 0.3245 | Train Acc: 89.23%
Val Loss:   0.2891 | Val Acc:   91.47%
Learning Rate: 0.001000

Early Stopping: 3/10 (无提升轮数)
Best Val Acc: 91.47% (Epoch 10)
```

---

## 4. 模型评估

### 4.1 评估测试集

使用测试集评估模型性能：

```bash
uv run python scripts/evaluate.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --data-root data/processed \
  --output-dir data/output/metrics
```

### 4.2 评估输出

评估会生成：

1. **分类报告** (`classification_report.txt`):
```
              precision    recall  f1-score   support

     Failure     1.0000    0.8667    0.9286        15
     Loading     0.8235    1.0000    0.9032        14
     Success     1.0000    0.9286    0.9630        14

    accuracy                         0.9302        43
```

2. **指标 JSON** (`test_metrics.json`): 包含详细的性能指标

3. **可视化图表**:
   - `test_confusion_matrix.png` - 混淆矩阵
   - `test_per_class_metrics.png` - 各类别指标对比

### 4.3 查看评估结果

```bash
# 查看文本报告
cat data/output/metrics/classification_report.txt

# 查看 JSON 指标
cat data/output/metrics/test_metrics.json | python -m json.tool
```

---

## 5. 模型导出

### 5.1 导出所有格式（推荐）

一键导出 ONNX 和 CoreML 格式：

```bash
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --formats onnx coreml \
  --model-name screenshot_classifier \
  --output-dir data/output/exported_models
```

### 5.2 按平台导出

**导出 ONNX（跨平台）：**
```bash
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --formats onnx \
  --model-name screenshot_classifier
```

**导出 CoreML（iOS/macOS）：**
```bash
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --formats coreml \
  --model-name screenshot_classifier
```

**导出 TFLite（Android）：**

⚠️ TFLite 需要额外依赖，推荐使用在线转换工具：

**方法 1: 在线转换（推荐）**
1. 先导出 ONNX 格式
2. 访问 https://convertmodel.com/
3. 上传 `screenshot_classifier.onnx`
4. 下载 `.tflite` 文件

**方法 2: 本地转换**
```bash
# 安装 TFLite 依赖
uv pip install tensorflow onnx onnx-tf

# 导出
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --formats tflite \
  --model-name screenshot_classifier
```

### 5.3 模型量化（减小体积）

```bash
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --formats coreml tflite \
  --quantize  # 启用 FP16 量化
```

### 5.4 导出结果

```
data/output/exported_models/
├── screenshot_classifier.onnx         # ONNX 模型
├── screenshot_classifier.onnx.data    # ONNX 外部数据
├── screenshot_classifier.mlpackage/   # CoreML 模型
└── screenshot_classifier.tflite       # TFLite 模型（如果导出）
```

---

## 6. 模型使用

### 6.1 批量推理（Python）

对文件夹中的所有图片进行分类：

```bash
uv run python scripts/batch_inference.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --input-dir /path/to/images \
  --output predictions.json
```

**复制图片到分类文件夹：**
```bash
uv run python scripts/batch_inference.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --input-dir /path/to/images \
  --output predictions.json \
  --copy-to-folders \
  --output-dir data/output/classified_images
```

### 6.2 Python 代码示例

#### 单张图片推理

```python
import torch
from PIL import Image
from pathlib import Path

# 1. 加载模型
from src.models.model_factory import load_model_from_checkpoint
from src.data.transforms import get_val_transforms
from src.utils.device import get_device

checkpoint_path = "data/output/checkpoints/best_model.pth"
model, checkpoint = load_model_from_checkpoint(checkpoint_path)
device = get_device()
model.to(device)
model.eval()

# 2. 加载图片
image_path = "test_image.png"
transform = get_val_transforms(img_size=224)
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# 3. 推理
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# 4. 解析结果
class_names = ['Failure', 'Loading', 'Success']
predicted_class = class_names[predicted.item()]
confidence_score = confidence.item()

print(f"预测类别: {predicted_class}")
print(f"置信度: {confidence_score:.2%}")
print(f"\n所有类别概率:")
for i, class_name in enumerate(class_names):
    prob = probabilities[0][i].item()
    print(f"  {class_name}: {prob:.2%}")
```

#### 批量推理（优化版）

```python
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 创建数据集
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image), str(self.image_paths[idx].name)

# 加载模型和数据
model, _ = load_model_from_checkpoint("data/output/checkpoints/best_model.pth")
model.to(device).eval()

image_paths = list(Path("input_dir").glob("*.png"))
dataset = ImageDataset(image_paths, get_val_transforms())
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 批量推理
results = {}
class_names = ['Failure', 'Loading', 'Success']

with torch.no_grad():
    for images, filenames in tqdm(dataloader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        for i, filename in enumerate(filenames):
            pred_idx = probs[i].argmax().item()
            results[filename] = {
                'class': class_names[pred_idx],
                'confidence': probs[i][pred_idx].item(),
                'probabilities': {
                    class_names[j]: probs[i][j].item()
                    for j in range(len(class_names))
                }
            }

# 保存结果
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 6.3 ONNX 推理（跨平台）

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. 加载 ONNX 模型
session = ort.InferenceSession("data/output/exported_models/screenshot_classifier.onnx")

# 2. 预处理图片
def preprocess_onnx(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))

    # 转换为数组并归一化
    img_array = np.array(img).astype(np.float32) / 255.0

    # ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # 调整维度: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# 3. 推理
input_data = preprocess_onnx("test_image.png")
outputs = session.run(None, {'input': input_data})[0]

# 4. Softmax + 解析
from scipy.special import softmax
probs = softmax(outputs[0])

class_names = ['Failure', 'Loading', 'Success']
pred_idx = np.argmax(probs)

print(f"预测: {class_names[pred_idx]} ({probs[pred_idx]:.2%})")
```

### 6.4 iOS 集成（CoreML）

```swift
import CoreML
import Vision
import UIKit

class ScreenshotClassifier {
    private let model: screenshot_classifier

    init() throws {
        self.model = try screenshot_classifier(configuration: MLModelConfiguration())
    }

    func classify(image: UIImage, completion: @escaping (String, Double) -> Void) {
        // 1. 转换为 CVPixelBuffer
        guard let pixelBuffer = image.toCVPixelBuffer(size: CGSize(width: 224, height: 224)) else {
            return
        }

        // 2. 创建请求
        guard let vnModel = try? VNCoreMLModel(for: model.model) else { return }

        let request = VNCoreMLRequest(model: vnModel) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else { return }

            completion(topResult.identifier, Double(topResult.confidence))
        }

        // 3. 执行推理
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}

// 使用示例
let classifier = try ScreenshotClassifier()
let image = UIImage(named: "test_screenshot")!

classifier.classify(image: image) { predictedClass, confidence in
    print("预测: \(predictedClass), 置信度: \(confidence)")
}

// UIImage 扩展（转换为 CVPixelBuffer）
extension UIImage {
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
}
```

### 6.5 Android 集成（TFLite）

```kotlin
import org.tensorflow.lite.Interpreter
import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ScreenshotClassifier(private val modelPath: String) {
    private val interpreter: Interpreter
    private val classNames = arrayOf("Failure", "Loading", "Success")

    init {
        interpreter = Interpreter(loadModelFile(modelPath))
    }

    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // 1. 预处理图片
        val inputBuffer = preprocessImage(bitmap)

        // 2. 推理
        val outputBuffer = Array(1) { FloatArray(3) }
        interpreter.run(inputBuffer, outputBuffer)

        // 3. 解析结果
        val probabilities = outputBuffer[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0

        return Pair(classNames[maxIndex], probabilities[maxIndex])
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 224
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Resize
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // ImageNet 标准化参数
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // 提取像素并标准化
        val intValues = IntArray(inputSize * inputSize)
        scaledBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in intValues) {
            val r = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
            val g = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
            val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }

    private fun loadModelFile(modelPath: String): java.nio.MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}

// 使用示例
val classifier = ScreenshotClassifier("screenshot_classifier.tflite")
val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)

val (predictedClass, confidence) = classifier.classify(bitmap)
Log.d("Classifier", "预测: $predictedClass, 置信度: ${confidence * 100}%")
```

---

## 7. 常见问题

### Q1: 训练时显示 "MPS backend out of memory"

**解决方案：**
```bash
# 减小批次大小
uv run python scripts/train.py --batch-size 8

# 或使用 CPU
uv run python scripts/train.py --device cpu
```

### Q2: 如何恢复中断的训练？

训练会自动保存最佳模型到 `data/output/checkpoints/best_model.pth`，如需继续训练需要修改代码支持。

### Q3: 模型准确率不理想怎么办？

**优化建议：**

1. **增加数据量**：每个类别至少 150+ 样本
2. **数据增强**：训练时已自动应用
3. **两阶段训练**：使用 `--two-stage` 参数
4. **调整超参数**：
   ```bash
   uv run python scripts/train.py \
     --two-stage \
     --stage1-epochs 15 \
     --stage2-epochs 25 \
     --lr 5e-4 \
     --dropout 0.4
   ```

### Q4: 如何在服务器上运行推理？

**ONNX Runtime 部署：**

```python
# 安装
pip install onnxruntime  # CPU
# 或
pip install onnxruntime-gpu  # GPU

# API 示例 (Flask)
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
session = ort.InferenceSession("screenshot_classifier.onnx")
class_names = ['Failure', 'Loading', 'Success']

@app.route('/predict', methods=['POST'])
def predict():
    # 接收图片
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # 预处理
    input_data = preprocess_onnx(img)  # 使用前面的预处理函数

    # 推理
    outputs = session.run(None, {'input': input_data})[0]
    probs = softmax(outputs[0])

    # 返回结果
    pred_idx = np.argmax(probs)
    return jsonify({
        'class': class_names[pred_idx],
        'confidence': float(probs[pred_idx]),
        'probabilities': {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Q5: TFLite 转换失败怎么办？

**推荐方案：**

1. 使用 ONNX 格式（兼容性最好）
2. 在线转换工具：https://convertmodel.com/
3. 使用 ONNX Runtime Mobile（支持 Android/iOS）

### Q6: 如何查看模型详细信息？

```bash
# PyTorch 模型
uv run python -c "
from src.models.model_factory import load_model_from_checkpoint
model, ckpt = load_model_from_checkpoint('data/output/checkpoints/best_model.pth')
print(model)
"

# ONNX 模型（使用 Netron）
# 访问 https://netron.app/
# 上传 screenshot_classifier.onnx 查看
```

---

## 📞 技术支持

- **项目文档**: `CLAUDE.md`
- **评估报告**: `data/output/模型评估与导出总结.md`
- **训练日志**: `data/output/logs/`

---

## 📝 快速命令参考

```bash
# 训练模型
uv run python scripts/train.py --two-stage --pretrained

# 评估模型
uv run python scripts/evaluate.py --checkpoint data/output/checkpoints/best_model.pth

# 导出模型
uv run python scripts/export.py --checkpoint data/output/checkpoints/best_model.pth --formats onnx coreml

# 批量推理
uv run python scripts/batch_inference.py --checkpoint data/output/checkpoints/best_model.pth --input-dir /path/to/images --copy-to-folders
```

---

**最后更新**: 2025-12-27
**模型版本**: v1.0
**作者**: ImageClassifierModel Project
