# 模型API规格文档

本文档详细说明模型的输入输出格式、预处理要求和后处理方法。

## 目录

- [模型信息](#模型信息)
- [输入规格](#输入规格)
- [输出规格](#输出规格)
- [预处理流程](#预处理流程)
- [后处理流程](#后处理流程)
- [代码示例](#代码示例)

---

## 模型信息

### 基本信息

| 项目 | 值 |
|-----|-----|
| **模型架构** | MobileNetV2 |
| **分类数量** | 3类 (Failure, Loading, Success) |
| **输入尺寸** | 224 × 224 × 3 |
| **参数总量** | 2,881,283 |
| **可训练参数** | 657,411 (22.8%) |
| **预训练** | ImageNet |

### 性能指标

| 指标 | 值 |
|-----|-----|
| **测试集准确率** | 93.02% |
| **Failure - Precision** | 100.00% |
| **Failure - Recall** | 86.67% |
| **Loading - Precision** | 82.35% |
| **Loading - Recall** | 100.00% |
| **Success - Precision** | 100.00% |
| **Success - Recall** | 92.86% |

### 模型文件

| 格式 | 文件名 | 大小 | 推理耗时 |
|-----|--------|------|---------|
| **PyTorch** | best_model.pth | 29.1 MB | ~44 ms |
| **ONNX** | model.onnx | 11.3 MB | ~2.75 ms |
| **CoreML** | model.mlpackage | 10.8 MB | ~3.1 ms |
| **TFLite** | model.tflite | ~11 MB | ~3-4 ms |

---

## 输入规格

### 张量规格

```
名称: input
形状: [batch_size, 3, 224, 224]
类型: float32
取值范围: [-2.64, 2.64] (归一化后)
格式: CHW (Channel-Height-Width)
通道顺序: RGB
```

### 详细说明

- **batch_size**: 批次大小，可变（通常为1）
- **3**: RGB三通道
- **224 × 224**: 固定输入尺寸
- **float32**: 32位浮点数
- **CHW格式**: 与PyTorch一致，通道优先

---

## 输出规格

### 张量规格

```
名称: output
形状: [batch_size, 3]
类型: float32
含义: 每个类别的logits（未归一化的分数）
```

### Logits说明

输出是**原始logits**，不是概率。需要经过Softmax转换为概率：

```
logits: [-2.31, 0.82, 3.45]
        ↓ Softmax
probabilities: [0.0091, 0.2085, 0.7824]
```

### 类别映射

| 索引 | 类别名称 | 含义 |
|-----|---------|------|
| **0** | Failure | 操作失败状态 |
| **1** | Loading | 加载中/等待状态 |
| **2** | Success | 操作成功状态 |

---

## 预处理流程

### 流程图

```
原始图片 (任意尺寸, RGB)
    ↓
调整大小 (224×224, 保持宽高比可选)
    ↓
转换为float32数组 (0-255 → 0.0-1.0)
    ↓
ImageNet归一化
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    normalized = (pixel/255.0 - mean) / std
    ↓
转换为CHW格式
    ↓
添加batch维度 [1, 3, 224, 224]
```

### 数学公式

对于每个像素值 `pixel[c]` (c = R/G/B通道):

```
normalized[c] = (pixel[c] / 255.0 - mean[c]) / std[c]
```

其中：
- `mean = [0.485, 0.456, 0.406]`  (R, G, B)
- `std = [0.229, 0.224, 0.225]`   (R, G, B)

### Python实现

```python
import torch
from PIL import Image
from torchvision import transforms

# 定义预处理变换
preprocess = transforms.Compose([
    transforms.Resize(256),                      # 调整短边到256
    transforms.CenterCrop(224),                  # 中心裁剪到224×224
    transforms.ToTensor(),                       # 转为Tensor并归一化到[0,1]
    transforms.Normalize(                        # ImageNet归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载和预处理图片
image = Image.open('test.jpg').convert('RGB')
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # 添加batch维度: [1, 3, 224, 224]
```

### NumPy实现

```python
import numpy as np
from PIL import Image

def preprocess_numpy(image_path):
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))

    # 转为NumPy数组 [H, W, C]
    image_array = np.array(image, dtype=np.float32)

    # 归一化到[0, 1]
    image_array = image_array / 255.0

    # ImageNet归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std

    # 转为CHW格式: [C, H, W]
    image_array = np.transpose(image_array, (2, 0, 1))

    # 添加batch维度: [1, C, H, W]
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
```

---

## 后处理流程

### 流程图

```
模型输出 logits [1, 3]
    ↓
Softmax归一化
    ↓
概率分布 [P(Failure), P(Loading), P(Success)]
    ↓
argmax获取最大概率索引
    ↓
映射到类别名称
```

### Softmax公式

```
Softmax(x_i) = exp(x_i) / Σ(exp(x_j))
```

为数值稳定性，通常减去最大值：

```
Softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))
```

### Python实现

```python
import torch
import torch.nn.functional as F

# 模型推理
with torch.no_grad():
    outputs = model(input_batch)  # [1, 3]

# Softmax获取概率
probabilities = F.softmax(outputs, dim=1)  # [1, 3]

# 获取预测类别和置信度
confidence, predicted = torch.max(probabilities, 1)

# 类别映射
class_names = ['Failure', 'Loading', 'Success']
predicted_class = class_names[predicted.item()]
confidence_score = confidence.item()

print(f"预测类别: {predicted_class}")
print(f"置信度: {confidence_score:.2%}")
```

### NumPy实现

```python
import numpy as np

def softmax(logits):
    """数值稳定的Softmax"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# 后处理
logits = model_output[0]  # [3]
probabilities = softmax(logits)

predicted_idx = np.argmax(probabilities)
confidence = probabilities[predicted_idx]

class_names = ['Failure', 'Loading', 'Success']
predicted_class = class_names[predicted_idx]

print(f"预测类别: {predicted_class}")
print(f"置信度: {confidence:.2%}")
print(f"所有类别概率: {probabilities}")
```

---

## 代码示例

### 完整PyTorch推理流程

```python
import torch
from PIL import Image
from torchvision import transforms

# 1. 加载模型
checkpoint = torch.load('best_model.pth', map_location='cpu')
model = checkpoint['model']
model.eval()

# 2. 定义预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 3. 加载和预处理图片
image = Image.open('test.jpg').convert('RGB')
input_tensor = preprocess(image).unsqueeze(0)

# 4. 推理
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# 5. 结果
class_names = ['Failure', 'Loading', 'Success']
result = {
    'class': class_names[predicted.item()],
    'confidence': confidence.item(),
    'all_probabilities': {
        class_names[i]: probabilities[0][i].item()
        for i in range(len(class_names))
    }
}

print(result)
```

### 完整ONNX推理流程

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. 加载ONNX模型
session = ort.InferenceSession('model.onnx')

# 2. 预处理
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std

    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# 3. 推理
input_data = preprocess('test.jpg')
outputs = session.run(None, {'input': input_data})
logits = outputs[0][0]

# 4. Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

probabilities = softmax(logits)

# 5. 结果
class_names = ['Failure', 'Loading', 'Success']
predicted_idx = np.argmax(probabilities)

result = {
    'class': class_names[predicted_idx],
    'confidence': float(probabilities[predicted_idx]),
    'all_probabilities': {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }
}

print(result)
```

### 输出示例

```json
{
  "class": "Success",
  "confidence": 0.9847,
  "all_probabilities": {
    "Failure": 0.0012,
    "Loading": 0.0141,
    "Success": 0.9847
  }
}
```

---

## 性能测试

### 测试环境

- **设备**: Apple M2 Pro
- **加速器**: MPS (Metal Performance Shaders)
- **测试图片**: 43张 (来自data/processed/test/)

### 测试结果

| 模型格式 | 平均耗时 | 最小耗时 | 最大耗时 | 标准差 |
|---------|---------|---------|---------|--------|
| PyTorch | 44.37 ms | 40.01 ms | 49.66 ms | 1.65 ms |
| ONNX | 2.75 ms | 2.35 ms | 4.86 ms | 0.56 ms |

**性能提升**: ONNX比PyTorch快**16.1倍**

### 性能测试代码

```python
import time
import torch
from tqdm import tqdm

def benchmark(model, dataloader, num_runs=100):
    model.eval()
    times = []

    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            for images, _ in dataloader:
                start = time.perf_counter()
                _ = model(images)
                torch.mps.synchronize()  # 等待GPU完成
                end = time.perf_counter()
                times.append((end - start) * 1000)  # 转为ms

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }
```

---

## 常见问题

### Q1: 为什么需要ImageNet归一化？

**A**: 模型使用ImageNet预训练权重进行迁移学习，必须使用相同的归一化参数以保持特征分布一致。

### Q2: 能否使用其他输入尺寸？

**A**: 当前模型仅支持224×224输入。如需其他尺寸，需要重新训练或使用自适应池化。

### Q3: 输入图片必须是正方形吗？

**A**: 不必须。预处理时会自动调整大小。建议使用中心裁剪或保持宽高比缩放。

### Q4: 如何处理灰度图？

**A**: 将灰度图转换为RGB：
```python
image = Image.open('gray.jpg').convert('RGB')
```

### Q5: Softmax在哪里计算？

**A**: 模型输出的是logits，Softmax需要在后处理阶段手动计算。这样设计可以：
- 在训练时使用`CrossEntropyLoss`（内置Softmax）
- 推理时灵活选择是否需要概率

---

## 版本历史

| 版本 | 日期 | 变更 |
|-----|------|------|
| 1.0 | 2025-12-27 | 初始版本 |

**最后更新**: 2025-12-27
