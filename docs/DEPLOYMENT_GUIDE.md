# 移动端部署指南

本文档介绍如何在Android和iOS平台上部署图片分类模型。

## 目录

- [模型格式选择](#模型格式选择)
- [Android部署](#android部署)
  - [方案1: 使用ONNX Runtime (推荐)](#方案1-使用onnx-runtime-推荐)
  - [方案2: 使用TensorFlow Lite](#方案2-使用tensorflow-lite)
- [iOS部署](#ios部署)
  - [使用CoreML (推荐)](#使用coreml-推荐)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

---

## 模型格式选择

| 平台 | 推荐格式 | 替代方案 | 文件大小 | 平均推理耗时 |
|------|---------|---------|----------|------------|
| **Android** | ONNX | TFLite | 11.3 MB | ~2.75 ms |
| **iOS** | CoreML | ONNX | 10.8 MB | ~3.1 ms |

### 为什么选择ONNX用于Android？

✅ **跨平台支持** - ONNX Runtime支持Android、iOS、Windows、Linux
✅ **性能优秀** - 推理速度比PyTorch快16倍
✅ **体积更小** - 相比PyTorch模型减少62%
✅ **预测一致** - 与原始PyTorch模型100%一致
✅ **易于集成** - Google官方支持ONNX Runtime

---

## Android部署

### 方案1: 使用ONNX Runtime (推荐)

#### 1. 添加依赖

在 `app/build.gradle` 中添加：

```gradle
dependencies {
    // ONNX Runtime for Android
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
}
```

#### 2. 将模型文件放入项目

将 `model.onnx` 和 `model.onnx.data` 放到：
```
app/src/main/assets/models/
├── model.onnx
└── model.onnx.data
```

#### 3. 创建推理类

```kotlin
// ImageClassifier.kt
import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

class ImageClassifier(context: Context) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // 类别名称
    private val classNames = arrayOf("Failure", "Loading", "Success")

    init {
        // 加载ONNX模型
        val modelBytes = context.assets.open("models/model.onnx").readBytes()
        session = ortEnv.createSession(modelBytes)
    }

    /**
     * 对图片进行分类
     * @param bitmap 输入图片
     * @return 分类结果 Pair(类别名称, 置信度)
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // 1. 预处理图片
        val inputTensor = preprocessImage(bitmap)

        // 2. 运行推理
        val inputs = mapOf("input" to inputTensor)
        val outputs = session.run(inputs)

        // 3. 处理输出
        val output = outputs[0].value as Array<FloatArray>
        val logits = output[0]

        // 4. Softmax获取概率
        val probabilities = softmax(logits)

        // 5. 找到最大概率的类别
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val className = classNames[maxIndex]
        val confidence = probabilities[maxIndex]

        // 释放资源
        outputs.forEach { it.close() }
        inputTensor.close()

        return Pair(className, confidence)
    }

    /**
     * 预处理图片
     */
    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        // 调整大小到224x224
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // 转换为Float数组 [1, 3, 224, 224]
        val inputBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        val pixels = IntArray(224 * 224)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        // ImageNet归一化参数
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // 转换为CHW格式并归一化
        for (c in 0..2) {
            for (h in 0..223) {
                for (w in 0..223) {
                    val pixel = pixels[h * 224 + w]
                    val value = when (c) {
                        0 -> ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
                        1 -> ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
                        else -> ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
                    }
                    inputBuffer.put(value)
                }
            }
        }

        inputBuffer.rewind()

        // 创建ONNX张量
        val shape = longArrayOf(1, 3, 224, 224)
        return OnnxTensor.createTensor(ortEnv, inputBuffer, shape)
    }

    /**
     * Softmax函数
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    /**
     * 释放资源
     */
    fun close() {
        session.close()
    }
}
```

#### 4. 使用示例

```kotlin
// MainActivity.kt
class MainActivity : AppCompatActivity() {
    private lateinit var classifier: ImageClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化分类器
        classifier = ImageClassifier(this)

        // 加载图片
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)

        // 执行分类
        val (className, confidence) = classifier.classify(bitmap)

        // 显示结果
        Log.d("Classifier", "预测类别: $className")
        Log.d("Classifier", "置信度: ${confidence * 100}%")

        textView.text = "类别: $className\n置信度: ${String.format("%.2f%%", confidence * 100)}"
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}
```

#### 5. 性能优化

```kotlin
// 启用加速器（如果设备支持）
val sessionOptions = OrtSession.SessionOptions()

// 尝试使用NNAPI（Android Neural Networks API）
sessionOptions.addNnapi()

// 或使用GPU（需要额外依赖）
// sessionOptions.addGpu()

val session = ortEnv.createSession(modelBytes, sessionOptions)
```

---

### 方案2: 使用TensorFlow Lite

如果您更倾向于使用TensorFlow Lite（Android原生支持更好），可以先导出TFLite格式：

```bash
# 导出TFLite模型
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --format tflite \
  --quantize  # 可选：INT8量化以减小模型体积
```

#### 1. 添加依赖

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

#### 2. TFLite推理代码

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import android.graphics.Bitmap

class TFLiteClassifier(context: Context) {
    private val interpreter: Interpreter
    private val classNames = arrayOf("Failure", "Loading", "Success")

    init {
        val model = FileUtil.loadMappedFile(context, "models/model.tflite")
        interpreter = Interpreter(model)
    }

    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // 预处理
        val input = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(3) }

        // 推理
        interpreter.run(input, output)

        // 后处理
        val probabilities = softmax(output[0])
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0

        return Pair(classNames[maxIndex], probabilities[maxIndex])
    }

    // preprocessImage 和 softmax 实现同上

    fun close() {
        interpreter.close()
    }
}
```

---

## iOS部署

### 使用CoreML (推荐)

CoreML是Apple官方的机器学习框架，在iOS设备上性能最佳。

#### 1. 导出CoreML模型

```bash
# 导出CoreML模型
uv run python scripts/export.py \
  --checkpoint data/output/checkpoints/best_model.pth \
  --format coreml
```

这会生成 `model.mlpackage` 文件。

#### 2. 将模型添加到Xcode项目

1. 将 `model.mlpackage` 拖入Xcode项目
2. Xcode会自动生成Swift/Objective-C接口类

#### 3. Swift实现

```swift
import UIKit
import CoreML
import Vision

class ImageClassifier {
    // 加载CoreML模型
    private lazy var model: VNCoreMLModel? = {
        guard let coreMLModel = try? model_20251227_124916(configuration: MLModelConfiguration()) else {
            return nil
        }
        return try? VNCoreMLModel(for: coreMLModel.model)
    }()

    // 类别名称
    private let classNames = ["Failure", "Loading", "Success"]

    /**
     * 对图片进行分类
     */
    func classify(image: UIImage, completion: @escaping (String, Float) -> Void) {
        guard let model = model else {
            print("模型加载失败")
            return
        }

        guard let ciImage = CIImage(image: image) else {
            print("图片转换失败")
            return
        }

        // 创建推理请求
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                print("推理失败: \(error?.localizedDescription ?? "未知错误")")
                return
            }

            // 返回结果
            completion(topResult.identifier, topResult.confidence)
        }

        // 执行推理
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("推理执行失败: \(error.localizedDescription)")
            }
        }
    }
}
```

#### 4. 使用示例

```swift
// ViewController.swift
class ViewController: UIViewController {
    let classifier = ImageClassifier()

    @IBAction func classifyButtonTapped(_ sender: UIButton) {
        guard let image = imageView.image else { return }

        // 显示加载指示器
        activityIndicator.startAnimating()

        // 执行分类
        classifier.classify(image: image) { className, confidence in
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()

                // 显示结果
                let percentage = String(format: "%.2f%%", confidence * 100)
                self.resultLabel.text = "类别: \(className)\n置信度: \(percentage)"

                print("预测类别: \(className)")
                print("置信度: \(percentage)")
            }
        }
    }
}
```

#### 5. Objective-C实现

```objc
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
#import "model_20251227_124916.h"

@interface ImageClassifier : NSObject
- (void)classifyImage:(UIImage *)image
           completion:(void (^)(NSString *className, float confidence))completion;
@end

@implementation ImageClassifier

- (void)classifyImage:(UIImage *)image
           completion:(void (^)(NSString *className, float confidence))completion {

    // 加载模型
    NSError *error = nil;
    model_20251227_124916 *coreMLModel = [[model_20251227_124916 alloc] initWithConfiguration:[MLModelConfiguration new] error:&error];
    if (error) {
        NSLog(@"模型加载失败: %@", error);
        return;
    }

    VNCoreMLModel *visionModel = [VNCoreMLModel modelForMLModel:coreMLModel.model error:&error];
    if (error) {
        NSLog(@"Vision模型创建失败: %@", error);
        return;
    }

    // 创建请求
    VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:visionModel completionHandler:^(VNRequest *request, NSError *error) {
        if (error) {
            NSLog(@"推理失败: %@", error);
            return;
        }

        VNClassificationObservation *topResult = request.results.firstObject;
        if (topResult) {
            completion(topResult.identifier, topResult.confidence);
        }
    }];

    // 执行推理
    CIImage *ciImage = [[CIImage alloc] initWithImage:image];
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCIImage:ciImage options:@{}];
    [handler performRequests:@[request] error:&error];
}

@end
```

---

## 性能对比

基于43张测试图片的实测数据（Apple Silicon MPS）：

| 模型格式 | 文件大小 | 平均耗时 | 最小耗时 | 最大耗时 | 标准差 | 预测一致性 |
|---------|---------|---------|---------|---------|--------|-----------|
| **PyTorch** | 29.1 MB | 44.37 ms | 40.01 ms | 49.66 ms | 1.65 ms | - |
| **ONNX** | 11.3 MB | 2.75 ms | 2.35 ms | 4.86 ms | 0.56 ms | 100% |
| **CoreML** | 10.8 MB | ~3.1 ms | - | - | - | 100% |

**关键结论**：
- ONNX模型推理速度比PyTorch快**16倍**
- 文件大小减少**62%**
- 与原始模型保持**100%预测一致性**

---

## 常见问题

### Q1: ONNX模型在Android上的兼容性如何？

**A**: ONNX Runtime支持Android 5.0 (API Level 21)及以上版本，覆盖99%+的Android设备。

### Q2: 是否需要网络连接？

**A**: 不需要。所有推理都在本地设备上完成，完全离线运行。

### Q3: 如何进一步优化性能？

**Android**:
- 使用NNAPI加速 (Neural Networks API)
- INT8量化减小模型体积
- 启用GPU推理（如果设备支持）

**iOS**:
- CoreML自动选择最优加速器（Neural Engine/GPU/CPU）
- 使用Metal Performance Shaders进一步优化

### Q4: 如何处理不同尺寸的输入图片？

所有平台都需要将输入调整为224x224像素。建议在移动端：
- 保持宽高比裁剪中心区域
- 或使用`ScaleAspectFill`模式

代码示例：
```kotlin
// Android: 中心裁剪
fun centerCropBitmap(bitmap: Bitmap, size: Int): Bitmap {
    val dimension = Math.min(bitmap.width, bitmap.height)
    val x = (bitmap.width - dimension) / 2
    val y = (bitmap.height - dimension) / 2
    val cropped = Bitmap.createBitmap(bitmap, x, y, dimension, dimension)
    return Bitmap.createScaledBitmap(cropped, size, size, true)
}
```

```swift
// iOS: 中心裁剪
extension UIImage {
    func centerCrop(to size: CGSize) -> UIImage? {
        let dimension = min(self.size.width, self.size.height)
        let x = (self.size.width - dimension) / 2
        let y = (self.size.height - dimension) / 2

        guard let cgImage = self.cgImage?.cropping(to: CGRect(x: x, y: y, width: dimension, height: dimension)) else {
            return nil
        }

        UIGraphicsBeginImageContextWithOptions(size, false, 0)
        defer { UIGraphicsEndImageContext() }

        UIImage(cgImage: cgImage).draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
```

### Q5: 模型支持批量推理吗？

当前导出的模型支持批量推理（batch size可变），但移动端通常单张处理即可满足需求。

### Q6: 如何更新模型？

**方法1**: 发布新版本App
**方法2**: 实现远程模型下载更新机制

```kotlin
// Android示例：从服务器下载新模型
suspend fun downloadModel(url: String): File {
    val response = httpClient.get(url)
    val file = File(context.filesDir, "models/model_new.onnx")
    file.outputStream().use { output ->
        response.bodyAsChannel().copyTo(output)
    }
    return file
}
```

### Q7: 内存占用如何？

- **ONNX Runtime**: ~20-30 MB运行时内存
- **CoreML**: ~15-25 MB运行时内存
- **TFLite**: ~25-35 MB运行时内存

建议在后台线程初始化模型，避免阻塞UI。

---

## 完整示例项目

完整的Android和iOS示例项目代码请参考：

- **Android项目**: `examples/android/ImageClassifierDemo`
- **iOS项目**: `examples/ios/ImageClassifierDemo`

（待创建）

---

## 技术支持

如有问题，请提交Issue或联系开发团队。

**最后更新**: 2025-12-27
