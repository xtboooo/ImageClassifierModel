"""CoreML 模型导出器（iOS）"""
import torch
import coremltools as ct


class CoreMLExporter:
    """CoreML 模型导出器（用于 iOS 部署）"""

    def __init__(self, model, img_size=224, class_names=None):
        """
        Args:
            model: PyTorch 模型
            img_size: 输入图像尺寸
            class_names: 类别名称列表
        """
        self.model = model
        self.img_size = img_size
        self.class_names = class_names or ['Failure', 'Loading', 'Success']

    def export(self, save_path, quantize=False):
        """
        导出为 CoreML 格式

        Args:
            save_path: 保存路径
            quantize: 是否进行量化（减小模型大小）

        Returns:
            str: 保存路径
        """
        print(f"\n导出 CoreML 模型到: {save_path}")

        # 设置为评估模式
        self.model.eval()
        self.model.cpu()

        # 创建示例输入
        example_input = torch.randn(1, 3, self.img_size, self.img_size)

        # Trace 模型
        print("正在 trace 模型...")
        traced_model = torch.jit.trace(self.model, example_input)

        # 转换为 CoreML
        print("正在转换为 CoreML...")
        try:
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(
                    name="input",
                    shape=example_input.shape
                )],
                classifier_config=ct.ClassifierConfig(self.class_names),
                compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32
            )

            # 添加元数据
            coreml_model.author = 'ImageClassifierModel Project'
            coreml_model.license = 'MIT'
            coreml_model.short_description = 'Mobile screenshot classifier for Failure/Loading/Success states'
            coreml_model.version = '1.0'

            # 保存
            coreml_model.save(save_path)

            # 打印模型信息
            self._print_model_info(save_path, quantize)

            print(f"✓ CoreML 模型导出成功: {save_path}\n")
            return save_path

        except Exception as e:
            print(f"❌ CoreML 导出失败: {e}")
            print("提示: CoreML 导出在某些情况下可能会失败，建议优先使用 ONNX 格式")
            raise

    def _print_model_info(self, model_path, quantize):
        """
        打印 CoreML 模型信息

        Args:
            model_path: 模型路径
            quantize: 是否量化
        """
        import os

        # 文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

        print("\nCoreML 模型信息:")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  量化: {'是 (FP16)' if quantize else '否 (FP32)'}")
        print(f"  类别: {', '.join(self.class_names)}")
        print(f"  输入尺寸: ({1, 3, self.img_size, self.img_size})")
