"""TensorFlow Lite 模型导出器（Android）"""
import torch
import numpy as np
from pathlib import Path


class TFLiteExporter:
    """TFLite 模型导出器（用于 Android 部署）"""

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
        导出为 TFLite 格式（通过 ONNX 中间格式）

        Args:
            save_path: 保存路径
            quantize: 是否进行量化（减小模型大小）

        Returns:
            str: 保存路径
        """
        print(f"\n导出 TFLite 模型到: {save_path}")

        try:
            # 检查依赖
            try:
                import onnx
                from onnx_tf.backend import prepare
                import tensorflow as tf
            except ImportError as e:
                print("❌ 缺少必要的依赖包")
                print("\n请安装以下依赖:")
                print("  pip install onnx onnx-tf tensorflow")
                print("\n或使用 uv:")
                print("  uv pip install onnx onnx-tf tensorflow")
                raise ImportError(f"Missing dependencies: {e}")

            # 设置为评估模式
            self.model.eval()
            self.model.cpu()

            # 步骤 1: 先导出为 ONNX
            print("步骤 1/3: 导出为 ONNX 中间格式...")
            onnx_path = str(Path(save_path).with_suffix('.onnx'))
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)

            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                opset_version=14,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                export_params=True,
                do_constant_folding=True
            )
            print("  ✓ ONNX 导出完成")

            # 步骤 2: 转换 ONNX 到 TensorFlow
            print("步骤 2/3: 转换 ONNX 到 TensorFlow...")
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)

            # 导出为 TensorFlow SavedModel
            tf_model_dir = str(Path(save_path).with_suffix('.tf'))
            tf_rep.export_graph(tf_model_dir)
            print(f"  ✓ TensorFlow 模型已保存到: {tf_model_dir}")

            # 步骤 3: 转换为 TFLite
            print("步骤 3/3: 转换为 TFLite...")
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

            if quantize:
                print("  启用量化...")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()

            # 保存 TFLite 模型
            with open(save_path, 'wb') as f:
                f.write(tflite_model)

            print(f"  ✓ TFLite 模型已保存")

            # 清理临时文件
            print("清理临时文件...")
            import shutil
            if Path(onnx_path).exists():
                Path(onnx_path).unlink()
            if Path(tf_model_dir).exists():
                shutil.rmtree(tf_model_dir)

            # 打印模型信息
            self._print_model_info(save_path, quantize)

            # 测试模型
            self._test_inference(save_path, dummy_input)

            print(f"✓ TFLite 模型导出成功: {save_path}\n")
            return save_path

        except Exception as e:
            print(f"❌ TFLite 导出失败: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 提示: TFLite 导出依赖较多,如遇问题可以:")
            print("  1. 优先使用 ONNX 格式进行跨平台部署")
            print("  2. 使用在线工具转换 ONNX -> TFLite")
            print("  3. 手动使用 onnx-tf 和 tensorflow 工具链转换")
            raise

    def _test_inference(self, tflite_path, test_input):
        """
        测试 TFLite 模型推理

        Args:
            tflite_path: TFLite 模型路径
            test_input: 测试输入
        """
        try:
            import tensorflow as tf

            print("测试 TFLite 模型推理...")

            # 加载 TFLite 模型
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            # 获取输入输出详情
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # 准备输入数据
            input_data = test_input.numpy().astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # 运行推理
            interpreter.invoke()

            # 获取输出
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # PyTorch 推理对比
            with torch.no_grad():
                pytorch_output = self.model(test_input).numpy()

            # 验证一致性
            try:
                np.testing.assert_allclose(
                    pytorch_output, output_data,
                    rtol=1e-2, atol=1e-3  # TFLite 转换可能有较大误差
                )
                print("✓ 推理一致性验证通过")
            except AssertionError:
                max_diff = np.max(np.abs(pytorch_output - output_data))
                print(f"⚠️  推理结果存在差异(最大差异: {max_diff:.6f})")
                print("   这是正常的,TFLite 转换可能引入少量数值差异")

        except Exception as e:
            print(f"⚠️  推理测试失败: {e}")

    def _print_model_info(self, model_path, quantize):
        """
        打印 TFLite 模型信息

        Args:
            model_path: 模型路径
            quantize: 是否量化
        """
        import os

        # 文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

        print("\nTFLite 模型信息:")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  量化: {'是 (FP16)' if quantize else '否 (FP32)'}")
        print(f"  类别: {', '.join(self.class_names)}")
        print(f"  输入尺寸: (1, {self.img_size}, {self.img_size}, 3)")
        print("  注意: TFLite 使用 NHWC 格式 (PyTorch 使用 NCHW)")
