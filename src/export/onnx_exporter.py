"""ONNX 模型导出器"""
import torch
import onnx
import onnxruntime as ort
import numpy as np


class ONNXExporter:
    """ONNX 模型导出器"""

    def __init__(self, model, img_size=224):
        """
        Args:
            model: PyTorch 模型
            img_size: 输入图像尺寸
        """
        self.model = model
        self.img_size = img_size

    def export(self, save_path, opset_version=14):
        """
        导出为 ONNX 格式（单张图片输入）

        Args:
            save_path: 保存路径
            opset_version: ONNX opset 版本

        Returns:
            str: 保存路径
        """
        print(f"\n导出 ONNX 模型到: {save_path}")
        print(f"Opset 版本: {opset_version}")
        print(f"输入模式: 单张图片 (batch_size=1)")

        # 设置为评估模式
        self.model.eval()
        self.model.cpu()  # 导出时使用CPU

        # 创建虚拟输入 (batch_size=1, channels=3, height=img_size, width=img_size)
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size)

        # 导出 (固定 batch_size=1，每次只处理一张图片)
        print("正在导出...")
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            # 移除 dynamic_axes，固定 batch_size=1
            export_params=True,
            do_constant_folding=True
        )

        # 验证ONNX模型
        print("验证 ONNX 模型...")
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 模型验证通过")

        # 测试推理一致性
        self._test_inference(save_path, dummy_input)

        # 打印模型信息
        self._print_model_info(save_path)

        print(f"✓ ONNX 模型导出成功: {save_path}\n")
        return save_path

    def _test_inference(self, onnx_path, test_input):
        """
        测试 ONNX 模型推理一致性

        Args:
            onnx_path: ONNX 模型路径
            test_input: 测试输入
        """
        print("测试推理一致性...")

        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = self.model(test_input).numpy()

        # ONNX 推理
        session = ort.InferenceSession(onnx_path)
        onnx_output = session.run(None, {'input': test_input.numpy()})[0]

        # 验证一致性
        try:
            np.testing.assert_allclose(
                pytorch_output, onnx_output,
                rtol=1e-3, atol=1e-5
            )
            print("✓ 推理一致性验证通过")
        except AssertionError as e:
            print(f"⚠️  推理一致性验证失败: {e}")
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            print(f"   最大差异: {max_diff}")

    def _print_model_info(self, onnx_path):
        """
        打印 ONNX 模型信息

        Args:
            onnx_path: ONNX 模型路径
        """
        import os

        # 文件大小
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

        # 加载模型获取详细信息
        session = ort.InferenceSession(onnx_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        print("\nONNX 模型信息:")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  输入名称: {input_info.name}")
        print(f"  输入形状: {input_info.shape} (固定为单张图片)")
        print(f"  输入类型: {input_info.type}")
        print(f"  输出名称: {output_info.name}")
        print(f"  输出形状: {output_info.shape} (batch_size=1)")
        print(f"  输出类型: {output_info.type}")
        print(f"\n  使用说明: 此模型每次只接受一张图片输入")
