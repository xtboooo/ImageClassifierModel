"""ONNX 模型导出器"""
import torch
import onnx
import onnxruntime as ort
import numpy as np

from ..utils.logger import logger


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

    def export(self, save_path, opset_version=18):
        """
        导出为 ONNX 格式（单张图片输入）

        Args:
            save_path: 保存路径
            opset_version: ONNX opset 版本

        Returns:
            str: 保存路径
        """
        logger.info("导出 ONNX 模型",
                    save_path=str(save_path),
                    opset_version=opset_version,
                    input_mode="单张图片 (batch_size=1)")

        # 设置为评估模式
        self.model.eval()
        self.model.cpu()  # 导出时使用CPU

        # 创建虚拟输入 (batch_size=1, channels=3, height=img_size, width=img_size)
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size)

        # 导出 (固定 batch_size=1，每次只处理一张图片)
        logger.info("正在导出 ONNX...")
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
        logger.info("验证 ONNX 模型...")
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        logger.success("ONNX 模型验证通过")

        # 测试推理一致性
        self._test_inference(save_path, dummy_input)

        # 打印模型信息
        self._print_model_info(save_path)

        logger.success(f"ONNX 模型导出成功: {save_path}")
        return save_path

    def _test_inference(self, onnx_path, test_input):
        """
        测试 ONNX 模型推理一致性

        Args:
            onnx_path: ONNX 模型路径
            test_input: 测试输入
        """
        logger.info("测试推理一致性...")

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
            logger.success("推理一致性验证通过")
        except AssertionError as e:
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            logger.warning("推理一致性验证失败",
                          error=str(e),
                          max_diff=float(max_diff))

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

        logger.info("ONNX 模型信息",
                    file_size_mb=f"{file_size:.2f}",
                    input_name=input_info.name,
                    input_shape=str(input_info.shape),
                    input_type=input_info.type,
                    output_name=output_info.name,
                    output_shape=str(output_info.shape),
                    output_type=output_info.type,
                    note="此模型每次只接受一张图片输入")
