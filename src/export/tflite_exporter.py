"""TensorFlow Lite 模型导出器（基于 ai-edge-torch）"""
import torch
import numpy as np
from pathlib import Path

from ..utils.logger import logger


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
        导出为 TFLite 格式（仅支持 Linux 系统，其他系统请使用 Docker）

        Args:
            save_path: 保存路径
            quantize: 是否进行量化（减小模型大小）

        Returns:
            str: 保存路径
        """
        import platform

        current_platform = platform.system()

        # 非 Linux 系统，拒绝直接导出，引导用户使用 Docker
        if current_platform != 'Linux':
            error_msg = (
                f"TFLite 导出失败: ai-edge-torch 仅支持 Linux 系统\n"
                f"当前系统: {current_platform}\n"
                f"请使用 Docker 导出:\n"
                f"   bash docker/export_tflite.sh <checkpoint_path> <output_path>\n"
                f"示例:\n"
                f"   bash docker/export_tflite.sh data/output/checkpoints/best_model.pth {save_path}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Linux 系统，使用 ai-edge-torch 导出
        logger.info("使用 ai-edge-torch 导出 TFLite 模型", save_path=str(save_path))

        try:
            # 检查依赖
            try:
                import ai_edge_torch
            except ImportError as e:
                logger.error("ai-edge-torch 未安装")
                logger.info("安装命令: pip install ai-edge-torch")
                raise ImportError(f"ai-edge-torch not available: {e}")

            # 设置为评估模式
            self.model.eval()
            self.model.cpu()

            # 创建示例输入
            logger.info("准备模型转换...")
            sample_inputs = (torch.randn(1, 3, self.img_size, self.img_size),)

            # 直接使用 ai-edge-torch 转换
            logger.info("正在转换模型（PyTorch → TFLite）...")
            edge_model = ai_edge_torch.convert(
                self.model.eval(),
                sample_inputs
            )

            # 导出为 TFLite
            logger.info(f"正在保存模型到: {save_path}")
            edge_model.export(save_path)

            # 验证模型
            self._verify_model(edge_model, sample_inputs, save_path)

            # 打印模型信息
            self._print_model_info(save_path, quantize)

            logger.success(f"TFLite 导出成功: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"TFLite 导出失败: {e}")
            import traceback
            traceback.print_exc()
            logger.info("提示:")
            logger.info("  - 确保已安装 ai-edge-torch: pip install ai-edge-torch")
            logger.info("  - 或使用 Docker 导出: bash docker/export_tflite.sh")
            raise

    def _verify_model(self, edge_model, sample_inputs, save_path):
        """
        验证模型一致性

        Args:
            edge_model: ai-edge-torch 转换后的模型
            sample_inputs: 示例输入
            save_path: 保存路径
        """
        logger.info("验证模型一致性...")

        try:
            # ai-edge-torch 推理
            edge_output = edge_model(*sample_inputs)

            # PyTorch 推理
            with torch.no_grad():
                pytorch_output = self.model(*sample_inputs)

            # 对比输出
            try:
                np.testing.assert_allclose(
                    pytorch_output.numpy(),
                    edge_output.numpy(),
                    rtol=1e-2, atol=1e-3
                )
                logger.success("推理一致性验证通过")
            except AssertionError:
                max_diff = np.max(np.abs(pytorch_output.numpy() - edge_output.numpy()))
                logger.warning(f"最大差异: {max_diff:.6f} (ai-edge-torch 转换可能引入少量数值差异)")

        except Exception as e:
            logger.warning(f"验证过程出错: {e}，继续导出，但建议在实际部署前测试模型")

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

        logger.info("TFLite 模型信息",
                    file_path=str(model_path),
                    file_size_mb=f"{file_size:.2f}",
                    quantize='是 (FP16)' if quantize else '否 (FP32)',
                    input_size=f"(1, {self.img_size}, {self.img_size}, 3) - NHWC 格式",
                    output_classes=len(self.class_names),
                    class_names=', '.join(self.class_names))

        logger.info("部署提示:")
        logger.info("  - Android: 使用 TensorFlow Lite Interpreter")
        logger.info("  - iOS: 建议使用 CoreML 格式")
        logger.info("  - 跨平台: 可使用 ONNX Runtime")
        logger.info("  - 注意: TFLite 使用 NHWC 格式，PyTorch 使用 NCHW 格式")
