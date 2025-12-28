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

    def export(self, save_path, quantize=False, precision='fp32'):
        """
        导出为 TFLite 格式（仅支持 Linux 系统，其他系统请使用 Docker）

        Args:
            save_path: 保存路径
            quantize: 是否进行量化（向后兼容，等同于precision='int8'）
            precision: 精度选项 ('fp32', 'fp16', 'int8')

        Returns:
            str: 保存路径
        """
        import platform

        # quantize参数向后兼容
        if quantize and precision == 'fp32':
            precision = 'int8'

        current_platform = platform.system()

        # 非 Linux 系统，拒绝直接导出，引导用户使用 Docker
        if current_platform != 'Linux':
            error_msg = (
                f"TFLite 导出失败: ai-edge-torch 仅支持 Linux 系统\n"
                f"当前系统: {current_platform}\n"
                f"请使用 Docker 导出:\n"
                f"   bash docker/export_tflite.sh <checkpoint_path> <output_path> <precision>\n"
                f"示例:\n"
                f"   bash docker/export_tflite.sh data/output/checkpoints/best_model.pth {save_path} {precision}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Linux 系统，使用 ai-edge-torch 导出
        logger.info("使用 ai-edge-torch 导出 TFLite 模型",
                   save_path=str(save_path),
                   precision=precision.upper())

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
            self._verify_model_tflite(save_path, sample_inputs)

            # 打印模型信息
            self._print_model_info(save_path, 'fp32')

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

    def _verify_model_tflite(self, tflite_path, sample_inputs):
        """
        验证TFLite模型

        Args:
            tflite_path: TFLite模型路径
            sample_inputs: 示例输入
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("TensorFlow 未安装，跳过验证")
            return

        logger.info("验证 TFLite 模型...")

        try:
            # 加载TFLite模型
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()

            # 获取输入输出详情
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # 准备输入数据
            input_data = sample_inputs[0].numpy().astype(np.float32)

            # 设置输入
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # 运行推理
            interpreter.invoke()

            # 获取输出
            output_data = interpreter.get_tensor(output_details[0]['index'])

            logger.success("TFLite 模型验证通过")

        except Exception as e:
            logger.warning(f"TFLite 验证失败: {e}")

    def _print_model_info(self, model_path, precision='fp32'):
        """
        打印 TFLite 模型信息

        Args:
            model_path: 模型路径
            precision: 精度选项
        """
        import os

        # 文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

        logger.info("TFLite 模型信息",
                    file_path=str(model_path),
                    file_size_mb=f"{file_size:.2f}",
                    precision=precision.upper(),
                    input_size=f"(1, {self.img_size}, {self.img_size}, 3) - NHWC 格式",
                    output_classes=len(self.class_names),
                    class_names=', '.join(self.class_names))

        logger.info("部署提示:")
        logger.info("  - Android: 使用 TensorFlow Lite Interpreter")
        logger.info("  - iOS: 建议使用 CoreML 格式")
        logger.info("  - 跨平台: 可使用 ONNX Runtime")
        logger.info("  - 注意: TFLite 使用 NHWC 格式，PyTorch 使用 NCHW 格式")
