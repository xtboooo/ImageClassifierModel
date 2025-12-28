"""模型导出脚本 - 导出为 ONNX/CoreML/TFLite 格式"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_factory import load_model_from_checkpoint
from src.export.onnx_exporter import ONNXExporter
from src.export.coreml_exporter import CoreMLExporter

# 导入日志和Rich工具
from src.utils.logger import setup_logger, logger
from src.utils.rich_console import print_header, print_table, print_panel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='导出训练好的模型')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--formats', type=str, nargs='+',
                        default=['onnx', 'coreml'],
                        choices=['onnx', 'coreml', 'tflite'],
                        help='导出格式 (默认: onnx coreml)')
    parser.add_argument('--output-dir', type=str, default='data/output/exported_models',
                        help='输出目录')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--quantize', action='store_true',
                        help='量化模型（减小大小）')
    parser.add_argument('--model-name', type=str, default='model',
                        help='导出的模型名称前缀')
    parser.add_argument('--use-timestamp', action='store_true',
                        help='使用时间戳命名避免覆盖已有模型')

    return parser.parse_args()


def export_onnx(model, output_path, img_size=224):
    """导出 ONNX 模型"""
    try:
        exporter = ONNXExporter(model, img_size=img_size)
        exporter.export(output_path)
        return True
    except Exception as e:
        logger.error(f"ONNX 导出失败: {e}")
        return False


def export_coreml(model, output_path, img_size=224, class_names=None, quantize=False):
    """导出 CoreML 模型"""
    try:
        exporter = CoreMLExporter(model, img_size=img_size, class_names=class_names)
        exporter.export(output_path, quantize=quantize)
        return True
    except Exception as e:
        logger.exception(f"CoreML 导出失败: {e}")
        return False


def export_tflite(model, output_path, img_size=224, class_names=None, quantize=False, checkpoint_path=None):
    """导出 TFLite 模型（统一使用 Docker 导出）"""
    import platform
    import subprocess

    # 检测平台
    current_platform = platform.system()
    logger.info(f"检测到系统: {current_platform}")

    # 统一使用 Docker 导出（避免 ONNX 版本转换问题）
    logger.info("使用 Docker 导出 TFLite...")

    # Docker 脚本路径
    script_path = Path(__file__).parent.parent / 'docker' / 'export_tflite.sh'

    if not script_path.exists():
        logger.error(f"Docker 导出脚本不存在: {script_path}")
        logger.info("请确保项目中存在 docker/export_tflite.sh 文件")
        return False

    # 检查 checkpoint 路径
    if checkpoint_path is None:
        logger.error("必须提供 checkpoint_path 参数")
        return False

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint 文件不存在: {checkpoint_path}")
        return False

    # 运行 Docker 导出
    try:
        logger.info("启动 Docker 导出...")
        logger.debug(f"脚本: {script_path}")
        logger.debug(f"输出: {output_path}")

        # 使用实时输出（不捕获输出）
        result = subprocess.run(
            ['bash', str(script_path), str(checkpoint_path), str(output_path)],
            timeout=600  # 10 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"Docker 导出失败 (退出码: {result.returncode})")
            return False

        logger.success(f"Docker TFLite 导出成功: {output_path}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Docker 导出超时（超过 10 分钟）")
        return False
    except Exception as e:
        logger.exception(f"Docker 导出失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录并初始化日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化日志系统
    run_dir = output_dir.parent  # 日志保存在 data/output
    setup_logger(run_dir, console_level="INFO", file_level="DEBUG")

    print_header("模型导出")

    # 准备配置数据
    rows = [
        ["检查点", args.checkpoint],
        ["导出格式", ', '.join(args.formats).upper()],
        ["输出目录", args.output_dir],
        ["量化", "是" if args.quantize else "否"],
        ["使用时间戳", "是" if args.use_timestamp else "否"]
    ]

    print_table(
        title="导出配置",
        headers=["参数", "值"],
        rows=rows
    )

    # 生成时间戳和基础文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.model_name}_{timestamp}" if args.use_timestamp else args.model_name

    # 备份checkpoint（如果使用时间戳）
    if args.use_timestamp:
        import shutil
        checkpoint_backup = output_dir / f"{args.model_name}_{timestamp}.pth"
        shutil.copy(args.checkpoint, checkpoint_backup)
        logger.success(f"检查点已备份到: {checkpoint_backup}")

    # 加载模型
    logger.info("加载模型...")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint)

    # 获取类别名称
    class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']

    # 导出结果统计
    results = {}

    # 导出各种格式
    for fmt in args.formats:
        # 根据格式确定输出路径
        if fmt == 'onnx':
            output_path = output_dir / f"{base_name}.onnx"
        elif fmt == 'coreml':
            output_path = output_dir / f"{base_name}.mlpackage"
        elif fmt == 'tflite':
            output_path = output_dir / f"{base_name}.tflite"

        if fmt == 'onnx':
            success = export_onnx(model, str(output_path), img_size=args.img_size)
            results['onnx'] = success

        elif fmt == 'coreml':
            success = export_coreml(
                model, str(output_path),
                img_size=args.img_size,
                class_names=class_names,
                quantize=args.quantize
            )
            results['coreml'] = success

        elif fmt == 'tflite':
            success = export_tflite(
                model, str(output_path),
                img_size=args.img_size,
                class_names=class_names,
                quantize=args.quantize,
                checkpoint_path=args.checkpoint
            )
            results['tflite'] = success

    # 打印总结
    success_count = sum(results.values())
    total_count = len(results)

    # 准备结果表格
    result_rows = []
    for fmt, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        status_style = "[green]✓ 成功[/green]" if success else "[red]✗ 失败[/red]"
        result_rows.append([fmt.upper(), status])

    print_table(
        title="导出总结",
        headers=["格式", "状态"],
        rows=result_rows,
        caption=f"总计: {success_count}/{total_count} 成功"
    )

    if success_count > 0:
        logger.success(f"模型已导出到: {output_dir}")

        # 使用指南
        guide_content = "[bold cyan]移动端集成指南:[/bold cyan]\n"
        if 'onnx' in results and results['onnx']:
            guide_content += "\n• [yellow]ONNX[/yellow]: 可用于跨平台部署，或转换为其他格式"
        if 'coreml' in results and results['coreml']:
            guide_content += "\n• [yellow]CoreML[/yellow]: 可直接在 iOS/macOS 应用中使用"
            guide_content += "\n  导入步骤: 将 .mlpackage 文件拖入 Xcode 项目"
        if 'tflite' in results and results['tflite']:
            guide_content += "\n• [yellow]TFLite[/yellow]: 可在 Android 应用中使用 TensorFlow Lite"

        print_panel(guide_content, title="使用指南", style="cyan")

    sys.exit(0 if success_count == total_count else 1)


if __name__ == '__main__':
    main()
