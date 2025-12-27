"""模型导出脚本 - 导出为 ONNX/CoreML/TFLite 格式"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_factory import load_model_from_checkpoint
from src.export.onnx_exporter import ONNXExporter
from src.export.coreml_exporter import CoreMLExporter
from src.export.tflite_exporter import TFLiteExporter


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

    return parser.parse_args()


def export_onnx(model, output_path, img_size=224):
    """导出 ONNX 模型"""
    try:
        exporter = ONNXExporter(model, img_size=img_size)
        exporter.export(output_path)
        return True
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        return False


def export_coreml(model, output_path, img_size=224, class_names=None, quantize=False):
    """导出 CoreML 模型"""
    try:
        exporter = CoreMLExporter(model, img_size=img_size, class_names=class_names)
        exporter.export(output_path, quantize=quantize)
        return True
    except Exception as e:
        print(f"❌ CoreML 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_tflite(model, output_path, img_size=224, class_names=None, quantize=False):
    """导出 TFLite 模型"""
    try:
        exporter = TFLiteExporter(model, img_size=img_size, class_names=class_names)
        exporter.export(output_path, quantize=quantize)
        return True
    except Exception as e:
        print(f"❌ TFLite 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*70)
    print("模型导出")
    print("="*70)
    print(f"检查点: {args.checkpoint}")
    print(f"导出格式: {', '.join(args.formats).upper()}")
    print(f"输出目录: {args.output_dir}")
    print(f"量化: {'是' if args.quantize else '否'}")
    print("="*70 + "\n")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("加载模型...")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint)

    # 获取类别名称
    class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']

    # 导出结果统计
    results = {}

    # 导出各种格式
    for fmt in args.formats:
        output_path = output_dir / f"{args.model_name}.{fmt if fmt != 'coreml' else 'mlpackage'}"

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
            output_path = output_dir / f"{args.model_name}.tflite"
            success = export_tflite(
                model, str(output_path),
                img_size=args.img_size,
                class_names=class_names,
                quantize=args.quantize
            )
            results['tflite'] = success

    # 打印总结
    print("\n" + "="*70)
    print("导出总结")
    print("="*70)

    success_count = sum(results.values())
    total_count = len(results)

    for fmt, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {fmt.upper():<10} {status}")

    print("-"*70)
    print(f"总计: {success_count}/{total_count} 成功")
    print("="*70 + "\n")

    if success_count > 0:
        print(f"✅ 模型已导出到: {output_dir}\n")

        # 使用指南
        print("📱 移动端集成指南:")
        if 'onnx' in results and results['onnx']:
            print("  • ONNX: 可用于跨平台部署，或转换为其他格式")
        if 'coreml' in results and results['coreml']:
            print("  • CoreML: 可直接在 iOS/macOS 应用中使用")
            print("    导入步骤: 将 .mlpackage 文件拖入 Xcode 项目")
        if 'tflite' in results and results['tflite']:
            print("  • TFLite: 可在 Android 应用中使用 TensorFlow Lite")
        print()

    sys.exit(0 if success_count == total_count else 1)


if __name__ == '__main__':
    main()
