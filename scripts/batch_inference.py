"""批量图片推理脚本"""
import argparse
import sys
from pathlib import Path
import json
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from tqdm import tqdm

from src.models.model_factory import load_model_from_checkpoint
from src.data.transforms import get_val_transforms
from src.utils.device import get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量图片推理')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入图片目录')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='输出结果文件路径')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批次大小（默认1，逐张处理）')
    parser.add_argument('--copy-to-folders', action='store_true',
                        help='是否将图片复制到分类文件夹')
    parser.add_argument('--output-dir', type=str, default='data/output/classified_images',
                        help='分类图片输出目录')

    return parser.parse_args()


def load_image(image_path, transform):
    """
    加载并预处理图片

    Args:
        image_path: 图片路径
        transform: 预处理变换

    Returns:
        tensor: 预处理后的图片张量
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        return image_tensor
    except Exception as e:
        print(f"加载图片失败 {image_path}: {e}")
        return None


def predict_single(model, image_tensor, device, class_names):
    """
    对单张图片进行预测

    Args:
        model: 模型
        image_tensor: 图片张量
        device: 设备
        class_names: 类别名称列表

    Returns:
        dict: 预测结果
    """
    model.eval()

    with torch.no_grad():
        # 添加 batch 维度
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # 前向传播
        outputs = model(image_tensor)

        # 计算概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # 获取预测类别和置信度
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()

        # 获取所有类别的概率
        all_probs = {
            class_names[i]: float(probabilities[0][i])
            for i in range(len(class_names))
        }

        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'probabilities': all_probs
        }


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*70)
    print("批量图片推理")
    print("="*70)
    print(f"检查点: {args.checkpoint}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output}")
    print("="*70 + "\n")

    # 获取设备
    device = get_device()

    # 加载模型
    print("加载模型...")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    # 获取类别名称
    class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']
    print(f"类别: {class_names}\n")

    # 获取预处理变换
    transform = get_val_transforms(img_size=args.img_size)

    # 获取所有图片路径
    input_dir = Path(args.input_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]

    print(f"找到 {len(image_paths)} 张图片\n")

    if len(image_paths) == 0:
        print("没有找到任何图片文件！")
        return

    # 批量推理
    results = {}
    class_counts = {name: 0 for name in class_names}

    print("开始推理...")
    for image_path in tqdm(image_paths, desc="处理图片"):
        # 加载图片
        image_tensor = load_image(image_path, transform)

        if image_tensor is None:
            continue

        # 预测
        prediction = predict_single(model, image_tensor, device, class_names)

        # 保存结果
        results[str(image_path.name)] = prediction

        # 统计类别
        class_counts[prediction['predicted_class']] += 1

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 推理完成！结果已保存到: {output_path}")

    # 打印统计信息
    print("\n" + "="*70)
    print("分类统计")
    print("="*70)

    total = sum(class_counts.values())
    for class_name in class_names:
        count = class_counts[class_name]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{class_name:<15} {count:>4} 张 ({percentage:>5.1f}%)")

    print("-"*70)
    print(f"{'总计':<15} {total:>4} 张")
    print("="*70 + "\n")

    # 打印一些示例预测
    print("示例预测 (前10张):")
    print("-"*70)
    for i, (filename, prediction) in enumerate(list(results.items())[:10]):
        pred_class = prediction['predicted_class']
        confidence = prediction['confidence'] * 100
        print(f"{filename:<30} → {pred_class:<10} (置信度: {confidence:.1f}%)")
    print("="*70 + "\n")

    # 如果需要，将图片复制到分类文件夹
    if args.copy_to_folders:
        print("开始复制图片到分类文件夹...")
        output_dir = Path(args.output_dir)

        # 创建分类文件夹
        for class_name in class_names:
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片
        copied_count = 0
        for image_path in tqdm(image_paths, desc="复制图片"):
            filename = image_path.name
            if filename in results:
                pred_class = results[filename]['predicted_class']
                dest_path = output_dir / pred_class / filename

                # 复制文件
                shutil.copy2(image_path, dest_path)
                copied_count += 1

        print(f"\n✓ 已复制 {copied_count} 张图片到: {output_dir}")
        print("\n文件夹结构:")
        for class_name in class_names:
            class_dir = output_dir / class_name
            count = len(list(class_dir.glob('*')))
            print(f"  {class_dir}/  ({count} 张)")
        print()


if __name__ == '__main__':
    main()
