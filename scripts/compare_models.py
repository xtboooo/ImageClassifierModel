"""模型对比脚本 - 对比不同格式模型的性能"""
import argparse
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


class ModelLoader:
    """统一的模型加载器"""

    @staticmethod
    def load_pytorch(checkpoint_path):
        """加载PyTorch checkpoint"""
        from src.models.model_factory import load_model_from_checkpoint

        model, checkpoint = load_model_from_checkpoint(checkpoint_path)
        model.eval()
        return model, checkpoint

    @staticmethod
    def load_onnx(onnx_path):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: uv add onnxruntime")

        session = ort.InferenceSession(onnx_path)
        return session

    @staticmethod
    def load_tflite(tflite_path):
        """加载TFLite模型"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("请安装 tensorflow: 需要Python 3.11环境")

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        return interpreter


class InferenceEngine:
    """统一的推理引擎"""

    def __init__(self, model, model_type, img_size=224):
        self.model = model
        self.model_type = model_type
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # PyTorch模型移到设备上
        if model_type == 'pytorch':
            self.model.to(self.device)

    def preprocess(self, image_path):
        """预处理图片"""
        from src.data.transforms import get_val_transforms

        transform = get_val_transforms(self.img_size)
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image)

        if self.model_type == 'pytorch':
            return tensor
        elif self.model_type == 'onnx':
            # ONNX需要numpy数组
            return tensor.numpy()
        elif self.model_type == 'tflite':
            # TFLite需要NHWC格式 + batch维度
            tflite_input = tensor.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            return tflite_input
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def infer(self, input_data, measure_time=True):
        """执行推理"""
        if measure_time:
            if self.model_type == 'pytorch' and self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

        if self.model_type == 'pytorch':
            with torch.no_grad():
                input_tensor = input_data.unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                result = probabilities.cpu().numpy()[0]

        elif self.model_type == 'onnx':
            # ONNX推理
            input_array = input_data.reshape(1, 3, self.img_size, self.img_size).astype(np.float32)
            outputs = self.model.run(None, {'input': input_array})[0]

            # 手动应用softmax
            exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            result = (exp_outputs / exp_outputs.sum(axis=1, keepdims=True))[0]

        elif self.model_type == 'tflite':
            # TFLite推理
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()

            outputs = self.model.get_tensor(output_details[0]['index'])[0]

            # 手动应用softmax
            exp_outputs = np.exp(outputs - np.max(outputs))
            result = exp_outputs / exp_outputs.sum()

        if measure_time:
            end = time.perf_counter()
            inference_time_ms = (end - start) * 1000
        else:
            inference_time_ms = None

        predicted_class = np.argmax(result)
        confidence = result[predicted_class]

        return {
            'class_idx': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': result.tolist(),
            'inference_time_ms': inference_time_ms
        }


class ModelComparator:
    """模型对比器"""

    def __init__(self, models_config, test_images_dir, class_names):
        """
        Args:
            models_config: {
                'pytorch': {'path': '...', 'type': 'checkpoint'},
                'onnx': {'path': '...', 'type': 'onnx'},
            }
            test_images_dir: 测试图片目录
            class_names: ['Failure', 'Loading', 'Success']
        """
        self.models_config = models_config
        self.test_images_dir = Path(test_images_dir)
        self.class_names = class_names
        self.engines = {}

        # 加载所有模型
        self._load_models()

    def _load_models(self):
        """加载所有模型"""
        for name, config in self.models_config.items():
            print(f"加载 {name} 模型...")

            try:
                if config['type'] == 'checkpoint':
                    model, _ = ModelLoader.load_pytorch(config['path'])
                    engine = InferenceEngine(model, 'pytorch')
                elif config['type'] == 'onnx':
                    model = ModelLoader.load_onnx(config['path'])
                    engine = InferenceEngine(model, 'onnx')
                elif config['type'] == 'tflite':
                    model = ModelLoader.load_tflite(config['path'])
                    engine = InferenceEngine(model, 'tflite')
                else:
                    raise ValueError(f"不支持的模型类型: {config['type']}")

                self.engines[name] = engine
                print(f"  ✓ {name} 模型加载成功")
            except Exception as e:
                print(f"  ✗ {name} 模型加载失败: {e}")
                print(f"     跳过 {name} 模型的对比")

    def get_model_size(self, model_path):
        """获取模型大小（MB）"""
        import os
        path = Path(model_path)

        if path.is_dir():  # CoreML mlpackage
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        else:
            # 对于ONNX，可能有外部数据文件
            total_size = os.path.getsize(model_path)
            data_file = path.with_suffix('.onnx.data')
            if data_file.exists():
                total_size += os.path.getsize(data_file)

        return total_size / (1024 * 1024)

    def compare(self, num_samples=None):
        """执行对比"""
        if not self.engines:
            print("❌ 没有成功加载任何模型，无法对比")
            return None

        # 获取测试图片
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_paths = [
            p for p in self.test_images_dir.rglob('*')
            if p.is_file() and p.suffix in image_extensions
        ]

        if num_samples:
            image_paths = image_paths[:num_samples]

        print(f"\n开始对比 {len(image_paths)} 张图片...")

        results = {
            'model_info': {},
            'inference_results': [],
            'summary': {}
        }

        # 模型信息
        for name, config in self.models_config.items():
            if name not in self.engines:
                continue  # 跳过加载失败的模型

            size_mb = self.get_model_size(config['path'])
            results['model_info'][name] = {
                'path': str(config['path']),
                'size_mb': f"{size_mb:.2f}",
                'type': config['type']
            }

        # 逐张推理
        inference_times = {name: [] for name in self.engines.keys()}
        predictions = {name: [] for name in self.engines.keys()}

        for img_path in tqdm(image_paths, desc="推理中"):
            img_result = {'image': img_path.name}

            for model_name, engine in self.engines.items():
                try:
                    input_data = engine.preprocess(img_path)
                    pred = engine.infer(input_data, measure_time=True)

                    img_result[model_name] = {
                        'class': self.class_names[pred['class_idx']],
                        'confidence': f"{pred['confidence']:.4f}",
                        'time_ms': f"{pred['inference_time_ms']:.2f}"
                    }

                    inference_times[model_name].append(pred['inference_time_ms'])
                    predictions[model_name].append(pred['class_idx'])
                except Exception as e:
                    print(f"\n⚠️  {img_path.name} 在 {model_name} 上推理失败: {e}")
                    img_result[model_name] = {
                        'class': 'ERROR',
                        'confidence': '0.0000',
                        'time_ms': '0.00'
                    }

            results['inference_results'].append(img_result)

        # 计算一致性
        consistency = self._calculate_consistency(predictions)

        # 汇总统计
        for model_name in self.engines.keys():
            times = inference_times[model_name]
            if times:
                results['summary'][model_name] = {
                    'avg_time_ms': f"{np.mean(times):.2f}",
                    'min_time_ms': f"{np.min(times):.2f}",
                    'max_time_ms': f"{np.max(times):.2f}",
                    'std_time_ms': f"{np.std(times):.2f}",
                    'total_images': len(times)
                }

        results['summary']['consistency'] = consistency

        return results

    def _calculate_consistency(self, predictions):
        """计算预测一致性"""
        model_names = list(predictions.keys())

        if len(model_names) < 2:
            return {"note": "只有一个模型，无法计算一致性"}

        total_samples = len(predictions[model_names[0]])

        if total_samples == 0:
            return {"note": "没有有效的预测结果"}

        consistency = {}

        # 两两对比
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                pred1 = np.array(predictions[name1])
                pred2 = np.array(predictions[name2])

                agreement = np.sum(pred1 == pred2)
                percentage = agreement / total_samples * 100

                key = f"{name1}_vs_{name2}"
                consistency[key] = f"{percentage:.1f}%"

        # 全体一致性
        if len(model_names) >= 2:
            all_preds = np.array([predictions[name] for name in model_names])
            all_agree = np.all(all_preds == all_preds[0], axis=0)
            full_consistency = np.sum(all_agree) / total_samples * 100
            consistency['all_models'] = f"{full_consistency:.1f}%"

        return consistency


def generate_markdown_report(results, output_path):
    """生成Markdown格式报告"""

    report = ["# 模型对比报告\n"]
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 模型信息
    report.append("## 1. 模型信息\n")
    report.append("| 模型 | 类型 | 文件大小 (MB) | 路径 |")
    report.append("|------|------|---------------|------|")

    for name, info in results['model_info'].items():
        report.append(f"| {name} | {info['type']} | {info['size_mb']} | `{info['path']}` |")

    report.append("\n")

    # 2. 性能对比
    report.append("## 2. 推理性能\n")
    report.append("| 模型 | 平均耗时 (ms) | 最小耗时 (ms) | 最大耗时 (ms) | 标准差 (ms) | 总图片数 |")
    report.append("|------|---------------|---------------|---------------|-------------|----------|")

    for name, stats in results['summary'].items():
        if name == 'consistency':
            continue
        report.append(
            f"| {name} | {stats['avg_time_ms']} | {stats['min_time_ms']} | "
            f"{stats['max_time_ms']} | {stats['std_time_ms']} | {stats['total_images']} |"
        )

    report.append("\n")

    # 3. 预测一致性
    report.append("## 3. 预测一致性\n")
    report.append("| 对比 | 一致率 |")
    report.append("|------|--------|")

    consistency = results['summary']['consistency']
    for pair, rate in consistency.items():
        report.append(f"| {pair} | {rate} |")

    report.append("\n")

    # 4. 示例预测（前10张）
    report.append("## 4. 示例预测 (前10张)\n")

    model_names = list(results['model_info'].keys())
    header = "| 图片 | " + " | ".join(model_names) + " |"
    separator = "|------|" + "|".join(["------"] * len(model_names)) + "|"

    report.append(header)
    report.append(separator)

    for result in results['inference_results'][:10]:
        row = f"| {result['image']} |"
        for name in model_names:
            if name in result:
                pred = result[name]
                row += f" {pred['class']} ({pred['confidence']}, {pred['time_ms']}ms) |"
            else:
                row += " N/A |"
        report.append(row)

    report.append("\n---\n")
    report.append("*报告由 `compare_models.py` 自动生成*")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n✓ Markdown报告已生成: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对比不同格式的模型性能')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='PyTorch checkpoint路径')
    parser.add_argument('--onnx', type=str, required=True,
                        help='ONNX模型路径')
    parser.add_argument('--tflite', type=str, default=None,
                        help='TFLite模型路径（可选，需要Python 3.11环境）')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='测试图片目录')
    parser.add_argument('--output-dir', type=str, default='data/output/model_comparison',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='测试样本数量（默认全部）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*70)
    print("模型对比")
    print("="*70)
    print(f"PyTorch Checkpoint: {args.checkpoint}")
    print(f"ONNX 模型: {args.onnx}")
    print(f"TFLite 模型: {args.tflite or '未指定'}")
    print(f"测试目录: {args.test_dir}")
    print("="*70 + "\n")

    # 配置模型
    models_config = {
        'pytorch': {'path': args.checkpoint, 'type': 'checkpoint'},
        'onnx': {'path': args.onnx, 'type': 'onnx'}
    }

    if args.tflite:
        models_config['tflite'] = {'path': args.tflite, 'type': 'tflite'}

    class_names = ['Failure', 'Loading', 'Success']

    # 创建对比器
    comparator = ModelComparator(models_config, args.test_dir, class_names)

    # 执行对比
    results = comparator.compare(num_samples=args.num_samples)

    if results is None:
        print("\n❌ 对比失败")
        sys.exit(1)

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON
    json_path = output_dir / f"comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON报告已保存: {json_path}")

    # 生成Markdown报告
    md_path = output_dir / f"comparison_{timestamp}.md"
    generate_markdown_report(results, md_path)

    # 打印摘要
    print("\n" + "="*70)
    print("对比摘要")
    print("="*70)

    print("\n模型大小:")
    for name, info in results['model_info'].items():
        print(f"  {name:<10} {info['size_mb']:>8} MB")

    print("\n平均推理速度:")
    for name, stats in results['summary'].items():
        if name == 'consistency':
            continue
        print(f"  {name:<10} {stats['avg_time_ms']:>8} ms")

    print("\n预测一致性:")
    for pair, rate in results['summary']['consistency'].items():
        print(f"  {pair:<25} {rate}")

    print("="*70 + "\n")

    print(f"✅ 对比完成！报告已保存到: {output_dir}\n")


if __name__ == '__main__':
    main()
