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
from src.utils.logger import logger
from src.utils.rich_console import (
    RichProgressManager,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_table
)


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
            raise ImportError(
                "请安装 tensorflow: uv sync --extra tflite\n"
                "或者: uv pip install tensorflow>=2.16.0"
            )
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
            # TFLite（ai-edge-torch导出）使用NCHW格式（PyTorch格式）+ batch维度
            tflite_input = tensor.unsqueeze(0).numpy().astype(np.float32)
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
            logger.info("加载模型", model_name=name, type=config['type'])

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
                print_success(f"{name} 模型加载成功")
            except Exception as e:
                print_error(f"{name} 模型加载失败: {e}")
                logger.error("模型加载失败", model_name=name, error=str(e))
                print_warning(f"跳过 {name} 模型的对比")

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

    def compare(self, num_samples=None, output_dir=None):
        """执行对比

        Args:
            num_samples: 限制测试样本数量
            output_dir: 输出目录（如果提供，会保存CSV文件）
        """
        if not self.engines:
            print_error("没有成功加载任何模型，无法对比")
            return None

        # 获取测试图片
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_paths = [
            p for p in self.test_images_dir.rglob('*')
            if p.is_file() and p.suffix in image_extensions
        ]

        if num_samples:
            image_paths = image_paths[:num_samples]

        logger.info("开始模型对比", total_images=len(image_paths), num_models=len(self.engines))

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

        with RichProgressManager() as progress:
            task = progress.add_task("推理中", total=len(image_paths))

            for img_path in image_paths:
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
                        print_warning(f"{img_path.name} 在 {model_name} 上推理失败: {e}")
                        logger.warning("推理失败", image=img_path.name, model=model_name, error=str(e))
                        img_result[model_name] = {
                            'class': 'ERROR',
                            'confidence': '0.0000',
                            'time_ms': '0.00'
                        }

                results['inference_results'].append(img_result)
                progress.update("推理中", advance=1)

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

        # 保存CSV文件（如果提供了输出目录）
        if output_dir:
            self._save_csv_results(results, output_dir)

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

    def _save_csv_results(self, results, output_dir):
        """保存CSV结果文件

        Args:
            results: 对比结果字典
            output_dir: 输出目录
        """
        import csv

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_names = list(results['model_info'].keys())

        # 1. 为每个模型生成单独的CSV文件
        for model_name in model_names:
            csv_path = output_path / f'predictions_{model_name}.csv'

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['filename', 'predicted_class', 'confidence', 'inference_time_ms']
                )
                writer.writeheader()

                for result in results['inference_results']:
                    if model_name in result:
                        writer.writerow({
                            'filename': result['image'],
                            'predicted_class': result[model_name]['class'],
                            'confidence': result[model_name]['confidence'],
                            'inference_time_ms': result[model_name]['time_ms']
                        })

            print_success(f"已保存 {model_name} 预测结果: {csv_path.name}")

        # 2. 生成总的比较CSV文件
        comparison_csv_path = output_path / 'predictions_comparison.csv'

        with open(comparison_csv_path, 'w', newline='', encoding='utf-8') as f:
            # 构建表头
            fieldnames = ['filename']
            for model_name in model_names:
                fieldnames.extend([
                    f'{model_name}_class',
                    f'{model_name}_confidence',
                    f'{model_name}_time_ms'
                ])

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # 写入数据
            for result in results['inference_results']:
                row = {'filename': result['image']}

                for model_name in model_names:
                    if model_name in result:
                        row[f'{model_name}_class'] = result[model_name]['class']
                        row[f'{model_name}_confidence'] = result[model_name]['confidence']
                        row[f'{model_name}_time_ms'] = result[model_name]['time_ms']
                    else:
                        row[f'{model_name}_class'] = 'N/A'
                        row[f'{model_name}_confidence'] = 'N/A'
                        row[f'{model_name}_time_ms'] = 'N/A'

                writer.writerow(row)

        print_success(f"已保存比较结果: {comparison_csv_path.name}")
        logger.info("CSV文件已保存", output_dir=str(output_path))


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

    print_success(f"Markdown报告已生成: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对比不同格式的模型性能')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='PyTorch checkpoint路径')
    parser.add_argument('--onnx', type=str, required=True,
                        help='ONNX模型路径')
    parser.add_argument('--tflite', type=str, required=True,
                        help='TFLite模型路径')
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

    print_header("模型对比", "对比不同格式模型的性能和一致性")

    print_table(
        title="对比配置",
        headers=["配置项", "值"],
        rows=[
            ["PyTorch Checkpoint", args.checkpoint],
            ["ONNX 模型", args.onnx],
            ["TFLite 模型", args.tflite],
            ["测试目录", args.test_dir]
        ]
    )

    # 配置模型
    models_config = {
        'pytorch': {'path': args.checkpoint, 'type': 'checkpoint'},
        'onnx': {'path': args.onnx, 'type': 'onnx'},
        'tflite': {'path': args.tflite, 'type': 'tflite'}
    }

    class_names = ['Failure', 'Loading', 'Success']

    # 创建对比器
    comparator = ModelComparator(models_config, args.test_dir, class_names)

    # 执行对比
    results = comparator.compare(num_samples=args.num_samples)

    if results is None:
        print_error("对比失败")
        logger.error("模型对比失败")
        sys.exit(1)

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON
    json_path = output_dir / f"comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print_success(f"JSON报告已保存: {json_path}")

    # 生成Markdown报告
    md_path = output_dir / f"comparison_{timestamp}.md"
    generate_markdown_report(results, md_path)

    # 打印摘要
    print_header("对比摘要")

    # 模型大小表格
    size_rows = [[name, f"{info['size_mb']} MB"] for name, info in results['model_info'].items()]
    print_table(
        title="模型大小",
        headers=["模型", "大小"],
        rows=size_rows
    )

    # 推理速度表格
    speed_rows = [
        [name, f"{stats['avg_time_ms']} ms"]
        for name, stats in results['summary'].items()
        if name != 'consistency'
    ]
    print_table(
        title="平均推理速度",
        headers=["模型", "平均耗时"],
        rows=speed_rows
    )

    # 预测一致性表格
    consistency_rows = [[pair, rate] for pair, rate in results['summary']['consistency'].items()]
    print_table(
        title="预测一致性",
        headers=["对比", "一致率"],
        rows=consistency_rows
    )

    print_success(f"对比完成！报告已保存到: {output_dir}")
    logger.info("模型对比完成", output_dir=str(output_dir))


if __name__ == '__main__':
    main()
