"""一键训练流水线脚本 - 完整的训练、评估、导出、测试流程"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


class PipelineRunner:
    """流水线执行器"""

    def __init__(self, args):
        self.args = args
        self.run_dir = None
        self.config = None
        self.checkpoint_path = None
        self.results = {
            'start_time': None,
            'end_time': None,
            'stages': {},
            'errors': []
        }

    def run(self):
        """执行完整流水线"""
        self.results['start_time'] = time.time()

        print("\n" + "="*70)
        print("🚀 模型训练流水线")
        print("="*70)

        try:
            # 阶段1: 初始化
            self._run_stage("初始化", self.setup)

            # 阶段2: 数据准备
            self._run_stage("数据准备", self.prepare_data)

            # 阶段3: 模型训练
            if not self.args.skip_train:
                self._run_stage("模型训练", self.train_model)
            else:
                print("\n[阶段3/8] 模型训练")
                print("  ⊘ 已跳过（使用已有模型）")
                self.load_existing_model()

            # 阶段4: 模型评估
            self._run_stage("模型评估", self.evaluate_model)

            # 阶段5: 模型导出
            if not self.args.skip_export:
                self._run_stage("模型导出", self.export_models)
            else:
                print("\n[阶段5/8] 模型导出")
                print("  ⊘ 已跳过")

            # 阶段6: 测试图片推理
            if not self.args.skip_test:
                self._run_stage("测试图片推理", self.test_inference)
            else:
                print("\n[阶段6/8] 测试图片推理")
                print("  ⊘ 已跳过")

            # 阶段7: 模型对比（可选）
            if self.args.compare_models:
                self._run_stage("模型对比", self.compare_models_perf)
            else:
                print("\n[阶段7/8] 模型对比")
                print("  ⊘ 已跳过")

            # 阶段8: 生成总结报告
            self._run_stage("生成总结报告", self.generate_summary)

            self.results['end_time'] = time.time()
            self.print_final_summary()

        except Exception as e:
            print(f"\n❌ 流水线执行失败: {e}")
            import traceback
            traceback.print_exc()
            self.results['errors'].append(str(e))
            sys.exit(1)

    def _run_stage(self, name, func):
        """运行单个阶段并记录时间"""
        stage_num = len([s for s in self.results['stages'] if self.results['stages'][s].get('completed', False)]) + 1
        print(f"\n[阶段{stage_num}/8] {name}")

        start = time.time()
        try:
            func()
            elapsed = time.time() - start
            self.results['stages'][name] = {
                'completed': True,
                'time': elapsed,
                'error': None
            }
        except Exception as e:
            elapsed = time.time() - start
            self.results['stages'][name] = {
                'completed': False,
                'time': elapsed,
                'error': str(e)
            }
            raise

    def setup(self):
        """阶段1: 初始化"""
        # 创建运行目录
        if self.args.run_name:
            run_name = self.args.run_name
        else:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path(self.args.output_base) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"  ✓ 创建运行目录: {self.run_dir}")

        # 创建子目录
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.run_dir / 'metrics').mkdir(exist_ok=True)
        (self.run_dir / 'visualizations').mkdir(exist_ok=True)
        (self.run_dir / 'exported_models').mkdir(exist_ok=True)
        (self.run_dir / 'test_results').mkdir(exist_ok=True)

        # 保存配置
        config_dict = vars(self.args).copy()
        config_dict['run_dir'] = str(self.run_dir)
        config_dict['timestamp'] = datetime.now().isoformat()

        with open(self.run_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print("  ✓ 保存运行配置")
        print(f"  运行目录: {self.run_dir}")

    def prepare_data(self):
        """阶段2: 数据准备"""
        processed_dir = Path('data/processed')

        # 检查是否需要重新划分
        if not processed_dir.exists() or self.args.force_split:
            print("  正在划分数据集...")
            from src.config.training_config import BaseConfig
            from src.data.split_data import split_dataset

            base_config = BaseConfig()
            split_dataset(base_config)
            print("  ✓ 数据集划分完成")
        else:
            print("  ✓ 数据集已存在")

        # 统计数据
        train_count = len(list((processed_dir / 'train').rglob('*.jpg'))) + \
                     len(list((processed_dir / 'train').rglob('*.png')))
        val_count = len(list((processed_dir / 'val').rglob('*.jpg'))) + \
                   len(list((processed_dir / 'val').rglob('*.png')))
        test_count = len(list((processed_dir / 'test').rglob('*.jpg'))) + \
                    len(list((processed_dir / 'test').rglob('*.png')))

        print(f"  ✓ 训练集: {train_count}张 | 验证集: {val_count}张 | 测试集: {test_count}张")

        self.results['data'] = {
            'train': train_count,
            'val': val_count,
            'test': test_count
        }

    def train_model(self):
        """阶段3: 模型训练"""
        from src.config.training_config import TrainingConfig
        from src.data.dataloader import create_dataloaders
        from src.models.model_factory import create_model
        from src.training.trainer import Trainer

        # 创建配置
        config = TrainingConfig(
            num_epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            learning_rate=self.args.lr,
            data_root='data/processed'
        )

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root='data/processed',
            batch_size=self.args.batch_size,
            num_workers=4,
            img_size=self.args.img_size
        )

        # 创建模型
        model = create_model(
            model_name=self.args.model,
            num_classes=len(config.class_names),
            pretrained=not self.args.no_pretrained,
            dropout=self.args.dropout
        )

        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )

        # 执行训练
        if self.args.two_stage:
            print("  训练模式: 两阶段训练")
            print(f"    阶段1: 冻结主干 ({self.args.stage1_epochs} epochs)")
            print(f"    阶段2: 微调全模型 ({self.args.stage2_epochs} epochs)")

            # 导入两阶段训练函数
            import argparse as ap

            from scripts.train import train_two_stage

            # 创建临时参数
            train_args = ap.Namespace(
                data_root='data/processed',
                model=self.args.model,
                pretrained=not self.args.no_pretrained,
                batch_size=self.args.batch_size,
                lr=self.args.lr,
                stage1_epochs=self.args.stage1_epochs,
                stage2_epochs=self.args.stage2_epochs,
                stage2_lr=self.args.stage2_lr,
                unfreeze_from=self.args.unfreeze_from,
                num_workers=self.args.num_workers,
                device='auto',
                patience=self.args.patience,
                dropout=self.args.dropout,
                img_size=224,
                weight_decay=1e-4
            )

            _, history = train_two_stage(train_args)
        else:
            print(f"  训练模式: 标准训练 ({self.args.epochs} epochs)")
            history = trainer.train()

        # 保存checkpoint到运行目录
        checkpoint_src = Path('data/output/checkpoints/best_model.pth')
        self.checkpoint_path = self.run_dir / 'checkpoints' / 'best_model.pth'

        if checkpoint_src.exists():
            shutil.copy(checkpoint_src, self.checkpoint_path)
            print("  ✓ 模型已保存: checkpoints/best_model.pth")

        # 记录训练结果
        if history:
            best_val_acc = max(history['val_acc']) if 'val_acc' in history else 0
            self.results['training'] = {
                'best_val_acc': best_val_acc,
                'epochs': len(history['val_acc']) if 'val_acc' in history else 0,
                'history': history
            }
            print(f"  ✓ 最佳验证准确率: {best_val_acc*100:.2f}%")

    def load_existing_model(self):
        """加载已有模型"""
        if self.args.checkpoint:
            src_checkpoint = Path(self.args.checkpoint)
        else:
            src_checkpoint = Path('data/output/checkpoints/best_model.pth')

        if not src_checkpoint.exists():
            raise FileNotFoundError(f"找不到模型文件: {src_checkpoint}")

        # 复制到运行目录
        self.checkpoint_path = self.run_dir / 'checkpoints' / 'best_model.pth'
        shutil.copy(src_checkpoint, self.checkpoint_path)

        print(f"  使用模型: {src_checkpoint}")

    def evaluate_model(self):
        """阶段4: 模型评估"""
        from src.data.dataloader import create_dataloaders
        from src.models.model_factory import load_model_from_checkpoint
        from src.training.metrics import MetricsCalculator
        from src.utils.device import get_device
        from src.utils.visualization import Visualizer

        # 加载模型
        model, checkpoint = load_model_from_checkpoint(str(self.checkpoint_path))
        device = get_device()

        # 加载测试集
        _, _, test_loader = create_dataloaders(
            'data/processed',
            batch_size=self.args.batch_size,
            num_workers=4,
            img_size=self.args.img_size
        )

        # 评估
        model.to(device)
        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # 计算指标
        class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']
        metrics_calculator = MetricsCalculator(class_names)
        metrics = metrics_calculator.compute_metrics(y_true, y_pred)

        # 保存结果
        metrics_dir = self.run_dir / 'metrics'
        vis_dir = self.run_dir / 'visualizations'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 保存指标JSON
        metrics_json = {k: v for k, v in metrics.items() if k != 'confusion_matrix_array'}
        with open(metrics_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)

        # 保存分类报告
        with open(metrics_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
            report = metrics_calculator.get_classification_report(y_true, y_pred)
            f.write(report)

        # 生成可视化
        Visualizer.plot_confusion_matrix(
            metrics['confusion_matrix_array'],
            class_names,
            save_path=str(vis_dir / 'confusion_matrix.png')
        )

        Visualizer.plot_per_class_metrics(
            metrics,
            save_path=str(vis_dir / 'per_class_metrics.png')
        )

        # 绘制训练历史（如果有）
        if 'history' in checkpoint:
            Visualizer.plot_training_history(
                checkpoint['history'],
                save_path=str(vis_dir / 'training_history.png')
            )

        accuracy = metrics.get('accuracy', 0)
        print(f"  ✓ 测试集准确率: {accuracy*100:.2f}%")
        print("  ✓ 指标已保存: metrics/")
        print("  ✓ 可视化已生成: visualizations/")

        self.results['evaluation'] = metrics

    def export_models(self):
        """阶段5: 模型导出"""
        export_dir = self.run_dir / 'exported_models'
        export_dir.mkdir(exist_ok=True)

        formats = self.args.export_formats.split()

        for fmt in formats:
            try:
                if fmt == 'onnx':
                    from src.export.onnx_exporter import ONNXExporter
                    from src.models.model_factory import \
                        load_model_from_checkpoint

                    model, _ = load_model_from_checkpoint(str(self.checkpoint_path))
                    exporter = ONNXExporter(model, img_size=224)
                    output_path = export_dir / 'model.onnx'
                    exporter.export(str(output_path))

                    print("  ✓ ONNX导出成功")

                elif fmt == 'coreml':
                    from src.export.coreml_exporter import CoreMLExporter
                    from src.models.model_factory import \
                        load_model_from_checkpoint

                    model, checkpoint = load_model_from_checkpoint(str(self.checkpoint_path))
                    class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']

                    exporter = CoreMLExporter(model, img_size=224, class_names=class_names)
                    output_path = export_dir / 'model.mlpackage'
                    exporter.export(str(output_path), quantize=self.args.quantize)

                    print("  ✓ CoreML导出成功")

                elif fmt == 'tflite':
                    print("  ⚠ TFLite导出需要Python 3.11环境，已跳过")

            except Exception as e:
                print(f"  ✗ {fmt.upper()}导出失败: {e}")

    def test_inference(self):
        """阶段6: 测试图片推理"""
        test_images_dir = Path(self.args.test_images)

        if not test_images_dir.exists():
            print(f"  ⚠ 测试图片目录不存在: {test_images_dir}")
            return

        # 执行批量推理
        import argparse as ap

        test_results_dir = self.run_dir / 'test_results'
        test_results_dir.mkdir(exist_ok=True)

        # 创建推理参数
        infer_args = ap.Namespace(
            checkpoint=str(self.checkpoint_path),
            input_dir=str(test_images_dir),
            output=str(test_results_dir / 'predictions.json'),
            img_size=224,
            batch_size=1,
            copy_to_folders=self.args.copy_images,
            output_dir=str(self.run_dir / 'classified_images'),
            output_csv=str(test_results_dir / 'predictions.csv'),
            measure_time=True
        )

        # 临时修改sys.argv来调用batch_inference
        original_argv = sys.argv
        sys.argv = ['batch_inference.py'] + [
            '--checkpoint', str(infer_args.checkpoint),
            '--input-dir', str(infer_args.input_dir),
            '--output', str(infer_args.output),
            '--measure-time'
        ]

        if infer_args.copy_to_folders:
            sys.argv.extend(['--copy-to-folders', '--output-dir', str(infer_args.output_dir)])

        try:
            # 直接调用推理逻辑（避免重复解析参数）
            import csv

            from PIL import Image
            from tqdm import tqdm

            from src.data.transforms import get_val_transforms
            from src.models.model_factory import load_model_from_checkpoint
            from src.utils.device import get_device

            device = get_device()
            model, checkpoint = load_model_from_checkpoint(str(self.checkpoint_path))
            model.to(device)
            model.eval()

            class_names = checkpoint.get('config').class_names if 'config' in checkpoint else ['Failure', 'Loading', 'Success']
            transform = get_val_transforms(img_size=224)

            # 获取图片
            image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
            image_paths = [p for p in test_images_dir.iterdir() if p.is_file() and p.suffix in image_extensions]

            results = {}
            class_counts = {name: 0 for name in class_names}

            for image_path in tqdm(image_paths, desc="  处理图片"):
                try:
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image)

                    # 推理
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        image_tensor = image_tensor.unsqueeze(0).to(device)
                        outputs = model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    end_time = time.perf_counter()

                    pred_class = class_names[predicted.item()]
                    conf_score = confidence.item()
                    inference_time_ms = (end_time - start_time) * 1000

                    results[image_path.name] = {
                        'predicted_class': pred_class,
                        'confidence': conf_score,
                        'inference_time_ms': inference_time_ms,
                        'probabilities': {
                            class_names[i]: float(probabilities[0][i])
                            for i in range(len(class_names))
                        }
                    }

                    class_counts[pred_class] += 1
                except Exception as e:
                    print(f"    ⚠ {image_path.name} 处理失败: {e}")

            # 保存JSON
            with open(test_results_dir / 'predictions.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 保存CSV
            with open(test_results_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'predicted_class', 'confidence', 'inference_time_ms'])
                writer.writeheader()
                for filename, pred in results.items():
                    writer.writerow({
                        'filename': filename,
                        'predicted_class': pred['predicted_class'],
                        'confidence': f"{pred['confidence']:.4f}",
                        'inference_time_ms': f"{pred['inference_time_ms']:.2f}"
                    })

            # 保存统计摘要
            total = sum(class_counts.values())
            with open(test_results_dir / 'summary.txt', 'w', encoding='utf-8') as f:
                f.write("测试图片分类统计\n")
                f.write("="*50 + "\n")
                for class_name in class_names:
                    count = class_counts[class_name]
                    percentage = (count / total * 100) if total > 0 else 0
                    f.write(f"{class_name}: {count}张 ({percentage:.1f}%)\n")
                f.write(f"总计: {total}张\n")

            # 打印统计
            for class_name in class_names:
                count = class_counts[class_name]
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  ✓ {class_name}: {count}张 ({percentage:.1f}%)")

            print("  ✓ 结果已保存: test_results/")

            self.results['test_inference'] = {
                'total': total,
                'class_counts': class_counts
            }

        finally:
            sys.argv = original_argv

    def compare_models_perf(self):
        """阶段7: 模型对比"""
        # 检查是否有导出的模型
        export_dir = self.run_dir / 'exported_models'

        onnx_path = export_dir / 'model.onnx'
        coreml_path = export_dir / 'model.mlpackage'

        if not onnx_path.exists():
            print("  ⚠ 未找到ONNX模型，跳过对比")
            return

        # 执行对比
        from scripts.compare_models import ModelComparator

        models_config = {
            'pytorch': {'path': str(self.checkpoint_path), 'type': 'checkpoint'},
            'onnx': {'path': str(onnx_path), 'type': 'onnx'}
        }

        class_names = ['Failure', 'Loading', 'Success']
        comparator = ModelComparator(models_config, 'data/processed/test', class_names)

        results = comparator.compare(num_samples=None)

        if results:
            # 保存结果
            comparison_dir = self.run_dir / 'model_comparison'
            comparison_dir.mkdir(exist_ok=True)

            with open(comparison_dir / 'comparison.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 生成Markdown报告
            from scripts.compare_models import generate_markdown_report
            generate_markdown_report(results, comparison_dir / 'comparison.md')

            print("  ✓ 对比报告已保存: model_comparison/")

            self.results['model_comparison'] = results['summary']

    def generate_summary(self):
        """阶段8: 生成总结报告"""
        total_time = time.time() - self.results['start_time']

        # 生成Markdown报告
        report = []
        report.append("# 模型训练流水线运行报告\n")
        report.append(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"运行名称: {self.run_dir.name}")
        report.append(f"总耗时: {format_time(total_time)}\n")
        report.append("---\n")

        # 1. 运行配置
        report.append("## 1. 运行配置\n")
        report.append(f"- 训练模式: {'两阶段训练' if self.args.two_stage else '标准训练'}")
        if self.args.two_stage:
            report.append(f"- 阶段1: {self.args.stage1_epochs} epochs (冻结主干)")
            report.append(f"- 阶段2: {self.args.stage2_epochs} epochs (微调全模型)")
        else:
            report.append(f"- Epochs: {self.args.epochs}")
        report.append(f"- 批次大小: {self.args.batch_size}")
        report.append(f"- 学习率: {self.args.lr}")
        report.append(f"- 导出格式: {self.args.export_formats}\n")
        report.append("---\n")

        # 2. 训练结果
        if 'training' in self.results:
            training = self.results['training']
            report.append("## 2. 训练结果\n")
            report.append(f"- 最佳验证准确率: **{training.get('best_val_acc', 0)*100:.2f}%**")
            report.append(f"- 训练轮数: {training.get('epochs', 0)}")
            if '模型训练' in self.results['stages']:
                train_time = self.results['stages']['模型训练']['time']
                report.append(f"- 训练耗时: {format_time(train_time)}\n")
            report.append("---\n")

        # 3. 评估指标
        if 'evaluation' in self.results:
            eval_metrics = self.results['evaluation']
            report.append("## 3. 评估指标\n")
            report.append(f"- 测试集准确率: **{eval_metrics.get('accuracy', 0)*100:.2f}%**")

            if 'per_class' in eval_metrics:
                report.append("\n### 各类别表现\n")
                report.append("| 类别 | Precision | Recall | F1-Score |")
                report.append("|------|-----------|--------|----------|")
                for class_name, metrics in eval_metrics['per_class'].items():
                    report.append(f"| {class_name} | {metrics['precision']*100:.2f}% | {metrics['recall']*100:.2f}% | {metrics['f1_score']*100:.2f}% |")
            report.append("\n---\n")

        # 4. 测试图片分类
        if 'test_inference' in self.results:
            test_inf = self.results['test_inference']
            report.append("## 4. 测试图片分类\n")
            report.append(f"- 总图片数: {test_inf['total']}张")
            for class_name, count in test_inf['class_counts'].items():
                percentage = (count / test_inf['total'] * 100) if test_inf['total'] > 0 else 0
                report.append(f"- {class_name}: {count}张 ({percentage:.1f}%)")
            report.append("\n---\n")

        # 5. 模型对比
        if 'model_comparison' in self.results:
            report.append("## 5. 模型对比\n")
            comp = self.results['model_comparison']

            report.append("| 模型 | 平均耗时 (ms) |")
            report.append("|------|---------------|")
            for model_name, stats in comp.items():
                if model_name == 'consistency':
                    continue
                report.append(f"| {model_name} | {stats.get('avg_time_ms', 'N/A')} |")
            report.append("\n---\n")

        # 6. 输出文件清单
        report.append("## 6. 输出文件清单\n")
        report.append("```")
        report.append(f"{self.run_dir.name}/")

        for subdir in ['checkpoints', 'metrics', 'visualizations', 'exported_models', 'test_results']:
            path = self.run_dir / subdir
            if path.exists():
                files = list(path.iterdir())
                if files:
                    report.append(f"├── {subdir}/")
                    for file in files[:5]:  # 最多显示5个文件
                        report.append(f"│   ├── {file.name}")

        report.append("└── run_summary.md")
        report.append("```\n")

        report.append("---\n")
        report.append("*报告由 pipeline.py 自动生成*")

        # 保存报告
        with open(self.run_dir / 'run_summary.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print("  ✓ 总结已保存: run_summary.md")

    def print_final_summary(self):
        """打印最终总结"""
        total_time = self.results['end_time'] - self.results['start_time']

        print("\n" + "="*70)
        print("✅ 流水线执行完成！")
        print("="*70)
        print(f"总耗时: {format_time(total_time)}")
        print(f"输出目录: {self.run_dir}")
        print()
        print("查看完整报告:")
        print(f"  cat {self.run_dir}/run_summary.md")
        print()
        if (self.run_dir / 'metrics' / 'test_metrics.json').exists():
            print("查看评估指标:")
            print(f"  cat {self.run_dir}/metrics/test_metrics.json")
        print("="*70 + "\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='一键训练流水线')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--two-stage', action='store_true',
                        help='启用两阶段训练')
    parser.add_argument('--stage1-epochs', type=int, default=10,
                        help='阶段1 epoch数')
    parser.add_argument('--stage2-epochs', type=int, default=20,
                        help='阶段2 epoch数')
    parser.add_argument('--stage2-lr', type=float, default=1e-4,
                        help='阶段2学习率')
    parser.add_argument('--unfreeze-from', type=int, default=14,
                        help='从第几层开始解冻')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        help='模型架构')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='不使用预训练权重')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比例')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')

    # 数据相关
    parser.add_argument('--train-data', type=str, default='data/input/data1226',
                        help='训练数据路径')
    parser.add_argument('--test-images', type=str, default='data/test_images',
                        help='测试图片路径')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--force-split', action='store_true',
                        help='强制重新划分数据集')

    # 导出相关
    parser.add_argument('--export-formats', type=str, default='onnx coreml',
                        help='导出格式（空格分隔）')
    parser.add_argument('--quantize', action='store_true',
                        help='量化模型')

    # 输出相关
    parser.add_argument('--output-base', type=str, default='data/output/runs',
                        help='输出基础目录')
    parser.add_argument('--run-name', type=str, default=None,
                        help='自定义运行名称')
    parser.add_argument('--copy-images', action='store_true',
                        help='复制分类后的图片')

    # 流程控制
    parser.add_argument('--skip-train', action='store_true',
                        help='跳过训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='使用指定checkpoint')
    parser.add_argument('--skip-export', action='store_true',
                        help='跳过模型导出')
    parser.add_argument('--skip-test', action='store_true',
                        help='跳过测试图片推理')
    parser.add_argument('--compare-models', action='store_true',
                        help='对比不同格式模型性能')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    runner = PipelineRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
