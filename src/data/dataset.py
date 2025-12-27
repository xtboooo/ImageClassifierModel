"""手机截图数据集"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ScreenshotDataset(Dataset):
    """
    手机截图数据集

    数据集结构:
        data_dir/
            Failure/
                img1.jpg
                img2.jpg
            Loading/
                img3.jpg
            Success/
                img4.jpg
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: 数据目录路径（包含Failure/Loading/Success子文件夹）
            transform: 数据增强/预处理transformations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 类别定义（按字母序排列以保证一致性）
        self.classes = sorted(['Failure', 'Loading', 'Success'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # 加载所有图片路径和标签
        self.samples = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"在目录 {data_dir} 中未找到任何图片")

    def _load_samples(self):
        """扫描目录并加载所有样本"""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name

            if not class_dir.exists():
                print(f"警告: 类别目录 {class_dir} 不存在，跳过")
                continue

            # 支持的图片格式
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

            # 加载所有图片
            for ext in image_extensions:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            tuple: (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # 加载图片并转换为RGB（处理RGBA图片）
        image = Image.open(img_path).convert('RGB')

        # 应用数据增强/预处理
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """
        获取类别分布

        Returns:
            dict: {类别名: 样本数量}
        """
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution

    @property
    def num_classes(self):
        """返回类别数量"""
        return len(self.classes)
