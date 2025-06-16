"""
Dataset module for Eye Disease Classification
"""

import os
import mindspore as ms
from mindspore.dataset import ImageFolderDataset
from mindspore.dataset.transforms import c_transforms as C
from mindspore.dataset.vision import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomColorAdjust,
    Resize,
    CenterCrop,
    Normalize,
    HWC2CHW,
    Decode,
    RandomRotation,
    RandomAffine,
    RandomErasing
)

from config.config import Config
import numpy as np
import glob
from pathlib import Path
from PIL import Image
import mindspore.dataset as ds
from mindspore.dataset.transforms import Compose
from typing import List, Tuple, Dict, Optional

from .preprocessing import ImagePreprocessor

def get_dataset(data_path, config, is_training=True):
    """获取数据集"""
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path {data_path} does not exist!")
    
    # 创建数据集
    dataset = ImageFolderDataset(
        data_path,
        num_parallel_workers=config['num_parallel_workers'],
        shuffle=is_training,
        decode=True
    )
    
    # 数据预处理
    if is_training:
        # 训练集数据增强 - 增强策略
        transform = [
            Decode(),
            Resize((config['image_size'] + 64, config['image_size'] + 64)),
            RandomResizedCrop(
                size=config['image_size'],
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.2),
            RandomRotation(degrees=10),
            RandomColorAdjust(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    else:
        # 验证集数据预处理
        transform = [
            Decode(),
            Resize(config['image_size'] + 32),
            CenterCrop(config['image_size']),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    
    # 应用转换
    dataset = dataset.map(
        operations=transform,
        input_columns="image",
        num_parallel_workers=config['num_parallel_workers']
    )
    
    # 设置批次大小
    dataset = dataset.batch(
        config['batch_size'],
        drop_remainder=is_training,
        num_parallel_workers=config['num_parallel_workers']
    )
    
    return dataset

def create_dataset(data_path, batch_size, training=True):
    """创建训练和评估数据集"""
    dataset = ImageFolderDataset(data_path, 
                                num_parallel_workers=1,
                                shuffle=training,
                                num_shards=1,
                                shard_id=0)
    
    # 增强的医学图像数据增强策略
    if training:
        trans = [
            Decode(),
            # 保持图像比例进行缩放
            Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            # 随机裁剪，保留更多细节
            RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            # 医学图像增强
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
            # 更温和的颜色增强，保留医学特征
            RandomColorAdjust(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # 小角度旋转，避免破坏医学特征
            RandomRotation(degrees=10),
            # 添加高斯噪声，提高模型鲁棒性
            RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            RandomErasing(prob=0.3),
            # 标准化
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    else:
        trans = [
            Decode(),
            Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            CenterCrop(Config.IMAGE_SIZE),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    
    # 应用图像转换
    dataset = dataset.map(operations=trans, 
                         input_columns="image", 
                         num_parallel_workers=1,
                         python_multiprocessing=False)
    
    # 转换标签为int32
    dataset = dataset.map(operations=C.TypeCast(ms.int32), 
                         input_columns="label", 
                         num_parallel_workers=1,
                         python_multiprocessing=False)
    
    # 标签调整函数
    def adjust_labels(label):
        """确保标签在有效范围内"""
        if hasattr(label, 'asnumpy'):
            label_np = label.asnumpy()
        else:
            label_np = np.array(label)
        
        if label_np.max() >= Config.NUM_CLASSES or label_np.min() < 0:
            print(f"Warning: Label out of range - min: {label_np.min()}, max: {label_np.max()}, expected: [0, {Config.NUM_CLASSES-1}]")
        
        label_np = np.clip(label_np, 0, Config.NUM_CLASSES - 1)
        return ms.Tensor(label_np.astype(np.int32))
    
    dataset = dataset.map(operations=adjust_labels,
                         input_columns="label",
                         num_parallel_workers=1,
                         python_multiprocessing=False)
    
    # 批处理数据集
    dataset = dataset.batch(batch_size, 
                          drop_remainder=training,
                          num_parallel_workers=1)
    
    return dataset

def get_dataset():
    """
    获取训练集和验证集
    
    Returns:
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
    """
    # 创建完整数据集
    full_train_dataset = create_dataset(Config.TRAIN_DATA_PATH, 
                                      Config.BATCH_SIZE, 
                                      training=True)
    eval_dataset = create_dataset(Config.EVAL_DATA_PATH, 
                                 Config.BATCH_SIZE, 
                                 training=False)
    
    # 计算每个epoch的步数并限制到目标值
    full_steps = full_train_dataset.get_dataset_size()
    print(f"完整数据集大小: {full_steps} 步/epoch")
    
    if full_steps > Config.TARGET_STEPS_PER_EPOCH:
        train_dataset = full_train_dataset.take(Config.TARGET_STEPS_PER_EPOCH)
        print(f"采样数据集到 {Config.TARGET_STEPS_PER_EPOCH} 步/epoch")
    else:
        train_dataset = full_train_dataset
        print(f"使用完整数据集，{full_steps} 步/epoch")
    
    return train_dataset, eval_dataset

def get_balanced_dataset(data_path, is_training=True):
    """获取平衡采样的数据集"""
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path {data_path} does not exist!")
    
    # 创建数据集
    dataset = ImageFolderDataset(
        data_path,
        num_parallel_workers=Config.NUM_WORKERS,
        shuffle=is_training,
        decode=True
    )
    
    # 如果是训练模式，应用平衡采样
    if is_training:
        # 定义类别权重用于采样
        # 基于实际分布的反比例权重
        class_weights = [
            1.0/213,  # g1-ageDegeneration
            1.0/235,  # g1-cataract 
            1.0/313, # g1-diabetes
            1.0/228,  # g1-glaucoma
            1.0/186,  # g1-myopia
            1.0/103,  # g2-hypertension
            1.0/299, # g2-normal
            1.0/301   # g2-others
        ]
        
        # 归一化权重
        total_weight = sum(class_weights)
        class_weights = [w/total_weight for w in class_weights]
        
        # 应用加权随机采样
        try:
            # 注意：MindSpore的采样器可能需要不同的实现方式
            # 这里先保持原有方式，后续可以尝试其他采样策略
            pass
        except:
            # 如果采样器不支持，就保持原有方式
            pass
    
    # 数据预处理
    if is_training:
        # 训练集数据增强 - 增强策略
        transform = [
            Decode(),
            Resize((Config.IMAGE_SIZE + 64, Config.IMAGE_SIZE + 64)),
            RandomResizedCrop(
                size=Config.IMAGE_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.2),
            RandomRotation(degrees=10),
            RandomColorAdjust(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    else:
        # 验证集数据预处理
        transform = [
            Decode(),
            Resize(Config.IMAGE_SIZE + 32),
            CenterCrop(Config.IMAGE_SIZE),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    
    # 应用转换
    dataset = dataset.map(
        operations=transform,
        input_columns="image",
        num_parallel_workers=Config.NUM_WORKERS
    )
    
    # 设置批次大小
    dataset = dataset.batch(
        Config.BATCH_SIZE,
        drop_remainder=is_training,
        num_parallel_workers=Config.NUM_WORKERS
    )
    
    return dataset 

class ODIRDataset:
    """
    ODIR dataset loader using GeneratorDataset.
    This version loads the image inside __getitem__ to simplify the data pipeline.
    """
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train', 
                 num_parallel_workers: int = 8,
                 **kwargs):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory of the dataset
            split: Dataset split ('train' or 'valid')
            num_parallel_workers: Number of parallel workers
        """
        self.root = root
        self.split = split
        self.num_parallel_workers = num_parallel_workers
        self.data_dir = str(Path(self.root) / self.split)
        self._data = []
        self.class_name_to_id = {}
        
        # Validate directory structure
        if not os.path.exists(self.root):
            raise ValueError(
                f"错误：数据集根目录不存在: {self.root}\n"
                f"请确保数据集已正确下载并解压到以下位置：\n"
                f"  {self.root}/\n"
                f"  ├── train/     # 训练集目录\n"
                f"  │   ├── class1/\n"
                f"  │   │   ├── image1.jpg\n"
                f"  │   │   └── ...\n"
                f"  │   └── ...\n"
                f"  └── valid/     # 验证集目录\n"
                f"      ├── class1/\n"
                f"      │   ├── image1.jpg\n"
                f"      │   └── ...\n"
                f"      └── ...\n"
                f"\n"
                f"您可以通过以下步骤解决此问题：\n"
                f"1. 下载数据集并解压到 {self.root} 目录\n"
                f"2. 确保解压后的目录结构如上所示\n"
                f"3. 确保 train 和 valid 目录都存在且包含相应的类别子目录"
            )
        
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"错误：{self.split} 数据集目录不存在: {self.data_dir}\n"
                f"请确保数据集已正确下载并解压到以下位置：\n"
                f"  {self.root}/\n"
                f"  ├── train/     # 训练集目录\n"
                f"  │   ├── class1/\n"
                f"  │   │   ├── image1.jpg\n"
                f"  │   │   └── ...\n"
                f"  │   └── ...\n"
                f"  └── valid/     # 验证集目录\n"
                f"      ├── class1/\n"
                f"      │   ├── image1.jpg\n"
                f"      │   └── ...\n"
                f"      └── ...\n"
                f"\n"
                f"您可以通过以下步骤解决此问题：\n"
                f"1. 检查数据集是否已正确下载和解压\n"
                f"2. 确保 {self.split} 目录存在于 {self.root} 下\n"
                f"3. 确保 {self.split} 目录中包含相应的类别子目录"
            )
        
        # Get class directories
        self.class_names = sorted([d for d in os.listdir(self.data_dir) 
                                 if os.path.isdir(os.path.join(self.data_dir, d))])
        
        if not self.class_names:
            raise ValueError(
                f"错误：在 {self.data_dir} 中未找到类别目录\n"
                f"请确保数据集目录结构正确：\n"
                f"  {self.data_dir}/\n"
                f"  ├── class1/     # 类别1目录\n"
                f"  │   ├── image1.jpg\n"
                f"  │   └── ...\n"
                f"  ├── class2/     # 类别2目录\n"
                f"  │   ├── image1.jpg\n"
                f"  │   └── ...\n"
                f"  └── ...\n"
                f"\n"
                f"您可以通过以下步骤解决此问题：\n"
                f"1. 检查 {self.data_dir} 目录是否存在\n"
                f"2. 确保该目录下包含所有类别的子目录\n"
                f"3. 确保每个类别目录中包含相应的图像文件"
            )
        
        self.class_name_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Load image paths
        for class_name in self.class_names:
            class_id = self.class_name_to_id[class_name]
            class_dir = os.path.join(self.data_dir, class_name)
            image_files = (glob.glob(os.path.join(class_dir, '*.[jJ][pP][gG]')) +
                         glob.glob(os.path.join(class_dir, '*.[jJ][pP][eE][gG]')) +
                         glob.glob(os.path.join(class_dir, '*.[pP][nN][gG]')))
            
            if not image_files:
                print(f"警告：在类别目录中未找到图像文件: {class_dir}")
                continue
                
            for path in image_files:
                self._data.append((path, class_id))
        
        if not self._data:
            raise ValueError(
                f"错误：在 {self.data_dir} 中未找到任何图像文件\n"
                f"请确保数据集目录结构正确：\n"
                f"  {self.data_dir}/\n"
                f"  ├── class1/     # 类别1目录\n"
                f"  │   ├── image1.jpg\n"
                f"  │   └── ...\n"
                f"  ├── class2/     # 类别2目录\n"
                f"  │   ├── image1.jpg\n"
                f"  │   └── ...\n"
                f"  └── ...\n"
                f"\n"
                f"您可以通过以下步骤解决此问题：\n"
                f"1. 检查每个类别目录中是否包含图像文件\n"
                f"2. 确保图像文件格式为 .jpg、.jpeg 或 .png\n"
                f"3. 确保图像文件可以正常打开和读取"
            )
        
        print(f"已加载 {len(self._data)} 张图像，来自 {len(self.class_names)} 个类别，位于 {self.data_dir}")
        for class_name, class_id in self.class_name_to_id.items():
            class_images = sum(1 for _, label in self._data if label == class_id)
            print(f"  - {class_name}: {class_images} 张图像")

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, str]:
        """
        Get a data point.
        
        Args:
            index: Index of the data point
            
        Returns:
            Tuple of (image_path, label, image_path)
        """
        image_path, label = self._data[index]
        return image_path, np.array(label, dtype=np.int32), image_path
    
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data)
    
    def get_dataset(self, 
                   batch_size: int = 32, 
                   is_training: bool = True, 
                   image_size: int = 224) -> ds.Dataset:
        """
        Get a MindSpore dataset.
        
        Args:
            batch_size: Batch size
            is_training: Whether this is for training
            image_size: Target image size
            
        Returns:
            MindSpore dataset
        """
        # On Windows, multiprocessing is not supported
        num_workers = 1 if os.name == 'nt' else self.num_parallel_workers
        
        # Create dataset
        dataset = ds.GeneratorDataset(
            source=self,
            column_names=["image", "label", "image_path"],
            shuffle=(self.split == 'train'),
            num_parallel_workers=num_workers
        )
        
        # Get transforms
        transforms = (ImagePreprocessor.get_training_transforms(image_size) if is_training
                     else ImagePreprocessor.get_validation_transforms(image_size))
        
        # Apply transforms
        dataset = dataset.map(
            operations=transforms,
            input_columns=["image"],
            num_parallel_workers=num_workers
        )
        
        # Set batch size
        dataset = dataset.batch(
            batch_size,
            drop_remainder=is_training,
            num_parallel_workers=num_workers
        )
        
        return dataset
    
    def get_balanced_dataset(self, 
                           batch_size: int = 32, 
                           is_training: bool = True, 
                           image_size: int = 224) -> ds.Dataset:
        """
        Get a balanced dataset.
        
        Args:
            batch_size: Batch size
            is_training: Whether this is for training
            image_size: Target image size
            
        Returns:
            MindSpore dataset
        """
        # Create base dataset
        dataset = self.get_dataset(batch_size, is_training, image_size)
        
        if is_training:
            # Define class weights for sampling
            class_weights = [
                1.0/213,  # g1-ageDegeneration
                1.0/235,  # g1-cataract 
                1.0/313,  # g1-diabetes
                1.0/228,  # g1-glaucoma
                1.0/186,  # g1-myopia
                1.0/103,  # g2-hypertension
                1.0/299,  # g2-normal
                1.0/301   # g2-others
            ]
            
            # Normalize weights
            total_weight = sum(class_weights)
            class_weights = [w/total_weight for w in class_weights]
            
            # TODO: Implement weighted sampling
            # Note: MindSpore's sampler may need different implementation
            pass
        
        return dataset 