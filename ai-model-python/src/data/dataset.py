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

from src.config.config import Config
import numpy as np

def get_dataset(data_path, is_training=True):
    """获取数据集"""
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path {data_path} does not exist!")
    
    # 创建数据集
    dataset = ImageFolderDataset(
        data_path,
        num_parallel_workers=Config.NUM_WORKERS,
        shuffle=is_training,
        decode=True
    )
    
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
            # 添加高斯模糊增强鲁棒性
            # RandomAffine(degrees=5, translate=(0.05, 0.05)),
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