"""
Dataset module for Eye Disease Classification
"""

import mindspore as ms
from mindspore.dataset import ImageFolderDataset
from mindspore.dataset.transforms import transforms as T
from mindspore.dataset.vision import transforms as V
from src.config.config import Config
import numpy as np

def create_dataset(data_path, batch_size, training=True):
    """创建训练和评估数据集"""
    dataset = ImageFolderDataset(data_path, 
                                num_parallel_workers=1,  # 使用1以避免兼容性问题
                                shuffle=training,
                                num_shards=1,
                                shard_id=0)
    
    # 定义眼底图像的数据增强
    if training:
        trans = [
            V.Decode(),
            V.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            # 医学图像增强 - 保留诊断特征
            V.RandomHorizontalFlip(prob=0.5),  # 眼睛图像可以水平翻转
            V.RandomVerticalFlip(prob=0.5),    # 眼睛图像也可以垂直翻转
            V.RandomColorAdjust(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # 保留颜色信息
            V.RandomRotation(degrees=15),  # 医学图像的小角度旋转
            # 为真实世界的医学图像添加一些噪声容差
            V.Normalize(mean=Config.MEAN, std=Config.STD),
            V.HWC2CHW()
        ]
    else:
        trans = [
            V.Decode(),
            V.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            V.Normalize(mean=Config.MEAN, std=Config.STD),
            V.HWC2CHW()
        ]
    
    # 应用图像转换
    dataset = dataset.map(operations=trans, 
                         input_columns="image", 
                         num_parallel_workers=1,
                         python_multiprocessing=False)
    
    # 转换标签为int32
    dataset = dataset.map(operations=T.TypeCast(ms.int32), 
                         input_columns="label", 
                         num_parallel_workers=1,
                         python_multiprocessing=False)
    
    # 标签调整函数
    def adjust_labels(label):
        """确保标签在有效范围内"""
        # 转换为numpy进行处理
        if hasattr(label, 'asnumpy'):
            label_np = label.asnumpy()
        else:
            label_np = np.array(label)
        
        # 调试：检查标签范围
        if label_np.max() >= Config.NUM_CLASSES or label_np.min() < 0:
            print(f"Warning: Label out of range - min: {label_np.min()}, max: {label_np.max()}, expected: [0, {Config.NUM_CLASSES-1}]")
        
        # 裁剪到有效范围
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
        # 采样数据集以达到目标步数
        train_dataset = full_train_dataset.take(Config.TARGET_STEPS_PER_EPOCH)
        print(f"采样数据集到 {Config.TARGET_STEPS_PER_EPOCH} 步/epoch")
    else:
        train_dataset = full_train_dataset
        print(f"使用完整数据集，{full_steps} 步/epoch")
    
    return train_dataset, eval_dataset 