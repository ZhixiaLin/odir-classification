"""
Configuration module for Eye Disease Classification
"""

import os
import json
from datetime import datetime

class Config:
    """配置类"""
    # 基本设置
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 设备配置
    DEVICE_TARGET = "CPU"  # 或 "GPU"
    
    # 数据配置
    DATA_ROOT = os.path.join(ROOT_DIR, 'data', 'odir4')
    TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'train')
    VALID_DATA_PATH = os.path.join(DATA_ROOT, 'valid')
    EVAL_DATA_PATH = os.path.join(DATA_ROOT, 'valid')
    IMAGE_SIZE = 224
    BATCH_SIZE = 16  # 增加批次大小以提高梯度稳定性
    NUM_WORKERS = 2  # 减少工作进程数
    
    # 模型配置
    MODEL_PREFIX = "eye_disease"
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')
    KEEP_CHECKPOINT_MAX = 5
    NUM_CLASSES = 8
    
    # 训练配置 - 10小时内完成
    EPOCHS = 18  # 18轮 × 34分钟 = 10.2小时
    LEARNING_RATE = 0.0008  # 稍微提高学习率补偿轮数减少
    WEIGHT_DECAY = 5e-5  # 进一步减少正则化
    MOMENTUM = 0.9
    EARLY_STOPPING_PATIENCE = 15  # 减少耐心，避免浪费时间
    MIN_DELTA = 0.001  # 敏感的改善检测
    
    # 训练目标设置
    TARGET_STEPS_PER_EPOCH = 100  # 适中的步数
    FAST_TRAINING = False  # 快速训练模式标志
    
    # 类别名称
    CLASS_NAMES = [
        "g1-ageDegeneration",
        "g1-cataract",
        "g1-diabetes",
        "g1-glaucoma",
        "g1-myopia",
        "g2-hypertension",
        "g2-normal",
        "g2-others"
    ]
    
    # 图像预处理参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # 学习率调度器配置
    MIN_LR = 1e-7  # 降低最小学习率
    MAX_LR = 0.0001  # 降低最大学习率

    @classmethod
    def to_dict(cls):
        """将配置转换为字典"""
        return {
            "device_target": cls.DEVICE_TARGET,
            "image_size": cls.IMAGE_SIZE,
            "batch_size": cls.BATCH_SIZE,
            "num_workers": cls.NUM_WORKERS,
            "epochs": cls.EPOCHS,
            "learning_rate": cls.LEARNING_RATE,
            "weight_decay": cls.WEIGHT_DECAY,
            "momentum": cls.MOMENTUM,
            "early_stopping_patience": cls.EARLY_STOPPING_PATIENCE,
            "min_delta": cls.MIN_DELTA,
            "num_classes": cls.NUM_CLASSES,
            "class_names": cls.CLASS_NAMES,
            "mean": cls.MEAN,
            "std": cls.STD,
            "min_lr": cls.MIN_LR,
            "max_lr": cls.MAX_LR
        }

    @classmethod
    def save_config(cls, save_path):
        """保存配置到文件"""
        config_dict = cls.to_dict()
        config_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

# 导出Config类
__all__ = ['Config'] 