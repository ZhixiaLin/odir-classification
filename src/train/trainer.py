"""
Training module for Eye Disease Classification
"""

import os
import json
import time
import logging
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore import context
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.metrics import Accuracy
from mindspore.nn.learning_rate_schedule import CosineDecayLR
from datetime import datetime
from mindspore.dataset import ImageFolderDataset
from mindspore.dataset.transforms import c_transforms as C
from mindspore.dataset.vision import RandomResizedCrop, RandomHorizontalFlip, RandomColorAdjust, Resize, CenterCrop, Normalize, HWC2CHW, Decode, RandomVerticalFlip, RandomRotation

from src.config.config import Config
from src.models.resnet import get_model

class EarlyStopping(Callback):
    """早停回调"""
    def __init__(self, patience=10, min_delta=0.001):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, run_context):
        """每个epoch结束时调用"""
        cb_params = run_context.original_args()
        current_loss = cb_params.get("net_outputs", float('inf'))
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            self.best_epoch = cb_params.cur_epoch_num
            self.best_weights = cb_params.train_network.parameters_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = cb_params.cur_epoch_num
                run_context.request_stop()
                print(f"\nEarly stopping triggered at epoch {self.stopped_epoch}")
                print(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")

class ValidationMonitor(Callback):
    """验证监控器"""
    def __init__(self, network, eval_dataset, patience=10, min_delta=0.001):
        super(ValidationMonitor, self).__init__()
        self.network = network
        self.eval_dataset = eval_dataset
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0.0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

    def on_train_epoch_end(self, run_context):
        """每个训练epoch结束时的回调"""
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        
        # 设置网络为评估模式
        self.network.set_train(False)
        
        # 计算验证集准确率
        correct = 0
        total = 0
        for data in self.eval_dataset.create_dict_iterator():
            images = data['image']
            labels = data['label']
            outputs = self.network(images)
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().asnumpy()
            total += labels.shape[0]
        
        accuracy = correct / total
        
        # 检查是否需要早停
        if accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.wait = 0
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                run_context.request_stop()
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best accuracy: {self.best_accuracy:.2%} at epoch {self.best_epoch}")
        
        print(f"\nValidation accuracy: {accuracy:.2%}")
        
        # 恢复训练模式
        self.network.set_train(True)

class TrainingLogger(Callback):
    """训练日志记录器"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{self.timestamp}.json")
        self.metrics_file = os.path.join(log_dir, f"metrics_{self.timestamp}.json")
        
        # 初始化日志数据
        self.log_data = {
            "timestamp": self.timestamp,
            "training_start": datetime.now().isoformat(),
            "epochs": [],
            "best_accuracy": 0.0,
            "best_epoch": 0,
            "training_end": None,
            "total_time": None,
            "final_metrics": None
        }
        
        # 初始化指标数据
        self.metrics_data = {
            "timestamp": self.timestamp,
            "epochs": []
        }
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 保存初始配置
        self.config_path = Config.save_config(log_dir)
        self.log_data["config_path"] = self.config_path

    def on_train_epoch_begin(self, run_context):
        """每个训练epoch开始时的回调"""
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """每个训练epoch结束时的回调"""
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        epoch_time = time.time() - self.epoch_start_time
        
        # 记录epoch信息
        epoch_info = {
            "epoch": epoch,
            "time": epoch_time
        }
        
        # 尝试获取损失值
        if hasattr(cb_params, 'net_outputs'):
            loss = cb_params.net_outputs
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            epoch_info["loss"] = float(loss.asnumpy())
        
        self.log_data["epochs"].append(epoch_info)
        self.metrics_data["epochs"].append(epoch_info)
        
        # 保存日志
        self._save_logs()

    def on_train_end(self, run_context):
        """训练结束时的回调"""
        cb_params = run_context.original_args()
        
        # 尝试获取最终指标
        try:
            if hasattr(cb_params, 'metrics'):
                self.log_data["final_metrics"] = cb_params.metrics
        except Exception as e:
            print(f"Warning: Could not get final metrics: {str(e)}")
        
        self.log_data["training_end"] = datetime.now().isoformat()
        self.log_data["total_time"] = time.time() - self.epoch_start_time
        self._save_logs()

    def _save_logs(self):
        """保存日志到文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=4, ensure_ascii=False)
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_data, f, indent=4, ensure_ascii=False)

def validate_image(image_path):
    """验证图像文件是否有效"""
    try:
        from PIL import Image
        import numpy as np
        
        # 尝试打开图像
        with Image.open(image_path) as img:
            # 检查图像模式
            if img.mode not in ['RGB', 'RGBA', 'L']:
                return False
            
            # 检查图像大小
            if img.size[0] < 32 or img.size[1] < 32:
                return False
            
            # 尝试转换为numpy数组
            img_array = np.array(img)
            
            # 检查数组形状
            if len(img_array.shape) < 2:
                return False
            
            # 检查是否有有效的像素值
            if img_array.size == 0:
                return False
            
            return True
            
    except Exception as e:
        print(f"Image validation error for {image_path}: {e}")
        return False

def clean_dataset(dataset_path):
    """清理数据集中的无效图像"""
    import os
    import shutil
    
    invalid_files = []
    backup_dir = os.path.join(os.path.dirname(dataset_path), "invalid_images")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                if not validate_image(file_path):
                    invalid_files.append(file_path)
    
    if invalid_files:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Found {len(invalid_files)} invalid image files:")
        for file_path in invalid_files:
            print(f"  - {file_path}")
            # 移动无效文件到备份目录
            rel_path = os.path.relpath(file_path, dataset_path)
            backup_path = os.path.join(backup_dir, rel_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            try:
                shutil.move(file_path, backup_path)
                print(f"    Moved to: {backup_path}")
            except Exception as e:
                print(f"    Failed to move: {e}")
    
    return len(invalid_files)

def get_dataset(dataset_path, is_training=True):
    """获取数据集"""
    # 首先清理无效图像
    print(f"Validating images in {dataset_path}...")
    invalid_count = clean_dataset(dataset_path)
    if invalid_count > 0:
        print(f"Removed {invalid_count} invalid images")
    
    # 数据增强 - 增强的数据变换以提升模型鲁棒性
    if is_training:
        transform = [
            # 先调整大小，确保图像有正确的维度
            Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            # 然后进行随机裁剪
            RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            # 随机翻转
            RandomHorizontalFlip(prob=0.5),
            # 小角度随机旋转（眼部图像的小旋转是合理的）
            RandomRotation(degrees=10),
            # 颜色增强
            RandomColorAdjust(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # 归一化
            Normalize(Config.MEAN, Config.STD),
            HWC2CHW()
        ]
    else:
        transform = [
            # 验证时只进行必要的变换
            Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            Normalize(Config.MEAN, Config.STD),
            HWC2CHW()
        ]
    
    # 创建数据集
    dataset = ImageFolderDataset(
        dataset_path,
        num_parallel_workers=1,  # 减少并行工作进程数以便调试
        shuffle=is_training,
        decode=True  # 确保自动解码
    )
    
    # 应用转换
    dataset = dataset.map(
        operations=transform,
        input_columns="image",
        num_parallel_workers=1,  # 减少并行工作进程数以便调试
        python_multiprocessing=False  # 禁用多进程以避免问题
    )
    
    # 设置批次大小
    dataset = dataset.batch(Config.BATCH_SIZE, drop_remainder=is_training)
    
    return dataset

def get_lr_scheduler(epochs, steps_per_epoch):
    """获取学习率调度器"""
    # 使用更简单的余弦退火学习率
    decay_steps = epochs * steps_per_epoch
    
    lr_scheduler = CosineDecayLR(
        min_lr=Config.MIN_LR,
        max_lr=Config.MAX_LR,
        decay_steps=decay_steps
    )
    
    return lr_scheduler

def train():
    """训练模型"""
    # 设置运行环境
    ms.set_context(device_target=Config.DEVICE_TARGET)
    context.set_context(mode=context.PYNATIVE_MODE)
    
    # 创建日志目录
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(Config.LOG_DIR, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 记录训练配置
    logging.info("Training Configuration:")
    logging.info(json.dumps(Config.to_dict(), indent=2))
    
    # 保存配置到文件
    config_path = os.path.join(Config.LOG_DIR, f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    Config.save_config(config_path)
    
    # 获取数据集
    train_dataset = get_dataset(Config.TRAIN_DATA_PATH, is_training=True)
    valid_dataset = get_dataset(Config.VALID_DATA_PATH, is_training=False)
    
    # 获取模型
    network = get_model()
    
    # 获取学习率调度器
    steps_per_epoch = train_dataset.get_dataset_size()
    lr_scheduler = get_lr_scheduler(Config.EPOCHS, steps_per_epoch)
    
    # 定义优化器 - 使用AdamWeightDecay获得更好的性能
    optimizer = nn.AdamWeightDecay(
        network.trainable_params(),
        learning_rate=lr_scheduler,
        weight_decay=Config.WEIGHT_DECAY,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    )
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 定义训练步骤
    def forward_fn(data, label):
        logits = network(data)
        loss = loss_fn(logits, label)
        return loss, logits
    
    # 定义梯度函数
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    
    # 定义训练步骤
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss
    
    # 训练循环
    best_acc = 0.0
    no_improve_epochs = 0
    start_time = time.time()
    
    for epoch in range(Config.EPOCHS):
        # 训练阶段
        network.set_train()
        total_loss = 0
        step_time = time.time()
        
        for step, (data, label) in enumerate(train_dataset.create_tuple_iterator()):
            loss = train_step(data, label)
            total_loss += loss.asnumpy()
            
            if (step + 1) % 10 == 0:
                step_time = time.time() - step_time
                logging.info(f"epoch: {epoch + 1} step: {step + 1}, loss is {loss.asnumpy()}")
                logging.info(f"step time: {step_time * 1000:.2f} ms")
                step_time = time.time()
        
        # 验证阶段
        network.set_train(False)
        correct = 0
        total = 0
        
        for data, label in valid_dataset.create_tuple_iterator():
            logits = network(data)
            pred = ms.ops.Argmax(axis=1)(logits)
            correct += (pred == label).sum().asnumpy()
            total += label.shape[0]
        
        accuracy = correct / total
        logging.info(f"Validation accuracy: {accuracy:.2%}")
        
        # 保存最佳模型和每个epoch的模型
        if accuracy > best_acc:
            best_acc = accuracy
            no_improve_epochs = 0
            # 保存最佳模型
            best_ckpt_file = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_PREFIX}_best.ckpt")
            ms.save_checkpoint(network, best_ckpt_file)
            logging.info(f"Saved best model with accuracy: {accuracy:.2%}")
        else:
            no_improve_epochs += 1
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            epoch_ckpt_file = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_PREFIX}_epoch_{epoch+1}.ckpt")
            ms.save_checkpoint(network, epoch_ckpt_file)
            logging.info(f"Saved epoch {epoch+1} model")
        
        # 早停检查
        if no_improve_epochs >= Config.EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # 记录训练结果
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time/3600:.2f} hours")
    logging.info(f"Best validation accuracy: {best_acc:.2%}")
    
    # 保存训练配置和结果
    results = {
        "best_accuracy": float(best_acc),
        "training_time": float(training_time),
        "epochs_trained": epoch + 1,
        "final_epoch": epoch + 1,
        "early_stopped": no_improve_epochs >= Config.EARLY_STOPPING_PATIENCE
    }
    
    with open(os.path.join(Config.LOG_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    train() 