"""
Training module for Eye Disease Classification - FIXED VERSION
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
import numpy as np

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

class FocalLoss(nn.Cell):
    """Focal Loss用于处理极端类别不平衡
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    """
    def __init__(self, num_classes, alpha=None, gamma=2.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.class_weights = class_weights
        
        # 设置alpha权重
        if alpha is None:
            self.alpha = ms.Tensor([1.0] * num_classes, dtype=ms.float32)
        else:
            self.alpha = ms.Tensor(alpha, dtype=ms.float32)
        
        # MindSpore操作
        self.softmax = ms.ops.Softmax(axis=1)
        self.log_softmax = ms.ops.LogSoftmax(axis=1)
        self.onehot = ms.ops.OneHot()
        self.gather = ms.ops.GatherD()
        self.pow = ms.ops.Pow()
        self.expand_dims = ms.ops.ExpandDims()
    
    def construct(self, logits, targets):
        # 计算概率
        log_probs = self.log_softmax(logits)
        probs = self.softmax(logits)
        
        # 创建one-hot编码
        targets_one_hot = self.onehot(targets, self.num_classes, 
                                     ms.Tensor(1.0, ms.float32), 
                                     ms.Tensor(0.0, ms.float32))
        
        # 获取目标类别的概率
        p_t = (probs * targets_one_hot).sum(axis=1)
        
        # 获取目标类别的log概率  
        log_p_t = (log_probs * targets_one_hot).sum(axis=1)
        
        # 获取alpha权重
        alpha_t = self.gather(self.alpha, 0, targets.view(-1))
        
        # 计算focal权重: (1-p_t)^gamma
        focal_weight = self.pow((1.0 - p_t), self.gamma)
        
        # 计算focal loss
        focal_loss = -alpha_t * focal_weight * log_p_t
        
        # 应用类别权重
        if self.class_weights is not None:
            class_weight_t = self.gather(self.class_weights, 0, targets.view(-1))
            focal_loss = focal_loss * class_weight_t
        
        return focal_loss.mean()

class AdvancedLoss(nn.Cell):
    """结合多种技术的高级损失函数"""
    def __init__(self, num_classes, class_weights, focal_gamma=2.0, label_smoothing=0.1):
        super(AdvancedLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        
        # Focal Loss
        self.focal_loss = FocalLoss(
            num_classes=num_classes,
            gamma=focal_gamma,
            class_weights=class_weights
        )
        
        # 标准交叉熵
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def construct(self, logits, targets):
        # 主要使用Focal Loss
        focal_loss = self.focal_loss(logits, targets)
        
        # 添加少量标准交叉熵作为正则化
        ce_loss = self.ce_loss(logits, targets)
        if self.class_weights is not None:
            weight_mask = ms.ops.GatherD()(self.class_weights, 0, targets)
            ce_loss = ce_loss * weight_mask
        ce_loss = ce_loss.mean()
        
        # Label smoothing
        if self.label_smoothing > 0:
            log_probs = ms.ops.LogSoftmax(axis=1)(logits)
            smooth_loss = -log_probs.mean(axis=1)
            if self.class_weights is not None:
                weight_mask = ms.ops.GatherD()(self.class_weights, 0, targets)
                smooth_loss = smooth_loss * weight_mask
            smooth_loss = smooth_loss.mean()
            
            # 组合损失
            total_loss = 0.7 * focal_loss + 0.2 * ce_loss + 0.1 * smooth_loss
        else:
            total_loss = 0.8 * focal_loss + 0.2 * ce_loss
        
        return total_loss

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

def get_fast_dataset(dataset_path, is_training=True):
    """获取快速训练数据集 - 减少数据增强以提高速度"""
    # 首先清理无效图像
    print(f"Validating images in {dataset_path}...")
    invalid_count = clean_dataset(dataset_path)
    if invalid_count > 0:
        print(f"Removed {invalid_count} invalid images")
    
    # 增强的数据增强 - 帮助稀少类别学习
    if is_training:
        transform = [
            # 更强的数据变换以增加样本多样性
            Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.3),  # 增加垂直翻转
            RandomRotation(degrees=15),     # 增加旋转角度
            # 增强的颜色变换
            RandomColorAdjust(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
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

def get_dataset(dataset_path, is_training=True):
    """获取数据集"""
    # 首先清理无效图像
    print(f"Validating images in {dataset_path}...")
    invalid_count = clean_dataset(dataset_path)
    if invalid_count > 0:
        print(f"Removed {invalid_count} invalid images")
    
    # 数据增强 - 针对稀少类别的强化增强策略
    if is_training:
        transform = [
            # 更强的数据变换以增加样本多样性
            Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            # 增加翻转概率
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.3),  # 增加垂直翻转
            # 增加旋转角度帮助稀少类别
            RandomRotation(degrees=15),
            # 增强的颜色变换
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

def calculate_class_weights(dataset_path):
    """基于样本数量计算类别权重 - 强化小类别"""
    
    # 用户调整后的类别分布
    actual_counts = {
        "g1-ageDegeneration": 213,
        "g1-cataract": 235,
        "g1-diabetes": 313,
        "g1-glaucoma": 228,
        "g1-myopia": 186,
        "g2-hypertension": 103,
        "g2-normal": 299,
        "g2-others": 301
    }
    
    total_samples = sum(actual_counts.values())
    
    # 更激进的权重策略 - 平方根倒数，强化稀少类别
    weights = {}
    
    for class_name, count in actual_counts.items():
        # 使用平方根倒数，给稀少类别更大权重
        weight = (total_samples / count) ** 0.7  # 指数小于1，减缓权重差异
        weights[class_name] = weight
    
    # 温和的权重调整策略
    # 适度提升困难类别
    problem_classes = ["g2-hypertension"]
    for class_name in problem_classes:
        weights[class_name] *= 2.0  # 适度提升
    
    # 恢复所有类别的基础学习能力
    # 不过度压制任何类别
    
    # 转换为数组
    weight_list = [weights[class_name] for class_name in Config.CLASS_NAMES]
    weight_array = np.array(weight_list, dtype=np.float32)
    
    # 标准化权重，避免梯度爆炸
    weight_array = weight_array / weight_array.mean() * 2.0
    
    # 日志输出
    logging.info("=== 强化权重策略 ===")
    logging.info("为稀少类别提供更强的权重支持")
    
    max_ratio = weight_array.max() / weight_array.min()
    logging.info(f"权重比例范围: 1:{max_ratio:.1f}")
    
    for i, (class_name, weight) in enumerate(zip(Config.CLASS_NAMES, weight_array)):
        count = actual_counts[class_name]
        percentage = count / total_samples * 100
        logging.info(f"{class_name}: 样本数={count} ({percentage:.1f}%), 权重={weight:.2f}")
    
    return weight_array

def get_balanced_training_strategy(train_dataset, class_weights):
    """获取平衡训练策略 - 简化版本"""
    
    logging.info("=== 训练策略 ===")
    logging.info("数据已手动平衡，采用温和权重 + Focal Loss策略")
    logging.info("预期效果：所有类别都能有效学习")
    
    return None  # 简化返回，不需要复杂的重复策略

def apply_progressive_learning(epoch, total_epochs, loss_fn):
    """简化的学习策略 - 数据平衡后"""
    progress = epoch / total_epochs
    
    if progress < 0.5:
        # 前半段：稳定学习
        focus_mode = "stable_learning"
        if epoch % 5 == 0:  # 减少日志频率
            logging.info(f"Epoch {epoch}: 稳定学习阶段")
    else:
        # 后半段：精细调优
        focus_mode = "fine_tuning"
        if epoch % 5 == 0:  # 减少日志频率
            logging.info(f"Epoch {epoch}: 精细调优阶段")
    
    return focus_mode

def analyze_dataset_distribution(dataset_path):
    """深度分析数据集分布，识别潜在问题"""
    import os
    from collections import defaultdict
    from PIL import Image
    import numpy as np
    
    class_stats = defaultdict(lambda: {
        'count': 0,
        'avg_size': [],
        'avg_brightness': [],
        'image_modes': [],
        'file_sizes': []
    })
    
    print("\n=== 数据集深度分析 ===")
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        stats = class_stats[class_name]
        valid_images = 0
        
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_path, img_file)
            try:
                # 获取文件大小
                file_size = os.path.getsize(img_path)
                stats['file_sizes'].append(file_size)
                
                # 分析图像属性
                with Image.open(img_path) as img:
                    # 图像尺寸
                    stats['avg_size'].append(img.size)
                    
                    # 图像模式
                    stats['image_modes'].append(img.mode)
                    
                    # 转换为RGB分析亮度
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 计算平均亮度
                    img_array = np.array(img)
                    brightness = np.mean(img_array)
                    stats['avg_brightness'].append(brightness)
                    
                    valid_images += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        stats['count'] = valid_images
    
    # 输出统计结果
    total_images = sum(stats['count'] for stats in class_stats.values())
    
    print(f"总图像数: {total_images}")
    print(f"类别数: {len(class_stats)}")
    print("\n各类别详细统计:")
    
    problematic_classes = []
    
    for class_name, stats in class_stats.items():
        if stats['count'] == 0:
            continue
            
        # 计算统计信息
        avg_brightness = np.mean(stats['avg_brightness'])
        brightness_std = np.std(stats['avg_brightness'])
        
        avg_file_size = np.mean(stats['file_sizes'])
        file_size_std = np.std(stats['file_sizes'])
        
        unique_modes = set(stats['image_modes'])
        avg_width = np.mean([size[0] for size in stats['avg_size']])
        avg_height = np.mean([size[1] for size in stats['avg_size']])
        
        percentage = stats['count'] / total_images * 100
        
        # 识别潜在问题
        issues = []
        if brightness_std > 50:  # 亮度变化大
            issues.append("亮度差异大")
        if len(unique_modes) > 1:  # 多种图像模式
            issues.append("图像模式不统一")
        if file_size_std > avg_file_size:  # 文件大小差异大
            issues.append("文件大小差异大")
        if avg_brightness < 50:  # 图像偏暗
            issues.append("图像偏暗")
        if avg_brightness > 200:  # 图像偏亮
            issues.append("图像偏亮")
            
        if issues:
            problematic_classes.append((class_name, issues))
        
        print(f"\n{class_name}:")
        print(f"  数量: {stats['count']} ({percentage:.1f}%)")
        print(f"  平均亮度: {avg_brightness:.1f} ± {brightness_std:.1f}")
        print(f"  平均尺寸: {avg_width:.0f}x{avg_height:.0f}")
        print(f"  图像模式: {unique_modes}")
        print(f"  平均文件大小: {avg_file_size/1024:.1f}KB ± {file_size_std/1024:.1f}KB")
        if issues:
            print(f"  ⚠️  潜在问题: {', '.join(issues)}")
    
    if problematic_classes:
        print(f"\n🚨 发现 {len(problematic_classes)} 个问题类别:")
        for class_name, issues in problematic_classes:
            print(f"  {class_name}: {', '.join(issues)}")
    
    return class_stats, problematic_classes

def create_improved_loss_function(class_weights, num_classes):
    """创建改进的损失函数"""
    
    class AdaptiveFocalLoss(nn.Cell):
        """自适应Focal Loss - 根据训练进度调整参数"""
        def __init__(self, num_classes, class_weights, alpha=1.0, gamma=2.0):
            super(AdaptiveFocalLoss, self).__init__()
            self.num_classes = num_classes
            self.alpha = alpha
            self.gamma = gamma
            self.class_weights = ms.Tensor(class_weights, dtype=ms.float32)
            
            # MindSpore操作
            self.softmax = ms.ops.Softmax(axis=1)
            self.log_softmax = ms.ops.LogSoftmax(axis=1)
            self.onehot = ms.ops.OneHot()
            self.gather = ms.ops.GatherD()
            self.pow = ms.ops.Pow()
            
        def construct(self, logits, targets):
            # 计算log概率
            log_probs = self.log_softmax(logits)
            probs = self.softmax(logits)
            
            # 创建one-hot编码
            targets_one_hot = self.onehot(targets, self.num_classes, 
                                         ms.Tensor(1.0, ms.float32), 
                                         ms.Tensor(0.0, ms.float32))
            
            # 获取目标类别的概率和log概率
            p_t = (probs * targets_one_hot).sum(axis=1)
            log_p_t = (log_probs * targets_one_hot).sum(axis=1)
            
            # 计算focal权重 - 更温和的参数
            focal_weight = self.pow((1.0 - p_t), self.gamma)
            
            # 应用类别权重
            class_weight_t = self.gather(self.class_weights, 0, targets.view(-1))
            
            # 计算focal loss
            focal_loss = -self.alpha * class_weight_t * focal_weight * log_p_t
            
            # 添加标准交叉熵作为稳定项
            ce_loss = -log_p_t
            
            # 组合损失：主要focal loss + 少量标准CE
            total_loss = 0.8 * focal_loss + 0.2 * ce_loss * class_weight_t
            
            return total_loss.mean()
    
    return AdaptiveFocalLoss(num_classes, class_weights, alpha=0.75, gamma=1.5)

def create_enhanced_model():
    """创建增强的模型架构"""
    
    class MedicalResNet(nn.Cell):
        """医学图像专用ResNet"""
        def __init__(self, num_classes=8):
            super(MedicalResNet, self).__init__()
            
            # 更适合医学图像的初始层
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
            
            # 构建ResNet层
            self.in_channels = 64
            self.layer1 = self._make_layer(64, 2, stride=1)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)
            
            # 全局平均池化
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 更强的分类器
            self.classifier = nn.SequentialCell([
                nn.Dense(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                
                nn.Dense(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                
                nn.Dense(256, num_classes)
            ])
            
        def _make_layer(self, out_channels, blocks, stride=1):
            layers = []
            
            # 第一个block可能需要下采样
            layers.append(self._make_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            
            # 其余blocks
            for _ in range(1, blocks):
                layers.append(self._make_block(out_channels, out_channels, 1))
                
            return nn.SequentialCell(layers)
        
        def _make_block(self, in_channels, out_channels, stride):
            """创建ResNet基本块"""
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.SequentialCell([
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                ])
            
            return BasicResBlock(in_channels, out_channels, stride, downsample)
        
        def construct(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = P.Flatten()(x)
            x = self.classifier(x)
            
            return x
    
    class BasicResBlock(nn.Cell):
        """基本ResNet块"""
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(BasicResBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, pad_mode='pad')
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, pad_mode='pad')
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample
            
        def construct(self, x):
            identity = x
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            if self.downsample is not None:
                identity = self.downsample(x)
                
            out = out + identity
            out = self.relu(out)
            
            return out
    
    return MedicalResNet()

def improved_training_strategy():
    """改进的训练策略"""
    
    class WarmupCosineScheduler:
        """预热+余弦退火学习率调度器"""
        def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, steps_per_epoch):
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.steps_per_epoch = steps_per_epoch
            self.warmup_steps = warmup_epochs * steps_per_epoch
            self.total_steps = total_epochs * steps_per_epoch
            
        def get_lr(self, step):
            if step < self.warmup_steps:
                # 预热阶段：线性增加
                return self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps
            else:
                # 余弦退火阶段
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return WarmupCosineScheduler(
        max_lr=0.001,  # 稍微提高最大学习率
        min_lr=1e-6,
        warmup_epochs=3,  # 3个epoch预热
        total_epochs=Config.EPOCHS,
        steps_per_epoch=100
    )

def train():
    """训练模型 - 改进版本"""
    # 设置运行环境
    ms.set_device(Config.DEVICE_TARGET)
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
    
    # 深度分析数据集
    logging.info("=== 开始数据集深度分析 ===")
    class_stats, problematic_classes = analyze_dataset_distribution(Config.TRAIN_DATA_PATH)
    
    if problematic_classes:
        logging.warning(f"发现 {len(problematic_classes)} 个问题类别，这可能影响训练效果")
        for class_name, issues in problematic_classes:
            logging.warning(f"  {class_name}: {', '.join(issues)}")
    
    # 计算类别权重
    class_weights = calculate_class_weights(Config.TRAIN_DATA_PATH)
    logging.info("Class weights: %s", class_weights)
    
    # 获取平衡训练策略
    repeat_factors = get_balanced_training_strategy(None, class_weights)
    
    # 记录训练配置
    logging.info("Training Configuration:")
    logging.info(json.dumps(Config.to_dict(), indent=2))
    
    # 保存配置到文件
    config_path = os.path.join(Config.LOG_DIR, f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    Config.save_config(config_path)
    
    # 获取数据集 - 根据模式选择
    if getattr(Config, 'FAST_TRAINING', False):
        logging.info("Using fast dataset mode (simplified augmentation)")
        train_dataset = get_fast_dataset(Config.TRAIN_DATA_PATH, is_training=True)
        valid_dataset = get_fast_dataset(Config.VALID_DATA_PATH, is_training=False)
    else:
        logging.info("Using full dataset mode (enhanced augmentation)")
        train_dataset = get_dataset(Config.TRAIN_DATA_PATH, is_training=True)
        valid_dataset = get_dataset(Config.VALID_DATA_PATH, is_training=False)
    
    # 快速训练模式：限制每个epoch的步数
    full_steps = train_dataset.get_dataset_size()
    if full_steps > Config.TARGET_STEPS_PER_EPOCH:
        logging.info(f"Fast training mode: reducing steps from {full_steps} to {Config.TARGET_STEPS_PER_EPOCH} per epoch")
        train_dataset = train_dataset.take(Config.TARGET_STEPS_PER_EPOCH)
    else:
        logging.info(f"Using full dataset: {full_steps} steps per epoch")
    
    # 获取改进的模型
    try:
        network = create_enhanced_model()
        logging.info("使用医学图像专用ResNet模型")
    except Exception as e:
        logging.warning(f"创建增强模型失败，使用默认模型: {e}")
        network = get_model()
    
    # 获取改进的学习率调度器
    steps_per_epoch = train_dataset.get_dataset_size()
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    
    lr_scheduler = improved_training_strategy()
    
    # 动态学习率
    def get_dynamic_lr(step):
        return lr_scheduler.get_lr(step)
    
    # 创建学习率序列
    lr_values = []
    for step in range(Config.EPOCHS * steps_per_epoch):
        lr_values.append(get_dynamic_lr(step))
    
    # 定义优化器 - 改进参数
    # 使用更稳定的SGD优化器
    optimizer = nn.SGD(
        network.trainable_params(),
        learning_rate=ms.Tensor(lr_values, dtype=ms.float32),
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 使用改进的损失函数
    loss_fn = create_improved_loss_function(class_weights, Config.NUM_CLASSES)
    logging.info("使用自适应Focal Loss损失函数")
    
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
        # 应用简化的学习策略
        focus_mode = apply_progressive_learning(epoch, Config.EPOCHS, loss_fn)
        
        # 训练阶段
        network.set_train()
        total_loss = 0
        step_time = time.time()
        
        for step, (data, label) in enumerate(train_dataset.create_tuple_iterator()):
            loss = train_step(data, label)
            total_loss += loss.asnumpy()
            
            # 快速训练模式下减少日志频率
            log_interval = 20 if getattr(Config, 'FAST_TRAINING', False) else 10
            if (step + 1) % log_interval == 0:
                step_time = time.time() - step_time
                logging.info(f"epoch: {epoch + 1} step: {step + 1}, loss is {loss.asnumpy()}")
                logging.info(f"step time: {step_time * 1000:.2f} ms")
                step_time = time.time()
        
        # 验证阶段
        network.set_train(False)
        correct = 0
        total = 0
        class_correct = [0] * Config.NUM_CLASSES
        class_total = [0] * Config.NUM_CLASSES
        
        for data, label in valid_dataset.create_tuple_iterator():
            logits = network(data)
            pred = ms.ops.Argmax(axis=1)(logits)
            correct += (pred == label).sum().asnumpy()
            total += label.shape[0]
            
            # 统计每个类别的准确率
            pred_numpy = pred.asnumpy()
            label_numpy = label.asnumpy()
            
            for i in range(Config.NUM_CLASSES):
                class_mask = (label_numpy == i)
                class_total[i] += np.sum(class_mask)
                
                if np.sum(class_mask) > 0:
                    class_predictions = pred_numpy[class_mask]
                    class_correct[i] += np.sum(class_predictions == i)
        
        accuracy = correct / total if total > 0 else 0
        logging.info(f"Validation accuracy: {accuracy:.2%}")
        
        # 打印每个类别的准确率
        non_zero_classes = 0
        for i in range(Config.NUM_CLASSES):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                class_acc = max(0.0, min(1.0, class_acc))
                status = "[OK]" if class_acc > 0 else "[NO]"
                logging.info(f"Class {Config.CLASS_NAMES[i]} accuracy: {class_acc:.2%} ({class_correct[i]}/{class_total[i]}) {status}")
                
                if class_acc > 0:
                    non_zero_classes += 1
            else:
                logging.info(f"Class {Config.CLASS_NAMES[i]} accuracy: N/A (no samples)")
        
        # 显示有学习效果的类别数量
        logging.info(f"Learning classes: {non_zero_classes}/{Config.NUM_CLASSES}")
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            no_improve_epochs = 0
            best_ckpt_file = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_PREFIX}_best.ckpt")
            ms.save_checkpoint(network, best_ckpt_file)
            logging.info(f"整体准确率提升到: {accuracy:.2%} - 已保存最佳模型")
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
        "early_stopped": no_improve_epochs >= Config.EARLY_STOPPING_PATIENCE,
        "class_weights": class_weights.tolist()
    }
    
    with open(os.path.join(Config.LOG_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    train()