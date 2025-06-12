"""
Fast Training module for Eye Disease Classification - FIXED VERSION
"""

import os
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback
from mindspore import context
from src.config.config import Config
from src.models.resnet import get_model
from src.data.dataset import get_dataset

class EvalCallback(Callback):
    """评估回调函数"""
    def __init__(self, model, eval_dataset, save_path):
        super(EvalCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.save_path = save_path
        self.best_acc = 0.0

    def on_train_epoch_end(self, run_context):
        """每个epoch结束时调用"""
        cb_params = run_context.original_args()
        result = self.model.eval(self.eval_dataset)
        acc = result['accuracy']
        
        if acc > self.best_acc:
            self.best_acc = acc
            # 保存最佳模型
            ms.save_checkpoint(cb_params.train_network, 
                             os.path.join(self.save_path, 'best_model_fast.ckpt'))
            print(f"Best accuracy: {acc:.4f}")

def calculate_class_weights_fast(train_dataset):
    """快速计算类别权重"""
    # 统计每个类别的样本数
    class_counts = {}
    total_samples = 0
    
    # 遍历数据集统计类别分布
    for data in train_dataset.create_dict_iterator():
        labels = data['label'].asnumpy()
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += 1
    
    # 计算权重
    num_classes = len(class_counts)
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(1.0)  # 如果某个类别不存在，给默认权重
    
    return np.array(weights, dtype=np.float32)

def train_fast():
    """
    快速训练模型
    
    Returns:
        dict: 训练结果
    """
    # 设置设备和上下文
    ms.set_device(Config.DEVICE_TARGET)
    context.set_context(mode=context.GRAPH_MODE)
    
    # 获取数据集
    train_dataset = get_dataset(Config.TRAIN_DATA_PATH, is_training=True)
    valid_dataset = get_dataset(Config.VALID_DATA_PATH, is_training=False)
    
    # 计算类别权重
    try:
        class_weights = calculate_class_weights_fast(train_dataset)
        print(f"Calculated class weights: {class_weights}")
    except Exception as e:
        print(f"Error calculating class weights: {e}")
        # 使用默认权重
        num_classes = Config.NUM_CLASSES
        class_weights = np.ones(num_classes, dtype=np.float32)
    
    # 获取模型
    network = get_model()
    
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss(
        weight=ms.Tensor(class_weights, dtype=ms.float32),
        reduction='mean'
    )
    
    optimizer = nn.Momentum(
        network.trainable_params(),
        learning_rate=Config.LEARNING_RATE * 2,  # 增大学习率
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 定义模型
    model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    # 设置保存目录
    save_dir = Config.CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置回调函数
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    
    # 设置检查点保存
    config_ck = CheckpointConfig(
        save_checkpoint_steps=train_dataset.get_dataset_size(),
        keep_checkpoint_max=5
    )
    ckpoint_cb = ModelCheckpoint(
        prefix="fast_model",
        directory=save_dir,
        config=config_ck
    )
    
    eval_cb = EvalCallback(model, valid_dataset, save_dir)
    
    # 训练模型
    print("Starting fast training...")
    start_time = time.time()
    
    # 减少训练轮数
    fast_epochs = max(1, Config.EPOCHS // 4)
    print(f"Training for {fast_epochs} epochs...")
    
    try:
        model.train(
            fast_epochs,
            train_dataset,
            callbacks=[time_cb, loss_cb, eval_cb, ckpoint_cb],
            dataset_sink_mode=False  # 设置为False以获得更好的兼容性
        )
    except Exception as e:
        print(f"Training error: {e}")
        # 如果出现问题，尝试更简单的训练方式
        print("Trying alternative training approach...")
        
        # 简单的训练循环
        network.set_train()
        best_acc = 0.0
        
        for epoch in range(fast_epochs):
            epoch_loss = 0
            step_count = 0
            
            for data in train_dataset.create_dict_iterator():
                images = data['image']
                labels = data['label']
                
                # 前向传播
                logits = network(images)
                loss = loss_fn(logits, labels)
                
                # 反向传播
                optimizer.clear_gradients()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.asnumpy()
                step_count += 1
            
            avg_loss = epoch_loss / step_count
            print(f"Epoch {epoch + 1}/{fast_epochs}, Average Loss: {avg_loss:.4f}")
            
            # 验证
            if epoch % 2 == 0:  # 每2个epoch验证一次
                network.set_train(False)
                correct = 0
                total = 0
                
                for data in valid_dataset.create_dict_iterator():
                    images = data['image']
                    labels = data['label']
                    
                    logits = network(images)
                    pred = ms.ops.Argmax(axis=1)(logits)
                    correct += (pred == labels).sum().asnumpy()
                    total += labels.shape[0]
                
                accuracy = correct / total if total > 0 else 0
                print(f"Validation accuracy: {accuracy:.2%}")
                
                if accuracy > best_acc:
                    best_acc = accuracy
                    # 保存最佳模型
                    best_ckpt_file = os.path.join(save_dir, "best_fast_model.ckpt")
                    ms.save_checkpoint(network, best_ckpt_file)
                    print(f"Saved best model with accuracy: {accuracy:.2%}")
                
                network.set_train(True)
    
    training_time = time.time() - start_time
    print(f"Fast training completed in {training_time:.2f} seconds")
    
    # 返回训练结果
    return {
        'best_accuracy': eval_cb.best_acc if hasattr(eval_cb, 'best_acc') else best_acc,
        'training_time': training_time
    }

if __name__ == "__main__":
    train_fast() 