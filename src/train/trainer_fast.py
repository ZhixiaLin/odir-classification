"""
Fast Training module for Eye Disease Classification
"""

import os
import time
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback
from src.config.config import TRAIN_CONFIG, MODEL_CONFIG
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

    def epoch_end(self, run_context):
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

def train_fast():
    """
    快速训练模型
    
    Returns:
        dict: 训练结果
    """
    # 设置设备
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    
    # 获取数据集
    train_dataset, valid_dataset = get_dataset()
    
    # 获取模型
    network = get_model()
    
    # 定义损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(network.trainable_params(),
                          learning_rate=TRAIN_CONFIG['learning_rate'] * 2,  # 增大学习率
                          momentum=TRAIN_CONFIG['momentum'],
                          weight_decay=TRAIN_CONFIG['weight_decay'])
    
    # 定义模型
    model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    # 设置回调函数
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    eval_cb = EvalCallback(model, valid_dataset, TRAIN_CONFIG['save_dir'])
    
    # 训练模型
    print("Starting fast training...")
    start_time = time.time()
    
    # 减少训练轮数
    fast_epochs = TRAIN_CONFIG['epochs'] // 4
    
    model.train(fast_epochs,
                train_dataset,
                callbacks=[time_cb, loss_cb, eval_cb],
                dataset_sink_mode=True)
    
    training_time = time.time() - start_time
    print(f"Fast training completed in {training_time:.2f} seconds")
    
    # 返回训练结果
    return {
        'best_accuracy': eval_cb.best_acc,
        'training_time': training_time
    }

if __name__ == "__main__":
    train_fast() 