import os
import sys
import argparse
from mindspore import context
from mindspore import Model
from mindspore import nn
from mindspore import ops
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig, Callback
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from config.config import Config
import mindspore as ms
import time
import datetime
import json
import glob

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.odir import ODIRDataset

class LossCallback(Callback):
    def __init__(self, loss_log_file):
        super(LossCallback, self).__init__()
        self.loss_log_file = loss_log_file
        self.losses = []
        self.epoch = 1
        self.step = 0
        
    def on_train_epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        self.epoch = cb_params.cur_epoch_num
        self.step = 0
        
    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        self.losses.append(float(loss))
        self.step += 1
        
        # Write detailed loss to file
        with open(self.loss_log_file, 'a', encoding='utf-8') as f:
            f.write(f"epoch: {self.epoch} step: {self.step}, loss is {float(loss):.6f}\n")
        
        # Print to terminal (original format)
        print(f"epoch: {self.epoch} step: {self.step}, loss is {float(loss):.6f}")

class EpochEndCallback(Callback):
    def __init__(self, model, eval_dataset, log_file, loss_log_file):
        super(EpochEndCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.epoch_time = time.time()
        self.log_file = log_file
        self.loss_log_file = loss_log_file
        self.losses = []
        
    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        self.losses.append(float(loss))
        
    def on_train_epoch_end(self, run_context):
        epoch_time = time.time() - self.epoch_time
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        
        # Calculate average loss for this epoch
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        self.losses = []  # Reset losses for next epoch
        
        # Evaluate on validation set
        result = self.model.eval(self.eval_dataset)
        accuracy = result['acc']
        
        # Prepare log message
        log_msg = f"\nEpoch {epoch_num} completed:\n"
        log_msg += f"Average loss: {avg_loss:.6f}\n"
        log_msg += f"Validation accuracy: {accuracy:.4f}\n"
        log_msg += f"Epoch time: {epoch_time:.2f} seconds\n"
        
        # Print to console
        print(log_msg)
        
        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg)
        
        # Write epoch summary to loss log file
        with open(self.loss_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nEpoch {epoch_num} Summary:\n")
            f.write(f"Average loss: {avg_loss:.6f}\n")
            f.write(f"Validation accuracy: {accuracy:.4f}\n")
            f.write(f"Epoch time: {epoch_time:.2f} seconds\n")
            f.write("="*50 + "\n")
        
        self.epoch_time = time.time()

def find_latest_checkpoint(checkpoint_dir, model_name):
    """查找最新的检查点文件"""
    pattern = os.path.join(checkpoint_dir, f"{model_name}-*.ckpt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    # 按修改时间排序
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def load_checkpoint_info(checkpoint_path):
    """加载检查点信息"""
    try:
        param_dict = ms.load_checkpoint(checkpoint_path)
        # 尝试从参数名中提取epoch信息
        for key in param_dict.keys():
            if 'epoch' in key.lower():
                epoch = int(key.split('_')[-1])
                return epoch
    except:
        pass
    return 0

def setup_logging(config, args, start_epoch=0):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    loss_log_file = os.path.join(log_dir, f'loss_{timestamp}.log')
    
    # Log initial configuration
    config_dict = {
        'timestamp': timestamp,
        'config_file': args.config,
        'fast_mode': args.fast,
        'device_target': args.device_target,
        'start_epoch': start_epoch,
        'model_config': {
            'model': config['model'],
            'num_classes': config['num_classes'],
            'pretrained': config['pretrained'],
            'epoch_size': config['epoch_size'],
            'batch_size': config['batch_size'],
            'learning_rate': config['lr'],
            'optimizer': config['opt'],
            'loss': config['loss'],
            'label_smoothing': config['label_smoothing']
        }
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Training Configuration:\n")
        f.write(json.dumps(config_dict, indent=2))
        f.write("\n\nTraining Log:\n")
        f.write("="*50 + "\n")
    
    # Initialize loss log file
    with open(loss_log_file, 'w', encoding='utf-8') as f:
        f.write("Training Loss Log:\n")
        f.write("="*50 + "\n")
        f.write(f"Configuration:\n")
        f.write(json.dumps(config_dict, indent=2))
        f.write("\n\nDetailed Loss Records:\n")
        f.write("="*50 + "\n")
    
    return log_file, loss_log_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train ODIR dataset')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    parser.add_argument('--fast', action='store_true',
                      help='enable fast training mode')
    parser.add_argument('--fast_epochs', type=int,
                      help='number of epochs in fast mode (overrides config)')
    parser.add_argument('--fast_max_steps', type=int,
                      help='maximum steps per epoch in fast mode (overrides config)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set context using new API
    ms.set_device(args.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    
    # Load config
    config = Config(args.config)
    if args.fast:
        config.config['fast_training'] = {'enabled': True}
        # Override fast training parameters if specified
        if args.fast_epochs is not None:
            config.config['fast_training']['epochs'] = args.fast_epochs
        if args.fast_max_steps is not None:
            config.config['fast_training']['max_steps_per_epoch'] = args.fast_max_steps
        config._apply_fast_training()
    
    # 检查是否存在检查点
    checkpoint_dir = config['ckpt_save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, config['model'])
    
    start_epoch = 0
    if latest_checkpoint:
        start_epoch = load_checkpoint_info(latest_checkpoint)
        print(f"\n发现检查点文件: {latest_checkpoint}")
        print(f"上次训练停止在 epoch {start_epoch}")
        
        # 自动决定是否使用检查点
        if start_epoch > 0 and start_epoch < config['epoch_size']:
            print(f"自动从 epoch {start_epoch} 继续训练")
            config.config['ckpt_path'] = latest_checkpoint
        else:
            print("检查点无效或训练已完成，将从头开始训练")
            config.config['ckpt_path'] = ''
    else:
        print("未找到检查点，将从头开始训练")
        config.config['ckpt_path'] = ''
    
    # Setup logging
    log_file, loss_log_file = setup_logging(config, args, start_epoch)
    
    # Create dataset using our custom ODIRDataset
    odir_train = ODIRDataset(
        root=config['data_dir'],
        split=config['train_split'],
        num_parallel_workers=config['num_parallel_workers']
    )
    
    dataset_train = odir_train.get_dataset(
        batch_size=config['batch_size'],
        is_training=True,
        image_size=config['image_size']
    )
    
    # Project only image and label columns for training
    dataset_train = dataset_train.project(columns=["image", "label"])
    
    # Apply max steps per epoch limit if specified
    if config.get('max_steps_per_epoch') is not None:
        steps_per_epoch = dataset_train.get_dataset_size()
        if steps_per_epoch > config['max_steps_per_epoch']:
            print(f"\nLimiting training to {config['max_steps_per_epoch']} steps per epoch")
            dataset_train = dataset_train.take(config['max_steps_per_epoch'])
    
    odir_val = ODIRDataset(
        root=config['data_dir'],
        split=config['val_split'],
        num_parallel_workers=config['num_parallel_workers']
    )
    
    dataset_val = odir_val.get_dataset(
        batch_size=config['batch_size'],
        is_training=False,
        image_size=config['image_size']
    )
    
    # Project only image and label columns for validation
    dataset_val = dataset_val.project(columns=["image", "label"])
    
    # Create model
    network = create_model(
        model_name=config['model'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )
    
    # Load checkpoint if specified
    if config['ckpt_path']:
        param_dict = ms.load_checkpoint(config['ckpt_path'])
        # Filter out classifier parameters if they don't match
        if 'classifier.weight' in param_dict and param_dict['classifier.weight'].shape != network.classifier.weight.shape:
            del param_dict['classifier.weight']
        if 'classifier.bias' in param_dict and param_dict['classifier.bias'].shape != network.classifier.bias.shape:
            del param_dict['classifier.bias']
        ms.load_param_into_net(network, param_dict)
    
    # Create loss
    loss = create_loss(
        name=config['loss'],
        label_smoothing=config['label_smoothing'],
        reduction=config['reduction']
    )
    
    # Create optimizer
    steps_per_epoch = dataset_train.get_dataset_size()
    total_steps = steps_per_epoch * config['epoch_size']
    
    # Create learning rate scheduler
    lr = create_scheduler(
        steps_per_epoch=steps_per_epoch,
        scheduler=config['scheduler'],
        lr=config['lr'],
        min_lr=config['min_lr'],
        warmup_epochs=config['warmup_epochs'],
        warmup_factor=config['warmup_factor'],
        decay_epochs=config['epoch_size'],
        decay_rate=config['lr_gamma']
    )
    
    opt = create_optimizer(
        network.trainable_params(),
        opt=config['opt'],
        lr=lr,
        weight_decay=config['weight_decay'],
        momentum=config['momentum'],
        nesterov=config['use_nesterov'],
        loss_scale=config['loss_scale']
    )
    
    # Create model
    model = Model(network, loss_fn=loss, optimizer=opt, metrics={'acc'})
    
    # Set callbacks
    time_cb = TimeMonitor(data_size=steps_per_epoch)
    loss_cb = LossCallback(loss_log_file)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch,
        keep_checkpoint_max=config['keep_checkpoint_max']
    )
    ckpt_cb = ModelCheckpoint(
        prefix=config['model'],
        directory=config['ckpt_save_dir'],
        config=ckpt_config
    )
    epoch_end_cb = EpochEndCallback(model, dataset_val, log_file, loss_log_file)
    callbacks = [time_cb, loss_cb, ckpt_cb, epoch_end_cb]
    
    # Train
    start_msg = f"Starting training from epoch {start_epoch} for {config['epoch_size']} epochs..."
    print(start_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(start_msg + "\n")
    
    model.train(
        config['epoch_size'],
        dataset_train,
        callbacks=callbacks,
        dataset_sink_mode=config['dataset_sink_mode']
    )
    
    end_msg = "Training completed!"
    print(end_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(end_msg + "\n")

if __name__ == '__main__':
    main() 