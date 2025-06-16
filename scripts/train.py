"""
Training script for ODIR dataset.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import mindspore as ms
from mindspore import context
from mindspore import Model
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ODIRDataset
from src.models.resnet import ODIRModel
from src.utils.callbacks import LossCallback, EpochEndCallback
from src.utils.logging import setup_logging
from config.config import Config

def parse_args():
    """Parse command line arguments."""
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

def find_latest_checkpoint(checkpoint_dir: str, model_name: str) -> str:
    """
    Find the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name of the model
        
    Returns:
        Path to the latest checkpoint file
    """
    pattern = os.path.join(checkpoint_dir, f"{model_name}-*.ckpt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def load_checkpoint_info(checkpoint_path: str) -> int:
    """
    Load epoch information from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Epoch number
    """
    try:
        param_dict = ms.load_checkpoint(checkpoint_path)
        # Try to extract epoch from parameter names
        for key in param_dict.keys():
            if 'epoch' in key.lower():
                epoch = int(key.split('_')[-1])
                return epoch
    except:
        pass
    return 0

def main():
    """Main training function."""
    args = parse_args()
    
    # Set context
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
    
    # Check for existing checkpoint
    checkpoint_dir = config['ckpt_save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, config['model'])
    
    start_epoch = 0
    if latest_checkpoint:
        start_epoch = load_checkpoint_info(latest_checkpoint)
        print(f"\nFound checkpoint file: {latest_checkpoint}")
        print(f"Last training stopped at epoch {start_epoch}")
        
        # Automatically decide whether to use checkpoint
        if start_epoch > 0 and start_epoch < config['epoch_size']:
            print(f"Automatically continuing from epoch {start_epoch}")
            config.config['ckpt_path'] = latest_checkpoint
        else:
            print("Checkpoint invalid or training completed, starting from scratch")
            config.config['ckpt_path'] = ''
    else:
        print("No checkpoint found, starting from scratch")
        config.config['ckpt_path'] = ''
    
    # Setup logging
    log_file, loss_log_file = setup_logging(config, args, start_epoch)
    
    # Create datasets
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
    model = ODIRModel.create_model(config)
    network = model.get_model()
    
    # Load checkpoint if specified
    if config['ckpt_path']:
        model.load_checkpoint(config['ckpt_path'], strict=False)
    
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
        model.get_trainable_params(),
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