import os
import sys
import argparse
from mindspore import context
from mindspore import Model
from mindspore import nn
from mindspore import ops
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from config.config import Config
import mindspore as ms

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.odir import ODIRDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train ODIR dataset')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    parser.add_argument('--fast', action='store_true',
                      help='enable fast training mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set context
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    
    # Load config
    config = Config(args.config)
    if args.fast:
        config.config['fast_training'] = {'enabled': True}
        config._apply_fast_training()
    
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
    
    # Create model
    network = create_model(
        model_name=config['model'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )
    
    # Load checkpoint if specified
    if config['ckpt_path']:
        ms.load_checkpoint(config['ckpt_path'], network)
    
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
    loss_cb = LossMonitor()
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch,
        keep_checkpoint_max=config['keep_checkpoint_max']
    )
    ckpt_cb = ModelCheckpoint(
        prefix=config['model'],
        directory=config['ckpt_save_dir'],
        config=ckpt_config
    )
    callbacks = [time_cb, loss_cb, ckpt_cb]
    
    # Train
    print(f"Starting training for {config['epoch_size']} epochs...")
    model.train(
        config['epoch_size'],
        dataset_train,
        callbacks=callbacks,
        dataset_sink_mode=config['dataset_sink_mode']
    )
    print("Training completed!")

if __name__ == '__main__':
    main() 