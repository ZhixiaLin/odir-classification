import os
import sys
import argparse
from mindspore import context
from mindspore import Model
from mindspore import nn
from mindspore import ops
from mindcv.models import create_model
from mindcv.loss import create_loss
from config.config import Config

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.odir import ODIRDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Validate ODIR dataset')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='path to checkpoint file')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set context
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    
    # Load config
    config = Config(args.config)
    
    # Create dataset using our custom ODIRDataset
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
        pretrained=False,
        ckpt_path=args.ckpt_path
    )
    
    # Create loss
    loss = create_loss(
        name=config['loss'],
        label_smoothing=config['label_smoothing'],
        reduction=config['reduction']
    )
    
    # Create model
    model = Model(network, loss_fn=loss, metrics={'acc', 'top5_acc'})
    
    # Evaluate
    print("Starting evaluation...")
    result = model.eval(dataset_val)
    print(f"Evaluation results:")
    print(f"Top-1 accuracy: {result['acc']:.4f}")
    print(f"Top-5 accuracy: {result['top5_acc']:.4f}")

if __name__ == '__main__':
    main() 