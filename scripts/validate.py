"""
Validation script for ODIR dataset.
"""

import os
import sys
import argparse
from pathlib import Path

import mindspore as ms
from mindspore import context
from mindspore import Model

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ODIRDataset
from src.models.resnet import ODIRModel
from src.utils.metrics import calculate_category_metrics, calculate_overall_metrics, format_metrics_report
from src.utils.logging import setup_validation_logging
from config.config import Config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate ODIR dataset')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='path to checkpoint file')
    return parser.parse_args()

def main():
    """Main validation function."""
    args = parse_args()
    
    # Set context
    ms.set_device(args.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    
    # Load config
    config = Config(args.config)
    
    # Setup logging
    log_file = setup_validation_logging()
    
    # Create validation dataset
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
    model = ODIRModel.create_model(config)
    network = model.get_model()
    
    # Load checkpoint
    model.load_checkpoint(args.ckpt_path, strict=True)
    
    # Create model for evaluation
    eval_model = Model(network, metrics={'acc'})
    
    # Evaluate
    print("\nStarting validation...")
    result = eval_model.eval(dataset_val)
    print(f"\nValidation accuracy: {result['acc']:.4f}")
    
    # Calculate detailed metrics
    predictions = []
    labels = []
    
    for data in dataset_val.create_dict_iterator():
        pred = network(data['image'])
        pred = pred.asnumpy().argmax(axis=1)
        label = data['label'].asnumpy()
        predictions.extend(pred)
        labels.extend(label)
    
    # Calculate metrics
    category_metrics = calculate_category_metrics(
        predictions=predictions,
        labels=labels,
        class_names=config['class_names']
    )
    
    overall_metrics = calculate_overall_metrics(
        predictions=predictions,
        labels=labels
    )
    
    # Format and print report
    report = format_metrics_report(category_metrics, overall_metrics)
    print("\nDetailed metrics report:")
    print(report)
    
    # Save report to log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Validation accuracy: {result['acc']:.4f}\n\n")
        f.write("Detailed metrics report:\n")
        f.write(report)

if __name__ == '__main__':
    main() 