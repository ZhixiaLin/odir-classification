import os
import sys
import argparse
from mindspore import context
from mindspore import Model
from mindspore import nn
from mindspore import ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore import set_context, GRAPH_MODE, set_device
from mindcv.models import create_model
from mindcv.loss import create_loss
from config.config import Config
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.odir import ODIRDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Validate ODIR dataset by category')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='path to checkpoint file')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    parser.add_argument('--mode', type=str, choices=['accuracy_only', 'full'], default='full',
                      help='validation mode: accuracy_only or full (accuracy + file writing)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set context
    set_context(mode=GRAPH_MODE)
    set_device(args.device_target)
    
    # Load config
    config = Config(args.config)
    
    # Create dataset
    odir_val = ODIRDataset(
        root=config['data_dir'],
        split=config['val_split'],
        num_parallel_workers=config['num_parallel_workers']
    )
    
    dataset_val = odir_val.get_dataset(
        batch_size=1,  # Set batch size to 1 for per-image evaluation
        is_training=False,
        image_size=config['image_size']
    )
    
    # Create model
    network = create_model(
        model_name=config['model'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    # Load checkpoint
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    
    # Create loss
    loss = create_loss(
        name=config['loss'],
        label_smoothing=config['label_smoothing'],
        reduction=config['reduction']
    )
    
    # Create model
    model = Model(network, loss_fn=loss, metrics={'acc'})
    
    # Initialize counters
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    misclassified_images = []  # Use list since we don't need to deduplicate anymore
    
    # Get class names
    class_names = sorted(os.listdir(os.path.join(config['data_dir'], config['val_split'])))
    
    # Create misclassified directory if it doesn't exist
    misclassified_dir = os.path.join('data', 'odir4', 'misclassified')
    os.makedirs(misclassified_dir, exist_ok=True)
    output_file = os.path.join(misclassified_dir, 'misclassified_images.txt')
    
    # Check if output file exists (only in full mode)
    if args.mode == 'full' and os.path.exists(output_file):
        print(f"\nWarning: File {output_file} already exists and will be overwritten.")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    print(f"\nStarting evaluation in {args.mode} mode...")
    print(f"Data directory: {os.path.abspath(config['data_dir'])}")
    print(f"Validation split: {config['val_split']}")
    print(f"Classes found: {class_names}")
    
    # Create progress bar
    pbar = tqdm(total=len(dataset_val), desc="Processing images")
    
    # Evaluate each image
    for data in dataset_val:
        image, label, image_path_tensor = data    
        pred = network(image)
        pred_class = ops.argmax(pred, dim=1)
        
        # Get the actual class name
        actual_class = class_names[label.asnumpy()[0]]
        predicted_class = class_names[pred_class.asnumpy()[0]]
        
        # Update statistics
        category_stats[actual_class]['total'] += 1
        if actual_class == predicted_class:
            category_stats[actual_class]['correct'] += 1
        else:
            # Store misclassified image path (only in full mode)
            if args.mode == 'full':
                image_path = image_path_tensor.asnumpy().item()
                if image_path and os.path.exists(image_path):
                    misclassified_images.append(f"{image_path} | Actual: {actual_class} | Predicted: {predicted_class}")
                else:
                    print(f"Warning: Could not find image for class {actual_class}")
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Print final results
    print("\nCategory-wise Results:")
    print("-" * 50)
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"Category: {category}")
        print(f"Correct: {stats['correct']}/{stats['total']}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-" * 50)
    
    # Save misclassified images to file (only in full mode)
    if args.mode == 'full':
        if misclassified_images:
            with open(output_file, 'w') as f:
                for line in sorted(misclassified_images):  # Sort for consistent output
                    f.write(line + '\n')
            print(f"\nMisclassified images have been saved to {output_file}")
            print(f"Total misclassified images: {len(misclassified_images)}")
        else:
            print("\nNo misclassified images found to save.")

if __name__ == '__main__':
    main() 