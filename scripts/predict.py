"""
Prediction script for single image inference on ODIR dataset.
"""

import os
import sys
import argparse
from pathlib import Path

import mindspore as ms
from mindspore import context
import numpy as np
from PIL import Image

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import ImagePreprocessor
from src.models.resnet import ODIRModel
from src.utils.logging import setup_prediction_logging
from config.config import Config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict single image')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='path to checkpoint file')
    parser.add_argument('--image_path', type=str, required=True,
                      help='path to input image')
    return parser.parse_args()

def load_and_preprocess_image(image_path: str, image_size: int):
    """
    Load and preprocess image for prediction.
    
    Args:
        image_path: Path to the input image
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms
    transforms = ImagePreprocessor.get_prediction_transforms(image_size)
    
    # Apply transforms
    for transform in transforms:
        image = transform(image)
    
    # Add batch dimension
    image = image.expand_dims(axis=0)
    
    return image

def main():
    """Main prediction function."""
    args = parse_args()
    
    # Set context
    ms.set_device(args.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    
    # Load config
    config = Config(args.config)
    
    # Setup logging
    log_file = setup_prediction_logging()
    
    # Create model
    model = ODIRModel.create_model(config)
    network = model.get_model()
    
    # Load checkpoint
    model.load_checkpoint(args.ckpt_path, strict=True)
    
    # Load and preprocess image
    image = load_and_preprocess_image(args.image_path, config['image_size'])
    
    # Make prediction
    print("\nMaking prediction...")
    pred = network(image)
    pred = pred.asnumpy()
    
    # Get top-5 predictions
    top5_idx = np.argsort(pred[0])[-5:][::-1]
    top5_probs = pred[0][top5_idx]
    
    # Print results
    print("\nTop-5 predictions:")
    for idx, prob in zip(top5_idx, top5_probs):
        class_name = config['class_names'][idx]
        print(f"{class_name}: {prob:.4f}")
    
    # Save results to log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Top-5 predictions:\n")
        for idx, prob in zip(top5_idx, top5_probs):
            class_name = config['class_names'][idx]
            f.write(f"{class_name}: {prob:.4f}\n")

if __name__ == '__main__':
    main()

 