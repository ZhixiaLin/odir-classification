import os
import sys
import argparse
from mindspore import context
from mindspore import Model
from mindspore import nn
from mindspore import ops
from mindspore import load_checkpoint, load_param_into_net
from mindcv.models import create_model
from mindcv.loss import create_loss
from config.config import Config
import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import set_context, GRAPH_MODE, set_device

def parse_args():
    parser = argparse.ArgumentParser(description='Predict single image')
    parser.add_argument('--config', type=str, default='configs/odir/resnet50.yaml',
                      help='path to config file')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/resnet50-best.ckpt',
                      help='path to checkpoint file')
    parser.add_argument('--image_path', type=str, required=True,
                      help='path to the image to predict')
    parser.add_argument('--device_target', type=str, default='CPU',
                      help='device target, support CPU, GPU, Ascend')
    return parser.parse_args()

def preprocess_image(image_path, image_size):
    # Read image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms using new API
    transform = [
        vision.Resize(image_size),
        vision.CenterCrop(image_size),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW()
    ]
    
    # Create dataset
    dataset = ds.GeneratorDataset(
        lambda: [(image, 0)],  # Dummy label
        column_names=["image", "label"],
        shuffle=False
    )
    
    # Apply transforms
    dataset = dataset.map(transform, input_columns=["image"])
    dataset = dataset.batch(1)
    
    return dataset

def main():
    args = parse_args()
    
    # Set context
    set_context(mode=GRAPH_MODE)
    set_device(args.device_target)
    
    # Load config
    config = Config(args.config)
    
    # Create model
    network = create_model(
        model_name=config['model'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    # Load checkpoint
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    
    # Get class names
    class_names = sorted(os.listdir(os.path.join(config['data_dir'], config['val_split'])))
    
    # Preprocess image
    dataset = preprocess_image(args.image_path, config['image_size'])
    
    # Predict
    for data in dataset:
        image, _ = data
        pred = network(image)
        probabilities = ops.softmax(pred, axis=1)
        
        # Get all predictions
        probs = probabilities[0].asnumpy()
        indices = np.argsort(probs)[::-1]  # Sort in descending order
        
        print("\nPrediction Results:")
        print("-" * 50)
        for idx in indices:
            prob = probs[idx]
            if prob > 0:  # Only show predictions with confidence > 0
                print(f"Class: {class_names[idx]}")
                print(f"Confidence: {prob*100:.2f}%")
                print("-" * 50)

if __name__ == '__main__':
    main() 