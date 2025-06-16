"""
Logging utilities for the project.
"""

import os
import json
import datetime
import logging
from pathlib import Path

def setup_logging(config, args, start_epoch=0):
    """
    Set up logging for training.
    
    Args:
        config: Configuration object
        args: Command line arguments
        start_epoch (int): Starting epoch number
        
    Returns:
        tuple: (log_file_path, loss_log_file_path)
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log files with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{timestamp}.log'
    loss_log_file = log_dir / f'loss_{timestamp}.log'
    
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
    
    # Write configuration to log file
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
    
    return str(log_file), str(loss_log_file)

def setup_validation_logging():
    """
    Set up logging for validation.
    
    Returns:
        str: Path to the validation log file
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'validation_{timestamp}.log'
    
    # Initialize log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Validation Log:\n")
        f.write("="*50 + "\n")
    
    return str(log_file)

def setup_prediction_logging():
    """
    Set up logging for prediction.
    
    Returns:
        str: Path to the prediction log file
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_{timestamp}.log'
    
    # Initialize log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Prediction Log:\n")
        f.write("="*50 + "\n")
    
    return str(log_file) 