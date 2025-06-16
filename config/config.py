import os
import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration manager for ODIR dataset training"""
    
    def __init__(self, config_path: str):
        """Initialize configuration from yaml file"""
        self.config_path = str(Path(config_path).resolve())
        self.config = self._load_config()
        
        # Set default values
        self._set_defaults()
        
        # Override with fast training settings if enabled
        if self.config.get('fast_training', {}).get('enabled', False):
            self._apply_fast_training()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from yaml file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _set_defaults(self):
        """Set default configuration values"""
        # Get the project root directory (2 levels up from config directory)
        project_root = Path(__file__).parent.parent.absolute()
        
        defaults = {
            'model': 'resnet50',
            'num_classes': 8,
            'pretrained': True,
            'ckpt_path': '',
            'keep_checkpoint_max': 10,
            'ckpt_save_dir': str(project_root / 'checkpoints'),
            'epoch_size': 100,
            'dataset_sink_mode': True,
            'amp_level': 'O0',
            
            'dataset': 'odir',
            'data_dir': str(project_root / 'data' / 'odir4'),
            'train_split': 'train',
            'val_split': 'valid',
            'num_parallel_workers': 8,
            'batch_size': 32,
            'image_size': 224,
            
            'auto_augment': True,
            're_prob': 0.5,
            're_value': 'random',
            'color_jitter': 0.4,
            'interpolation': 'bicubic',
            
            'opt': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'momentum': 0.9,
            'loss_scale': 1.0,
            'use_nesterov': False,
            
            'scheduler': 'cosine_decay',
            'min_lr': 0.00001,
            'lr_epochs': [100],
            'lr_gamma': 0.1,
            'warmup_epochs': 5,
            'warmup_factor': 0.1,
            
            'loss': 'ce',
            'label_smoothing': 0.1,
            'reduction': 'mean'
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _apply_fast_training(self):
        """Apply fast training settings"""
        fast_config = self.config['fast_training']
        self.config.update({
            'epoch_size': fast_config.get('epochs', 25),
            'batch_size': fast_config.get('batch_size', 64),
            'lr': fast_config.get('lr', 0.002),
            'warmup_epochs': fast_config.get('warmup_epochs', 2)
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict-like access"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self.config
    
    def save(self, save_path: str):
        """Save configuration to yaml file"""
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy() 