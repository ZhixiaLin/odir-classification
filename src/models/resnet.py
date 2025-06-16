"""
ResNet model implementation for ODIR dataset.
"""

import mindspore as ms
from mindspore import nn
from mindcv.models import create_model
from typing import Optional, Dict, Any

class ODIRModel:
    """ODIR model wrapper."""
    
    def __init__(self,
                 model_name: str = 'resnet50',
                 num_classes: int = 8,
                 pretrained: bool = True,
                 **kwargs):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Create model
        self.network = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       strict: bool = True) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys match
        """
        param_dict = ms.load_checkpoint(checkpoint_path)
        
        if not strict:
            # Filter out classifier parameters if they don't match
            if 'classifier.weight' in param_dict and param_dict['classifier.weight'].shape != self.network.classifier.weight.shape:
                del param_dict['classifier.weight']
            if 'classifier.bias' in param_dict and param_dict['classifier.bias'].shape != self.network.classifier.bias.shape:
                del param_dict['classifier.bias']
        
        ms.load_param_into_net(self.network, param_dict)
    
    def get_trainable_params(self) -> list:
        """
        Get trainable parameters.
        
        Returns:
            List of trainable parameters
        """
        return self.network.trainable_params()
    
    def get_model(self) -> nn.Cell:
        """
        Get the model.
        
        Returns:
            The model
        """
        return self.network
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> 'ODIRModel':
        """
        Create model from config.
        
        Args:
            config: Model configuration
            
        Returns:
            ODIRModel instance
        """
        return ODIRModel(
            model_name=config['model'],
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        ) 