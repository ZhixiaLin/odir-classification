"""
Data preprocessing utilities for the ODIR dataset.
"""

import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset.vision import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomColorAdjust,
    Resize,
    CenterCrop,
    Normalize,
    HWC2CHW,
    Decode,
    RandomRotation,
    RandomAffine,
    RandomErasing
)

class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    @staticmethod
    def get_training_transforms(image_size: int):
        """
        Get training data transforms.
        
        Args:
            image_size: Target image size
            
        Returns:
            List of transforms
        """
        return [
            Decode(),
            Resize((image_size + 64, image_size + 64)),
            RandomResizedCrop(
                size=image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.2),
            RandomRotation(degrees=10),
            RandomColorAdjust(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            RandomErasing(prob=0.3),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    
    @staticmethod
    def get_validation_transforms(image_size: int):
        """
        Get validation data transforms.
        
        Args:
            image_size: Target image size
            
        Returns:
            List of transforms
        """
        return [
            Decode(),
            Resize(image_size + 32),
            CenterCrop(image_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]
    
    @staticmethod
    def get_prediction_transforms(image_size: int):
        """
        Get prediction data transforms.
        
        Args:
            image_size: Target image size
            
        Returns:
            List of transforms
        """
        return [
            Decode(),
            Resize(image_size),
            CenterCrop(image_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            HWC2CHW()
        ]

def create_dataset(data_path: str,
                  batch_size: int,
                  image_size: int,
                  is_training: bool = True,
                  num_parallel_workers: int = 8) -> ds.Dataset:
    """
    Create a dataset with appropriate transforms.
    
    Args:
        data_path: Path to the dataset
        batch_size: Batch size
        image_size: Target image size
        is_training: Whether this is for training
        num_parallel_workers: Number of parallel workers
        
    Returns:
        MindSpore dataset
    """
    # Create dataset
    dataset = ds.ImageFolderDataset(
        data_path,
        num_parallel_workers=num_parallel_workers,
        shuffle=is_training,
        decode=True
    )
    
    # Get transforms
    transforms = (ImagePreprocessor.get_training_transforms(image_size) if is_training
                 else ImagePreprocessor.get_validation_transforms(image_size))
    
    # Apply transforms
    dataset = dataset.map(
        operations=transforms,
        input_columns="image",
        num_parallel_workers=num_parallel_workers
    )
    
    # Set batch size
    dataset = dataset.batch(
        batch_size,
        drop_remainder=is_training,
        num_parallel_workers=num_parallel_workers
    )
    
    return dataset

def create_prediction_dataset(image_path: str,
                            image_size: int,
                            num_parallel_workers: int = 1) -> ds.Dataset:
    """
    Create a dataset for single image prediction.
    
    Args:
        image_path: Path to the image
        image_size: Target image size
        num_parallel_workers: Number of parallel workers
        
    Returns:
        MindSpore dataset
    """
    # Read image
    image = Image.open(image_path).convert('RGB')
    
    # Create dataset
    dataset = ds.GeneratorDataset(
        lambda: [(image, 0)],  # Dummy label
        column_names=["image", "label"],
        shuffle=False
    )
    
    # Get transforms
    transforms = ImagePreprocessor.get_prediction_transforms(image_size)
    
    # Apply transforms
    dataset = dataset.map(
        operations=transforms,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers
    )
    
    # Set batch size
    dataset = dataset.batch(1)
    
    return dataset 