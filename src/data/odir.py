import os
import mindspore as ms
from mindspore.dataset import ImageFolderDataset
from mindspore.dataset.transforms import Compose
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
    Grayscale,
    ToPIL
)
from pathlib import Path

class ODIRDataset:
    """ODIR dataset loader"""
    
    def __init__(self, root, split='train', num_parallel_workers=8, **kwargs):
        """
        Args:
            root (str): Root directory of the dataset
            split (str): 'train' or 'valid'
            num_parallel_workers (int): Number of parallel workers
        """
        self.root = str(Path(root).resolve())
        self.split = split
        self.num_parallel_workers = num_parallel_workers
        
        # Dataset paths
        self.data_dir = str(Path(self.root) / split)
        
        # Define class names
        self.classes = [
            'normal', 'cataract', 'glaucoma', 'amd',
            'hypertension', 'diabetic_retinopathy', 'myopia', 'other'
        ]
        
        # Create dataset
        self.dataset = ImageFolderDataset(
            self.data_dir,
            num_parallel_workers=num_parallel_workers,
            shuffle=(split == 'train')
        )
        
        # Get class names
        if os.path.exists(self.data_dir):
            self.class_names = sorted([d for d in os.listdir(self.data_dir) 
                                     if os.path.isdir(os.path.join(self.data_dir, d))])
        else:
            self.class_names = []
        self.num_classes = len(self.class_names)
    
    def get_dataset(self, batch_size=32, is_training=True, image_size=224):
        """
        Get dataset with transformations
        
        Args:
            batch_size (int): Batch size
            is_training (bool): Whether in training mode
            image_size (int): Image size
            
        Returns:
            Dataset: Transformed dataset
        """
        # First ensure proper decoding of images
        dataset = self.dataset.map(
            operations=[
                Decode(),  # Decode to RGB
                ToPIL(),   # Convert to PIL Image
                Grayscale(num_output_channels=3),  # Ensure 3 channels
            ],
            input_columns="image",
            num_parallel_workers=self.num_parallel_workers
        )
        
        if is_training:
            # Training transforms
            dataset = dataset.map(
                operations=[
                    RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    RandomHorizontalFlip(),
                    RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                    Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
                    HWC2CHW()  # Convert to CHW format after normalization
                ],
                input_columns="image",
                num_parallel_workers=self.num_parallel_workers
            )
        else:
            # Validation transforms
            dataset = dataset.map(
                operations=[
                    Resize(image_size),
                    CenterCrop(image_size),
                    Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
                    HWC2CHW()  # Convert to CHW format after normalization
                ],
                input_columns="image",
                num_parallel_workers=self.num_parallel_workers
            )
        
        # Batch the dataset
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        return dataset 