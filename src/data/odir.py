import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomColorAdjust,
    Resize,
    CenterCrop,
    Normalize,
    HWC2CHW,
)

class ODIRDataset:
    """
    ODIR dataset loader using GeneratorDataset.
    This version loads the image inside __getitem__ to simplify the data pipeline.
    """
    def __init__(self, root, split='train', num_parallel_workers=8, **kwargs):
        self.root = root
        self.split = split
        self.num_parallel_workers = num_parallel_workers # Kept for consistency, but see note in get_dataset
        self.data_dir = str(Path(self.root) / self.split)
        self._data = []
        self.class_name_to_id = {}

        if os.path.isdir(self.data_dir):
            self.class_names = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            self.class_name_to_id = {name: i for i, name in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)

            for class_name in self.class_names:
                class_id = self.class_name_to_id[class_name]
                class_dir = os.path.join(self.data_dir, class_name)
                image_files = glob.glob(os.path.join(class_dir, '*.[jJ][pP][gG]')) + \
                              glob.glob(os.path.join(class_dir, '*.[jJ][pP][eE][gG]')) + \
                              glob.glob(os.path.join(class_dir, '*.[pP][nN][gG]'))
                for path in image_files:
                    self._data.append((path, class_id))
        else:
            print(f"Warning: Data directory not found at {self.data_dir}")
            self.class_names = []
            self.num_classes = 0

    def __getitem__(self, index):
        """
        Returns the data point at a given index.
        Image is loaded here.
        """
        image_path, label = self._data[index]
        # ✅ **Key Change 1: Load the image directly inside __getitem__**
        image = Image.open(image_path).convert("RGB")
        # ✅ **Return all three items: the PIL image, the label, and the path string.**
        return image, np.array(label, dtype=np.int32), image_path

    def __len__(self):
        return len(self._data)

    def get_dataset(self, batch_size=32, is_training=True, image_size=224):
        # On Windows, multiprocessing is not supported, so workers will default to 1.
        # We can set it to 1 explicitly to avoid the warning.
        num_workers = 1 if os.name == 'nt' else self.num_parallel_workers

        # ✅ **Key Change 2: The GeneratorDataset now provides all 3 columns from the start.**
        dataset = ds.GeneratorDataset(
            source=self,
            column_names=["image", "label", "image_path"], # Matches the output of __getitem__
            shuffle=(self.split == 'train'),
            num_parallel_workers=num_workers
        )

        if is_training:
            transform_img = Compose([
                RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                RandomHorizontalFlip(),
                RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                          std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
                HWC2CHW()
            ])
        else: # Validation
            transform_img = Compose([
                Resize(image_size),
                CenterCrop(image_size),
                Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                          std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
                HWC2CHW()
            ])

        # ✅ **Key Change 3: The pipeline is now much simpler.**
        # We only need one .map() call to transform the 'image' column.
        # The 'label' and 'image_path' columns are automatically passed through untouched.
        dataset = dataset.map(
            operations=transform_img,
            input_columns=["image"],
            num_parallel_workers=num_workers
        )

        # The .project() operation is no longer needed because the columns are already correct.
        # We just batch the dataset.
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        return dataset