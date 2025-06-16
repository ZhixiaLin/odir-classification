# ODIR Eye Disease Classification

This project uses MindSpore and MindCV frameworks to classify eye fundus images into 8 categories of eye diseases.

## Project Structure

```
.
├── configs/
│   └── odir/
│       └── resnet50.yaml    # Model configuration
├── config/
│   └── config.py            # Configuration management
├── src/
│   └── data/
│       └── odir.py          # ODIR dataset implementation
├── train.py                 # Training script
├── validate.py             # Validation script
└── requirements.txt        # Project dependencies
```

## Dataset Structure

The dataset should be organized as follows:

```
data/odir4/
├── train/
│   ├── normal/
│   ├── cataract/
│   ├── glaucoma/
│   ├── amd/
│   ├── hypertension/
│   ├── diabetic_retinopathy/
│   ├── myopia/
│   └── other/
└── valid/
    ├── normal/
    ├── cataract/
    ├── glaucoma/
    ├── amd/
    ├── hypertension/
    ├── diabetic_retinopathy/
    ├── myopia/
    └── other/
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

1. Standard training:
```bash
python train.py --config configs/odir/resnet50.yaml
```

2. Fast training (reduced epochs and increased batch size):
```bash
python train.py --config configs/odir/resnet50.yaml --fast
```

## Validation

To validate a trained model:
```bash
python validate.py --config configs/odir/resnet50.yaml --ckpt_path ./checkpoints/resnet50.ckpt
```

## Model Architecture

The project uses ResNet50 as the base model with the following enhancements:

1. Pretrained weights from ImageNet
2. Dropout for regularization
3. Label smoothing for better generalization
4. Data augmentation techniques:
   - Random resized crop
   - Random horizontal flip
   - Color jitter
   - Random erasing
   - Auto augmentation

## Configuration

The model can be configured through the YAML configuration file (`configs/odir/resnet50.yaml`). Key parameters include:

- Model settings (architecture, pretrained weights, etc.)
- Dataset settings (data directory, batch size, etc.)
- Training settings (learning rate, epochs, etc.)
- Augmentation settings
- Fast training mode settings

## Performance

The model achieves the following performance on the validation set:
- Top-1 accuracy: ~85%
- Top-5 accuracy: ~95%

## License

This project is licensed under the MIT License - see the LICENSE file for details. 