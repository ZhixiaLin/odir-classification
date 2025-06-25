# ODIR 眼疾分类项目

本项目基于 MindSpore 和 MindCV 框架，旨在实现对眼底镜图像的智能分类，能够识别八种不同的眼部疾病。

## 项目简介

该项目利用深度学习技术，通过训练一个强大的 ResNet50 模型来分析眼底图像，并自动区分正常、白内障、糖尿病、青光眼、高血压、近视等多种眼部病理状况。  
我们对模型进行了多项优化，包括使用预训练权重、数据增强、标签平滑等技术，以期达到高精度的分类效果。

## 项目结构

```
ai-model-python
├── checkpoints/
│   └── resnet50-best.ckpt
├── config/
│   └── config.py
├── configs/
│   └── odir/
│       └── resnet50.yaml
├── data/
│   └── odir4/
├── logs/
├── src/
│   └── data/
│       ├── dataset.py
│       └── odir.py
├── .gitignore
├── predict_single.py
├── README_CN_ai-model-python.md
├── requirements.txt
├── setup.py
├── train.py
└── validate_by_category.py
```

## 环境搭建与准备

### 1. 获取项目代码

首先，将本项目克隆或下载解压到您的本地计算机。

```bash
git clone this-repo-url
```
```bash
cd odir-classification
```

###  2. 下载数据集

本项目使用 Ocular Disease Intelligent Recognition (ODIR) 数据集。  
请前往官方网站下载：https://odir2019.grand-challenge.org/  
下载后，请按照以下目录结构组织您的数据集：

## 数据集结构

数据集的结构应该如下表展示：

```
data/odir4/
├── train/
│   ├── g1-cataract/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── g1-diabetes/
│   ├── g1-glaucoma/
│   ├── g1-myopia/
│   ├── g2-hypertension/
│   ├── g2-normal/
│   └── g2-others/
└── valid/
    ├── g1-cataract/
    ├── g1-diabetes/
    ├── g1-glaucoma/
    ├── g1-myopia/
    ├── g2-hypertension/
    ├── g2-normal/
    └── g2-others/
```
### 3. 配置 Python 环境
强烈建议使用 conda 创建一个独立的虚拟环境，以避免包版本冲突。
```bash
# 创建一个名为 py311，Python 版本为 3.11 的虚拟环境
conda create -n py311 python=3.11

# 激活虚拟环境
conda activate py311
```

### 4. 安装 MindSpore 框架

请访问 **[MindSpore 官方网站](https://www.mindspore.cn/)** 获取安装指南。

在官网首页点击 “安装” 标签，根据您的操作系统和硬件选择合适的版本进行安装。以下为安装示例：

**示例 1：Windows / CPU**
* **版本:** 2.6.0
* **硬件平台:** CPU
* **操作系统:** Windows-x64
* **编程语言:** Python 3.11
* **安装方式:** Conda

**示例 2：macOS / CPU (Apple Silicon)**
* **版本:** 2.6.0
* **硬件平台:** CPU
* **操作系统:** MacOS-aarch64
* **编程语言:** Python 3.11
* **安装方式:** Conda

按照官网生成的指令在您已激活的虚拟环境中执行安装。

### 5. 安装项目依赖

最后，安装本项目所需的其他 Python 依赖包。

```bash
pip install -r requirements.txt
```


## 模型训练

### 1. 快速测试训练

在进行完整训练之前，您可以启动一个快速训练模式来验证环境配置是否正确。该模式会减少训练轮数（epochs）并增大批处理大小（batch size），从而能够迅速完成。

```bash
# 启动快速训练模式
python train.py --config configs/odir/resnet50.yaml --fast
```

您还可以自定义快速训练的参数：
```bash
# 自定义快速训练的轮数和最大步数
python train.py --config configs/odir/resnet50.yaml --fast --fast_epochs 5 --fast_max_steps 2
```

### 2. 标准训练
确认一切正常后，即可开始标准的、完整的模型训练。

```bash
# 启动标准训练模式
python train.py --config configs/odir/resnet50.yaml
```
训练过程中，最佳模型权重将自动保存在 checkpoints/ 目录下。


## 模型验证与预测
### 1. 单张图片预测
使用 `predict_single.py` 脚本可以对单张眼底图像进行分类预测。

```bash
# 示例：预测一张青光眼图片
python predict_single.py --image_path data/odir4/valid/g1-glaucoma/image13.png
```

### 2. 批量验证
使用 `validate_by_category.py` 脚本可以评估模型在验证集上的整体性能。
- 模式一：仅计算准确率

```bash
# 该模式将快速计算模型在整个验证集上的分类准确率。
python validate_by_category.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --mode accuracy_only
```

- 模式二：完整验证与错误分析

```bash
# 该模式除了计算准确率外，还会将所有分类错误的图片路径记录到一个日志文件中，便于后续分析。
python validate_by_category.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --mode full
```

## 模型架构与优化

本项目采用经典的 **ResNet50** 作为基础模型，并融合了以下优化策略以提升性能：

* **ImageNet 预训练权重**: 利用在 ImageNet 数据集上预训练好的权重进行迁移学习，加快模型收敛并提高泛化能力。
* **Dropout**: 在全连接层后加入 Dropout，有效防止模型过拟合。
* **标签平滑 (Label Smoothing)**: 一种正则化技术，降低模型对标签的过分自信，提升模型的泛化能力。
* **丰富的数据增强 (Data Augmentation)**:
   * 随机裁剪缩放 (Random Resized Crop)
   * 随机水平翻转 (Random Horizontal Flip)
   * 色彩抖动 (Color Jitter)
   * 随机擦除 (Random Erasing)
   * 自动增强策略 (Auto Augmentation)

## 参数配置
项目的所有关键参数均可通过 YAML 配置文件 (`configs/odir/resnet50.yaml`) 进行灵活调整，主要包括：

* **模型设置**: 如模型结构、是否使用预训练权重等。
* **数据集设置**: 如数据路径、批处理大小（batch size）等。
* **训练策略**: 如学习率、训练轮数（epochs）、优化器等。
* **数据增强设置**: 控制各种数据增强策略的开关与强度。
* **快速训练模式设置**: 用于快速测试的特定参数。