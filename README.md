# ODIR Eye Disease Classification

基于 MindSpore 和 MindCV 框架的眼部疾病分类系统，支持 8 种常见眼部疾病的自动识别。

## 主要功能

1. **模型训练**
   - 支持标准训练和快速训练模式
   - 自动保存检查点和恢复训练
   - 详细的训练日志记录
   - 支持多种数据增强策略

2. **模型评估**
   - 按类别评估模型性能
   - 生成详细的分类报告
   - 识别错误分类的样本

3. **数据管理**
   - 自动清理错误分类的样本
   - 支持按比例保留样本
   - 数据预处理和增强

4. **单图预测**
   - 支持单张图片的疾病预测
   - 提供详细的预测概率

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd odir-classification
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

标准训练：
```bash
python scripts/train.py --config configs/odir/resnet50.yaml
```

快速训练（减少轮次，增加批次大小）：
```bash
python scripts/train.py --config configs/odir/resnet50.yaml --fast
```

自定义快速训练参数：
```bash
python scripts/train.py --config configs/odir/resnet50.yaml --fast --fast_epochs 5 --fast_max_steps 2
```

指定设备：
```bash
python scripts/train.py --config configs/odir/resnet50.yaml --device_target GPU
```

### 2. 评估模型

仅计算准确率：
```bash
python scripts/validate.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --mode accuracy_only
```

生成完整评估报告（包括错误分类样本）：
```bash
python scripts/validate.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --mode full
```

指定设备：
```bash
python scripts/validate.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --device_target GPU
```

### 3. 数据清理

移动所有错误分类的图片：
```bash
python scripts/clean_data.py --action move_all
```

按比例保留错误分类的图片（例如保留20%）：
```bash
python scripts/clean_data.py --action move_percentage --keep_percentage 20
```

清理隔离区：
```bash
python scripts/clean_data.py --action clean_quarantine
```

预览操作（不实际执行）：
```bash
python scripts/clean_data.py --action move_all --dry_run
```

### 4. 单图预测

基本预测：
```bash
python scripts/predict.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --image_path path/to/image.jpg
```

指定设备：
```bash
python scripts/predict.py --config configs/odir/resnet50.yaml --ckpt_path checkpoints/resnet50-best.ckpt --image_path path/to/image.jpg --device_target GPU
```

## 项目结构

```
.
├── src/                    # 源代码
│   ├── data/              # 数据处理
│   │   ├── dataset.py     # 数据集加载
│   │   └── preprocessing.py # 数据预处理
│   ├── models/            # 模型定义
│   │   └── resnet.py      # ResNet模型实现
│   └── utils/             # 工具函数
│       ├── callbacks.py   # 训练回调
│       ├── logging.py     # 日志工具
│       └── metrics.py     # 评估指标
├── scripts/               # 可执行脚本
│   ├── train.py          # 训练脚本
│   ├── validate.py       # 验证脚本
│   ├── predict.py        # 预测脚本
│   └── clean_data.py     # 数据清理脚本
├── configs/               # 配置文件
│   └── odir/             # ODIR数据集配置
│       └── resnet50.yaml  # ResNet50模型配置
├── tests/                 # 测试代码
└── data/                  # 数据集
    └── odir4/            # ODIR数据集
        ├── train/        # 训练集
        ├── valid/        # 验证集
        └── misclassified/ # 错误分类样本
```

## 配置说明

主要配置参数（在 `configs/odir/resnet50.yaml` 中）：

- 模型设置：架构、预训练权重等
- 数据集设置：数据目录、批次大小等
- 训练设置：学习率、轮次等
- 数据增强设置
- 快速训练模式设置

## 性能指标

在验证集上的表现：
- Top-1 准确率：~85%
- Top-5 准确率：~95%

## 日志文件

训练过程中会生成以下日志文件：
- `logs/train_[timestamp].log`：训练日志
- `logs/loss_[timestamp].log`：损失记录
- `data/odir4/misclassified/misclassified_images.txt`：错误分类样本记录
- `data/odir4/misclassified/clean_misclassified.log`：数据清理日志

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。 