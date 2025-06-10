"""
ResNet model for Eye Disease Classification
"""

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from src.config.config import Config

class BasicBlock(nn.Cell):
    """ResNet基础块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, pad_mode='pad'),
                nn.BatchNorm2d(out_channels * self.expansion)
            ])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Cell):
    """ResNet瓶颈块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, pad_mode='pad')
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, pad_mode='pad'),
                nn.BatchNorm2d(out_channels * self.expansion)
            ])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Cell):
    """ResNet模型"""
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode='pad')
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class EyeDiseaseNet(nn.Cell):
    """
    基于ResNet50的眼疾分类模型
    """
    def __init__(self, num_classes=8):
        super(EyeDiseaseNet, self).__init__()
        
        # 创建ResNet50
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        
        # 添加SE注意力模块
        self.se = nn.SequentialCell([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, kernel_size=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, pad_mode='pad'),
            nn.Sigmoid()
        ])
        
    def construct(self, x):
        # 获取ResNet的特征
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # 应用SE注意力
        se_out = self.se(x)
        x = x * se_out
        
        # 全局平均池化
        x = self.backbone.avgpool(x)
        x = x.view(x.shape[0], -1)
        
        # 分类
        x = self.backbone.fc(x)
        return x

def get_model():
    """
    获取模型实例
    
    Returns:
        model: 模型实例
    """
    model = EyeDiseaseNet(num_classes=Config.NUM_CLASSES)
    return model