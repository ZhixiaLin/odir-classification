"""
ResNet model for Eye Disease Classification - Lightweight Version
"""

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import HeNormal, Normal
from mindspore import load_checkpoint, load_param_into_net
from src.config.config import Config

class BasicBlock(nn.Cell):
    """Basic block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode='pad',
            weight_init=HeNormal(mode='fan_out')
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode='pad',
            weight_init=HeNormal(mode='fan_out')
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.downsample = downsample
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out

class EnhancedResNet(nn.Cell):
    """轻量化ResNet模型 - 专为小数据集优化"""
    def __init__(self, num_classes=8):
        super(EnhancedResNet, self).__init__()
        
        # 减少初始通道数 (从64->32)
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=7, stride=2, padding=3, pad_mode='pad',
            weight_init=HeNormal(mode='fan_out')
        )
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        
        # 大幅减少层数和通道数
        self.in_channels = 32
        self.layer1 = self._make_layer(BasicBlock, 32, 2)  # 32->32
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)  # 32->64
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 64->128
        # 移除layer4，减少复杂度
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 进一步简化分类头，减少过拟合
        self.classifier = nn.SequentialCell([
            nn.Dense(128, 32, weight_init=Normal(sigma=0.02)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Dense(32, num_classes, weight_init=Normal(sigma=0.02))
        ])

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    weight_init=HeNormal(mode='fan_out')
                ),
                nn.BatchNorm2d(out_channels * block.expansion, momentum=0.9)
            ])

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
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
        
        x = self.avgpool(x)
        x = P.Flatten()(x)
        x = self.classifier(x)
        
        return x

def get_model(num_classes=None):
    """获取模型实例"""
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    
    # 使用轻量化的ResNet模型
    model = EnhancedResNet(num_classes=num_classes)
    print("🎯 使用轻量化ResNet，专为小数据集优化")
    
    return model