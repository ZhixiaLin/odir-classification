"""
Prediction module for Eye Disease Classification
"""

import os
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from PIL import Image
import mindspore.dataset.vision.c_transforms as CV
from src.models.resnet import get_model
from src.config.config import MODEL_CONFIG, CLASS_MAPPING, DATASET_CONFIG

def preprocess_image(image_path):
    """
    预处理单张图片
    
    Args:
        image_path: 图片路径
    
    Returns:
        tensor: 预处理后的图片张量
    """
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    transform = [
        CV.Resize(DATASET_CONFIG['image_size']),
        CV.CenterCrop(DATASET_CONFIG['image_size']),
        CV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        CV.HWC2CHW()
    ]
    
    # 应用转换
    for t in transform:
        image = t(image)
    
    # 转换为张量
    image = ms.Tensor(image, ms.float32)
    image = image.expand_dims(axis=0)  # 添加批次维度
    
    return image

def predict_image(image_path, model_path):
    """
    预测单张图片
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
    
    Returns:
        dict: 预测结果，包含类别和概率
    """
    # 加载模型
    network = get_model()
    param_dict = load_checkpoint(model_path)
    load_param_into_net(network, param_dict)
    
    # 预处理图片
    image = preprocess_image(image_path)
    
    # 预测
    network.set_train(False)
    output = network(image)
    probabilities = ms.ops.Softmax()(output)
    
    # 获取预测结果
    pred_class = int(probabilities.argmax(axis=1).asnumpy()[0])
    pred_prob = float(probabilities.max(axis=1).asnumpy()[0])
    
    return {
        'class': CLASS_MAPPING[pred_class],
        'probability': pred_prob
    }

def main():
    """
    主函数，用于测试预测功能
    """
    # 测试图片路径
    test_image = "data/test.jpg"  # 请替换为实际的测试图片路径
    model_path = os.path.join("checkpoints", "best_model.ckpt")
    
    if not os.path.exists(test_image):
        print(f"Error: Test image not found at {test_image}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # 预测
    result = predict_image(test_image, model_path)
    
    # 打印结果
    print("\nPrediction Result:")
    print(f"Class: {result['class']}")
    print(f"Probability: {result['probability']:.4f}")

if __name__ == "__main__":
    main() 