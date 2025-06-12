"""
Main entry point for Eye Disease Classification
"""

import os
import argparse
import mindspore as ms
import numpy as np
from PIL import Image
from mindspore import context
from src.config.config import Config
from src.models.resnet import get_model
from src.train.trainer import train

def setup_context():
    """设置运行环境"""
    ms.set_device(Config.DEVICE_TARGET)
    context.set_context(mode=context.GRAPH_MODE)

def preprocess_image(image_path):
    """预处理单张图片
    
    Args:
        image_path: 图片路径
        
    Returns:
        预处理后的图像张量
    """
    # 加载并调整图像大小
    image = Image.open(image_path).convert('RGB')
    image = image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    # 转换为numpy数组
    image_array = np.array(image, dtype=np.float32)
    
    # 手动归一化 (HWC格式)
    mean = np.array(Config.MEAN).reshape(1, 1, 3)
    std = np.array(Config.STD).reshape(1, 1, 3)
    image_array = (image_array / 255.0 - mean) / std
    
    # 转换为CHW格式
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # 转换为张量并添加批次维度
    image_tensor = ms.Tensor(image_array, dtype=ms.float32)
    image_tensor = image_tensor.expand_dims(0)
    
    return image_tensor

def predict_image(image_path, model_path):
    """预测单张图片"""
    # 加载模型
    network = get_model()
    param_dict = ms.load_checkpoint(model_path)
    ms.load_param_into_net(network, param_dict)
    network.set_train(False)

    # 预处理图像
    image_tensor = preprocess_image(image_path)
    
    # 进行预测
    logits = network(image_tensor)
    predictions = ms.ops.Argmax(axis=1)(logits)
    probs = ms.ops.Softmax(axis=1)(logits)
    
    # 获取预测结果
    pred_class = predictions.asnumpy()[0]
    confidence = probs.asnumpy()[0][pred_class]
    
    return {
        'class': Config.CLASS_NAMES[pred_class],
        'confidence': float(confidence),
        'all_probs': probs.asnumpy()[0].tolist()
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Eye Disease Classification')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', type=str, help='Test image path or directory')
    parser.add_argument('--fast', action='store_true', help='Use fast training mode')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    args = parser.parse_args()

    # 打印项目信息
    print("=" * 70)
    print("Eye Disease Classification - Medical AI")
    print("=" * 70)
    print("Target: Achieve high accuracy for medical diagnosis")
    print("Framework: MindSpore")
    print("Model: ResNet50 with SE attention")
    print("=" * 70)
    print("Eye Disease Classes:")
    for i, name in enumerate(Config.CLASS_NAMES):
        print(f"  {i}: {name}")
    print("=" * 70)
    print()

    if args.train:
        # 训练模式
        if args.fast:
            print("Starting fast training...")
            Config.EPOCHS = 25  # 快速训练使用25轮
            Config.FAST_TRAINING = True  # 设置快速训练标志
        else:
            print("Starting full training...")
            Config.FAST_TRAINING = False
        train()
        return

    if args.test:
        # 测试模式
        setup_context()
        
        # 查找最新的检查点
        checkpoint_dir = Config.CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
            return
            
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith(Config.MODEL_PREFIX) and f.endswith('.ckpt')]
        if not checkpoints:
            print(f"Error: No checkpoints found in {checkpoint_dir}")
            return
            
        # 按修改时间排序
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        model_path = os.path.join(checkpoint_dir, checkpoints[-1])
        
        print(f"Using model: {model_path}")
        
        if os.path.isfile(args.test):
            # 测试单张图片
            print(f"\nTesting image: {args.test}")
            result = predict_image(args.test, model_path)
            
            print("\nPrediction Results:")
            print(f"Class: {result['class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            if args.verbose:
                print("\nDetailed Probabilities:")
                for i, prob in enumerate(result['all_probs']):
                    print(f"{Config.CLASS_NAMES[i]}: {prob:.2%}")
        
        elif os.path.isdir(args.test):
            # 测试整个目录
            print(f"\nTesting directory: {args.test}")
            total = 0
            correct = 0
            
            for root, _, files in os.walk(args.test):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        try:
                            result = predict_image(image_path, model_path)
                            true_class = os.path.basename(os.path.dirname(image_path))
                            
                            if args.verbose:
                                print(f"\nImage: {image_path}")
                                print(f"True class: {true_class}")
                                print(f"Predicted: {result['class']}")
                                print(f"Confidence: {result['confidence']:.2%}")
                            
                            total += 1
                            if result['class'] == true_class:
                                correct += 1
                                
                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")
            
            if total > 0:
                accuracy = correct / total
                print(f"\nOverall accuracy: {accuracy:.2%}")
                print(f"Correct: {correct}/{total}")
        else:
            print(f"Error: Invalid test path {args.test}")
        return

    print("Please specify --train or --test")

if __name__ == '__main__':
    main() 