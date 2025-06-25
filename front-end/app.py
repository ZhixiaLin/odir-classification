import os
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from torchvision import transforms
from PIL import Image
from io import BytesIO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from double_nets import SE_ResNext50
from nets import SE_ResNext50_Mono
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load double-eye model
double_eye_model = SE_ResNext50(num_classes=8)
double_eye_checkpoint = torch.load("./runs3_bi5/best_precision_model_7_0.8187.pth")
double_eye_model.load_state_dict(double_eye_checkpoint['model_state_dict'])
double_eye_model.to(device)
double_eye_model.eval()

# Load single-eye model
single_eye_model = SE_ResNext50_Mono(num_classes=8)
single_eye_checkpoint = torch.load("./runs1/resnet_58_0.726027397260274.pth")
single_eye_model.load_state_dict(single_eye_checkpoint['model_state_dict'])
single_eye_model.to(device)
single_eye_model.eval()
def remove_black_background(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_image = image.crop((x, y, x + w, y + h))  # Using PIL Image.crop for cropping
        return cropped_image
    else:
        return image

# Update data_transform to include background removal
def updated_data_transform(image):
    # Remove black background first
    image = remove_black_background(image)
    
    # Apply the standard transformations
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform_pipeline(image)

# Use the updated transformation
def predict(left_img, right_img):
    # Apply the updated data transform
    left_tensor = updated_data_transform(left_img).unsqueeze(0).to(device)
    right_tensor = updated_data_transform(right_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = double_eye_model(left_tensor, right_tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten()

    # 类别名称
    class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    # 通过0.5的阈值筛选预测类别
    predictions = []
    for idx, prob in enumerate(probs):
        if prob > 0.5:
            predictions.append({
                'class': class_names[idx],
                'probability': float(prob)  # 转换为 Python 原生 float 类型
            })
    
    return predictions

# Grad-CAM visualization
def generate_gradcam(image, model, target_layers):
    image_tensor = updated_data_transform(image).unsqueeze(0).to(device)
    # 确保target_layers是列表形式
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    # 其余代码不变

    # Convert the image to numpy array for visualization
    image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # Convert the visualized image back to an OpenCV-friendly format (BGR)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    return visualization_bgr

@app.route('/test')
def test():
    return "test"

@app.route('/predict', methods=['POST'])  #返回的是识别结果，包括类别和置信度  对应批量识别
def predict_endpoint():
    print("start predict")
    if 'left_eye' not in request.files or 'right_eye' not in request.files:
        return jsonify({'error': 'Both left_eye and right_eye images are required'}), 400

    left_img = Image.open(request.files['left_eye']).convert('RGB')
    right_img = Image.open(request.files['right_eye']).convert('RGB')

    predictions = predict(left_img, right_img)
    print("predictions:", predictions)
    print("finish predict")
    return jsonify({'predictions': predictions})
@app.route('/visualize', methods=['POST'])    #对应批量可视化，返回的是图片
def visualize_endpoint():
    print("start visualize")
    
    if 'eye' not in request.files:
        return jsonify({'error': 'Eye image is required'}), 400

    # Open the image
    eye_img = Image.open(request.files['eye']).convert('RGB')
    
    # Generate Grad-CAM for the image
    vis_image = generate_gradcam(eye_img, single_eye_model, [single_eye_model.backbone.layer4])
    
    # Convert the image to a byte array to send over HTTP
    _, buffer = cv2.imencode('.png', vis_image)
    return send_file(BytesIO(buffer), mimetype='image/png')




@app.route('/preprocess', methods=['POST'])    #对单张图片进行预处理，返回的是预处理后的图片
def preprocess_endpoint():
    print("start preprocess")

    if 'eye' not in request.files:
        return jsonify({'error': 'Eye image is required'}), 400

    # 打开图片
    eye_img = Image.open(request.files['eye']).convert('RGB')

    # 经过预处理
    transformed_tensor = updated_data_transform(eye_img)
    
    # 反归一化处理，使图片可视化
    
    # 转换为PIL图片
    transformed_image = transforms.ToPILImage()(transformed_tensor)

    # 转换为字节流返回
    img_io = BytesIO()
    transformed_image.save(img_io, 'PNG')
    img_io.seek(0)

    print("finish preprocess")
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)