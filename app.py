from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import base64
import torch
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

app = Flask(__name__)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 1, kernel_size=5)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        # self.fc1 = nn.Linear(1024, 128, bias=True)
        # self.fc2 = nn.Linear(128, 10, bias=True)
        self.fc1 = nn.Linear(512, 10, bias=True)

    def forward(self, x):
        # x = self.conv1(x)
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv3(x)))
        x = x.view(-1, 512)
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc1(x)
        return x

# 加载模型
model_path = 'weights/model_epoch_8.pth'  
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    # 转换为灰度图像
    grayscale_image = image.convert("L")
    
    # 反转颜色，使白色字体变为黑色，黑色背景变为白色
    inverted_image = ImageOps.invert(grayscale_image)
    
    # 归一化处理
    normalized_image = transform(inverted_image).unsqueeze(0)
    
    return normalized_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image'].split(',')[1]  # 去除data URL前缀
    image = Image.open(io.BytesIO(base64.b64decode(img_data)))

    # input_tensor = preprocess_image(image)
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    return jsonify({'prediction': int(predicted[0])})

if __name__ == '__main__':
    app.run(debug=True)