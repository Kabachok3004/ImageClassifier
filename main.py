import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300000000
import torch.nn.functional as F
import sys
from models import CNN, GazeNet

    
colors = np.array([[0,44,58],[0,76,137],[0,104,169],[60,119,174],[91,155,204],[131,152,174],[147,172,191],[212,215,219],[217,215,216]])

def process_image(image_path: str, dif_range: int, colors: list, сompression: int):
    try:
        image = Image.open(image_path)
        image = image.resize((сompression,int(image.height/image.width*сompression)))
    except:
        return 1
    pixels = np.array(image).reshape(-1, 3)
    black = np.array([0, 0, 0])
    no_black_pixels=[]
    for i in pixels:
        if np.linalg.norm(i - black) > 30:
            no_black_pixels.append(i)
    
    filtered_pixels = []
    for pixel in no_black_pixels:
        matches = [np.linalg.norm(pixel - color) < dif_range for color in colors]
        if any(matches):
            filtered_pixels.append(pixel)
    
    if len(no_black_pixels) > 0:
        return len(filtered_pixels) / len(no_black_pixels)
    else:
        return 0
    
def color_filter(image_path: str):
    return process_image(image_path, 65, colors, 150)


class ModelAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def analyze_image(self, image_path: str):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        except:
            return 0.6
        with torch.no_grad():
            output = self.model(img_tensor)
        
        return output.tolist()[0][0]
        


class ModelEyesAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GazeNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def filter_image(self, image_path: str):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        except:
            return 0.
        with torch.no_grad():
            output = self.model(img_tensor)
        return output.tolist()[0][0]

eyes_model = ModelEyesAnalyzer("model_eyes.pth")
model = ModelAnalyzer("model_teDo_brand_compliance2.pth")

def load_images(image_dir):
    images = []
    labels = []
    for category in ['Positive', 'Negative']:
        class_label = 1 if category == 'Positive' else 0
        dir_path = os.path.join(image_dir, category)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            images.append(img_path)
            labels.append(class_label)
    return images, labels

def predict(images,filter1,filter2,model):
    prediction = []

    for i in images:
        fl1 = ((filter1(i)-0.3)**4)
        fl2 = (-(filter2(i) + 0.3)**4 + 1)/3 if filter2(i) < 0.7 else 0
        
    
        prediction.append(x*(1-fl1)*(1-fl2))
        
        print("pred", type(x), x)
        
    return prediction
    
images, labels = load_images(image_dir="dataset")
images, labels = images[500:700], labels[500:700]
# prediction = predict(images,eyes_model.filter_image, color_filter, model.analyze_image)
# binary_prediction = [1 if pred > 0.5 else 0 for pred in prediction]
 
 


def load_test(dir_path):
    images = []
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        images.append(img_path)
    return images

def save_predictions_to_file(predictions, images, filename):
    with open(filename, 'w') as file:
        for img, pred in zip(images, predictions):
            img_name = os.path.basename(img)
            file.write(f"{img_name};{pred}\n")

#   фактическое предсказание

images = load_test("test")
images, labels = load_images(image_dir="dataset")
images, labels = images[500:700], labels[500:700]

prediction = predict(images,eyes_model.filter_image, color_filter, model.analyze_image)

save_predictions_to_file(prediction, images, "predictions.txt")

binary_prediction = [1 if pred > 0.52 else 0 for pred in prediction]
print(prediction, len(prediction))
print("accuracy_score", accuracy_score(labels, binary_prediction))
print("roc_auc_score", roc_auc_score(labels, prediction))
