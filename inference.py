import numpy as np
import cv2
import torch
from ultralytics import YOLO # type: ignore
import os, sys

#### RUN CODE LATER _ VERIFY ###

def load_model(weights_path):
    try:
        return YOLO(weights_path)
    except FileNotFoundError as e:
        print(f"{e}: Model weights not found.")
    
def custom_image(image_dir):
    try:
        img = cv2.imread(image_dir)
    except FileNotFoundError as e:
        print(f"{e}: Custom image not found.")
        
    if img is None:
        raise ValueError(f"Unable to read image: {image_dir}")
    
    return img

def predict_dice_values(model, img):
    results = model(img)[0]
    
    dice_values = []
    
    for det in results.boxes:
        pred_class = int(det.cls.cpu().numpy()) # predicted class
        confidence = float(det.conf.cpu().numpy()) # predidction confidence
        bbox = det.xyxy.cpu().numpy()[0] # bounding box

        x_center = (bbox[0] + bbox[2]) / 2
        
        dice_values.append((pred_class, confidence, x_center))
    
    if dice_values is None:
        return np.array([])
        
    dice_values.sort(key=lambda x: x[2])
    dice_values_final = [d[0] for d in dice_values]

    return np.array(dice_values_final, dtype=np.int32)

def main():
    ...
    weights = sys.argv[1]
    image_path = sys.argv[2]
    
    model = load_model(weights)
    img = custom_image(image_path)
    
    dice_values = predict_dice_values(model, img)

    if dice_values.size == 0:
        print("No dice detected.")
    else:
        print(f"Detected dice values: {dice_values}")

if __name__ == '__main__':
    main()