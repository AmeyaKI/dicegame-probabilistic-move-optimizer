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
    
def custom_image_detector(image_dir):
    try:
        img = cv2.imread(image_dir)
    except FileNotFoundError as e:
        print(f"{e}: Custom image not found.")
        
    if img is None:
        raise ValueError(f"Unable to read image: {image_dir}")
    
    return img

def predict_dice_values(model, img):
    results = model(img)[0]
    
    dice_scores = []
    
    for det in results.boxes:
        pred_class = int(det.cls.cpu().numpy()) # predicted class
        confidence = float(det.conf.cpu().item()) # predidction confidence
        bbox = det.xyxy.cpu().numpy()[0] # bounding box

        x_center = (bbox[0] + bbox[2]) / 2
        
        dice_scores.append((pred_class, confidence, x_center))
    
    if dice_scores is None:
        return np.array([])
        
    dice_scores.sort(key=lambda x: x[2]) # sort dice by order in which they appear in image (left to right) thru x_center
    dice_values_final = [d[0] for d in dice_scores] # get dice pred_class values

    return np.array(dice_scores), np.array(dice_values_final, dtype=np.int64)


def main():
    weights = 'runs/yolo/dice_model_v1/weights/best.pt' # input("Enter path to weights: "")
    custom_image_path = os.path.join(os.getcwd(), 'sample_v4.jpg') # input("Enter path to custom image: ")
    
    model = load_model(weights)
    
    print("Loaded weights.")
    
    img = custom_image_detector(custom_image_path)
    
    print("Read image.")
    
    dice_scores, dice_values = predict_dice_values(model, img)
    
    print("Calculated dice values.")

    print(f"Holistic dice scores: {dice_scores}")
    
    if dice_values.size == 0:
        print("No dice detected.")
    else:
        print(f"Detected dice values: {dice_values}")

if __name__ == '__main__':
    main()