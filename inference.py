import numpy as np
import cv2
import torch
from ultralytics import YOLO # type: ignore
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 12/5/2025 NOTE: Works but model needs more diversified dataset
 
def load_model(weights_path):
    try:
        return YOLO(weights_path)
    except FileNotFoundError as e:
        print(f"{e}: Model weights not found.")
    
def find_custom_image(image_dir):
    try:
        img = Image.open(image_dir)
    except FileNotFoundError as e:
        print(f"{e}: Custom image not found.")
        
    if img is None:
        raise ValueError(f"Unable to read image: {image_dir}")
    
    return img


def predict_dice(model, img, img_path):
    # img = Image.open(img_path)
    result = model(img_path)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    
    drawn_img = ImageDraw.Draw(img)
    
    for x1, y1, x2, y2 in boxes:
        drawn_img.rectangle([x1, y1, x2, y2], outline='blue', width=2)

    dice_scores_raw = []
    for det in result.boxes:
        pred_class = int(det.cls.cpu().numpy()) # predicted class
        confidence = float(det.conf.cpu().item()) # predidction confidence
        bbox = det.xyxy.cpu().numpy()[0] # bounding box

        x_center = (bbox[0] + bbox[2]) / 2
        
        dice_scores_raw.append((pred_class, confidence, x_center))
        
    if dice_scores_raw is None:
        return np.array([])
        
    dice_scores_raw.sort(key=lambda x: x[2]) # sort dice by order in which they appear in image (left to right) thru x_center
    dice_values = np.array([d[0] for d in dice_scores_raw], dtype=np.int64) + 1 # get dice pred_class values

    # Return: drawn img, dice_scores_raw, final dice_values 
    return img, np.array(dice_scores_raw), dice_values

def main():
    weights = 'runs/yolo/dice_model_v2/weights/best.pt' # input("Enter path to weights: "")
    custom_image_path = os.path.join(os.getcwd(), 'sample_v4.jpg') # input("Enter path to custom image: ")
    
    model = load_model(weights)
    
    print("Loaded weights.")
    
    img = find_custom_image(custom_image_path)
    print("Image read.")
    
    image, dice_scores_raw, dice_values = predict_dice(model, img, custom_image_path)
    print("Calculated dice values.")
        

    print(f"Holistic dice scores:\n{dice_scores_raw}")
    
    if dice_values.size == 0:
        print("No dice detected.")
    else:
        print(f"Detected dice values: {dice_values}")
        
    print("Displaying bounding boxes")

    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()