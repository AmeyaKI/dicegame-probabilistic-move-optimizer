import numpy as np
import cv2, torch
from ultralytics import YOLO # type: ignore
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import asyncio

# 12/5/2025 NOTE: Works but model needs more diversified dataset
 
# Basic Image/Model finding functions with try/except error detection 
async def load_model(weights_path):
    try:
        model = YOLO(weights_path)
    except FileNotFoundError as e:
        print(f"{e}: Model weights not found.")
    print("Loaded weights.")
    return model
    
async def find_image(image_dir):
    try:
        img = Image.open(image_dir)
    except FileNotFoundError as e:
        print(f"{e}: Custom image not found.")
        
    if img is None:
        raise ValueError(f"Unable to read image: {image_dir}")

    print("Image read.")
    return img



# Predict dice values from given image
async def predict_dice(model, img_path):
    img = await find_image(img_path)
    
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

    # drawn img, dice_scores_raw, final dice_values 
    return img, np.array(dice_scores_raw), dice_values



# Use webcam to detect images
def use_webcam(model):
    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_AVFOUNDATION)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = model(frame, save=False)
        
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            frame = cv2.rectangle(frame, [x1, y1, x2, y2], (0, 255, 0), 2)

        cv2.imshow("YOLO Dice Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()



async def activate_mode(mode):
    mode = mode.strip().lower()
    
    async def execute_mode(model, img_path):
        if mode == 'webcam':
            use_webcam(model)
        elif mode == 'image':
            image, dice_scores_raw, dice_values = await predict_dice(model, img_path)
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
    return execute_mode



async def main():
    weights = 'runs/yolo/dice_model_v2/weights/best.pt' # default model weights
    
    model = await load_model(weights)
    
    mode = input("Webcam or Image: ")
    
    mode_on = await activate_mode(mode)

    img_path = os.path.join(os.getcwd(), 'sample_v4.jpg')

    await mode_on(model, img_path)



if __name__ == '__main__':
    asyncio.run(main())