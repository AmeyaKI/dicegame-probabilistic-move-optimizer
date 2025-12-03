import torch
from ultralytics import YOLO # type: ignore
import os

model = YOLO("yolo11n.pt")

print("###### Model Loaded ######")

output_dir = os.path.join(os.getcwd(), "runs/yolo")
os.makedirs(output_dir, exist_ok=True)

device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

results = model.train(
    data="yolo_data/data.yaml",
    project=output_dir,
    name="dice_model_v1",
    exist_ok=True,

    epochs=50, # gotta get map50-95 and map50 to converge
    imgsz=640, # image size
    batch=16,
    lr0=0.001,
    
    workers=4,
    device=device,
    verbose=True,
)

print("\n###### Training Finished ######")
print(results)

print("\n###### Running Validation ######")
val_results = model.val()

print("\n###### Validation Metrics ######")
precision     = val_results.results_dict.get("metrics/precision(B)", None)
recall        = val_results.results_dict.get("metrics/recall(B)", None)
map50         = val_results.results_dict.get("metrics/mAP50(B)", None)
map5095       = val_results.results_dict.get("metrics/mAP50-95(B)", None)

print(f"Precision: {precision:.4f}" if precision else "Precision not found")
print(f"Recall:    {recall:.4f}" if recall else "Recall not found")
print(f"mAP50:     {map50:.4f}" if map50 else "mAP50 not found")
print(f"mAP50-95:  {map5095:.4f}" if map5095 else "mAP50-95 not found")


print("\n###### Accuracy per Class (Dice #) ######")
for cls_id, cls_name in enumerate(model.names):
    precision = val_results.box.pr[cls_id]
    recall = val_results.box.re[cls_id]
    print(f"Class {cls_id} ({cls_name}): Precision={precision:.3f}, Recall={recall:.3f}")

print("\nBest weights saved at:")
print("  runs/detect/dice_model_v1/weights/best.pt")