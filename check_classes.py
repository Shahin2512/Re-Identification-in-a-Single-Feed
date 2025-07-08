from ultralytics import YOLO

model = YOLO("best.pt")
print("🔍 Model Class Names:", model.names)
