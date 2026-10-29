from ultralytics import YOLO

model = YOLO("runs/detect/train-3/weights/best.pt")

model.predict(
    source="dataset/images/val",
    conf=0.25,
    save=True
)