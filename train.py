from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    patience=10,
    cache=False
)