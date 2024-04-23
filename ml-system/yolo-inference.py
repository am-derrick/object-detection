from ultralytics import YOLO

model = YOLO('yolov8x')

model.predict('input-data/image2.jpeg', save=True)