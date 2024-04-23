from ultralytics import YOLO

model = YOLO('yolov8x')

model.predict('input-data/image.png', save=True)