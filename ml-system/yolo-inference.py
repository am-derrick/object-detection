from ultralytics import YOLO

# model = YOLO('yolov8x')
model = YOLO('training/runs/detect/train3/weights/last.pt')

res = model.predict('input-data/input_video.mp4', conf=0.2, save=True)

print(res)
print("boxes:")
for box in res[0].boxes:
    print(box)