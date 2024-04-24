from ultralytics import YOLO

model = YOLO('yolov8x')

res = model.predict('input-data/input_video.mp4', save=True)

print(res)
print("boxes:")
for box in res[0].boxes:
    print(box)