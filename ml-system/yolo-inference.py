from ultralytics import YOLO

model = YOLO('yolov8x')
# model = YOLO('trained-models/yolo5_last.pt')

res = model.track('input-data/input_video.mp4', conf=0.2, save=True)

"""
print(res)
print("boxes:")
for box in res[0].boxes:
    print(box)
"""