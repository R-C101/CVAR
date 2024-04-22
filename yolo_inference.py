from ultralytics import YOLO

model = YOLO('models/best.pt')  

results = model.predict('input_videos/08fd33_4.mp4', save = True, save_dir = 'output_videos')
print(results[0])
for box in results[0].boxes:
    print(box)
