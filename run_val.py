from ultralytics import YOLO

print('Starting YOLO validation (this may take several minutes)...')
model = YOLO('best.pt')
results = model.val(data='coco128.yaml')
print('Validation finished.')
print(results)
