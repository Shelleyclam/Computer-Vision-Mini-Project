from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load your image
img_path = '/content/cars.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load pretrained YOLOv8 3D model
# The 'yolov8n.pt' model can do basic object detection; for 3D, you may use 'yolov8n3d.pt' if available
model = YOLO('yolov8n.pt')  # replace with yolov8n3d.pt for 3D if available

# Run detection
results = model.predict(source=img_path, imgsz=640)

# Plot results
results[0].plot()
plt.imshow(results[0].plot())
plt.axis('off')
plt.title("3D Detection / Bounding Box Visualization")
plt.show()
