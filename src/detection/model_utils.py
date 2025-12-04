from ultralytics import YOLO

def load_model(weights_path: str = "models/yolov8_custom_detector/best.pt", device: str = "cpu"):
    """Load YOLOv8 model. Use 'cpu' or 'cuda'."""
    model = YOLO(weights_path)
    return model
