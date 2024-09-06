from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

if __name__ == '__main__':
    # Use the model
    results = model.train(data="config.yaml", epochs=100, workers=2)  # train the model

