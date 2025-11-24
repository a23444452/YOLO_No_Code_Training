from ultralytics import YOLO
import os
import shutil

class YOLOManager:
    def __init__(self):
        self.model = None

    def train(self, config, progress_callback=None, log_callback=None):
        """
        Train the model.
        config: dict with keys: project_name, model_name, version, train_images, ...
        """
        project_name = config.get('project_name', 'yolo_project')
        model_name = config.get('model_name', 'my_model')
        version = config.get('version', 'YOLOv8')
        epochs = config.get('epochs', 10)
        batch = config.get('batch', 16)
        imgsz = config.get('imgsz', 640)
        data_yaml = config.get('data_yaml')

        # Determine base model
        if version == 'YOLOv8':
            base_model = 'yolov8n.pt'
        elif version == 'YOLOv11':
            base_model = 'yolo11n.pt'
        elif version == 'YOLOv5':
            base_model = 'yolov5nu.pt' # Ultralytics supports v5 models
        else:
            base_model = 'yolov8n.pt'

        if log_callback:
            log_callback(f"Initializing {version} model: {base_model}...")
        
        self.model = YOLO(base_model)

        if log_callback:
            log_callback(f"Starting training for {epochs} epochs...")

        # Train
        # Note: Ultralytics train() is blocking. We rely on the worker thread to keep UI alive.
        # We can't easily hook into real-time progress without custom callbacks or parsing stdout.
        # For simplicity, we just run it.
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project_name,
            name=model_name,
            exist_ok=True, # Overwrite existing project/name
            verbose=True
        )

        if log_callback:
            log_callback("Training finished.")
            log_callback(f"Results saved to {results.save_dir}")

        # Export to ONNX
        if log_callback:
            log_callback("Exporting to ONNX...")
        self.model.export(format='onnx')
        
        return results.save_dir

    def predict(self, model_path, image_folder):
        """
        Run inference on a folder of images.
        """
        model = YOLO(model_path)
        
        # Get list of images
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if os.path.splitext(f)[1].lower() in valid_exts]
        
        results_data = []
        
        # Run prediction
        # stream=True returns a generator, good for memory
        results = model.predict(images, stream=True)
        
        for res in results:
            # Process result
            path = res.path
            boxes = res.boxes
            
            detections = []
            for box in boxes:
                # x1, y1, x2, y2
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                detections.append(coords + [conf, cls_name])
            
            results_data.append({
                'image_path': path,
                'detections': detections
            })
            
        return results_data
