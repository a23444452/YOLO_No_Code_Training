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
        
        # Advanced hyperparameters
        device_str = config.get('device', 'Auto')
        workers = config.get('workers', 8)
        optimizer = config.get('optimizer', 'auto')
        patience = config.get('patience', 50)

        # Map device string to YOLO format
        device = None
        if device_str == 'CPU':
            device = 'cpu'
        elif device_str == 'GPU (CUDA)':
            device = '0'
        elif device_str == 'GPU (MPS)':
            device = 'mps'
        # 'Auto' remains None

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

        # Attach progress callback
        if progress_callback:
            def on_train_epoch_end(trainer):
                current_epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                progress = int((current_epoch / total_epochs) * 100)
                progress_callback(progress)
            
            self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

        if log_callback:
            log_callback(f"Starting training for {epochs} epochs...")
            log_callback(f"Device: {device_str}, Workers: {workers}, Opt: {optimizer}, Patience: {patience}")

        # Train
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            optimizer=optimizer,
            patience=patience,
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

    def predict(self, model_path, image_folder, use_gray=False):
        """
        Run inference on a folder of images.
        """
        model = YOLO(model_path)
        
        # Get list of images
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if os.path.splitext(f)[1].lower() in valid_exts]
        
        results_data = []
        
        import cv2
        import numpy as np

        for img_path in images:
            # Prepare source
            if use_gray:
                # Read as gray, convert to BGR (YOLO expects 3 channels)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                source = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                source = img_path

            # Run prediction
            # stream=False for single image processing loop to handle custom source
            results = model.predict(source, verbose=False)
            
            for res in results:
                # Process result
                boxes = res.boxes
                
                detections = []
                for box in boxes:
                    # x1, y1, x2, y2
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    # Handle ONNX model names (sometimes missing or dict)
                    if hasattr(model, 'names') and model.names:
                        cls_name = model.names[cls_id]
                    else:
                        cls_name = str(cls_id)
                    
                    detections.append(coords + [conf, cls_name])
                
                results_data.append({
                    'image_path': img_path, # Keep original path for display
                    'detections': detections
                })
            
        return results_data
