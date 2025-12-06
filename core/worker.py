from PySide6.QtCore import QThread, Signal
from core.yolo_engine import YOLOManager
from core.dataset_utils import create_data_yaml
import traceback
import sys
import io

class TrainingWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.manager = YOLOManager()

    def run(self):
        try:
            self.log_signal.emit("Preparing dataset...")
            # Generate data.yaml
            data_yaml_path = "data.yaml" # Temp path or inside project
            create_data_yaml(
                self.config['train_images'],
                self.config['val_images'],
                self.config['classes'],
                data_yaml_path
            )
            self.config['data_yaml'] = data_yaml_path
            
            self.log_signal.emit(f"Data config created at {data_yaml_path}")
            
            # Start Training
            self.manager.train(
                self.config, 
                progress_callback=lambda p: self.progress_signal.emit(p),
                log_callback=lambda msg: self.log_signal.emit(msg)
            )
            
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
            self.log_signal.emit(f"Error: {traceback.format_exc()}")

class InferenceWorker(QThread):
    results_signal = Signal(list)
    error_signal = Signal(str)

    def __init__(self, model_path, image_folder, use_gray=False):
        super().__init__()
        self.model_path = model_path
        self.image_folder = image_folder
        self.use_gray = use_gray
        self.manager = YOLOManager()

    def run(self):
        try:
            results = self.manager.predict(self.model_path, self.image_folder, self.use_gray)
            self.results_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))
