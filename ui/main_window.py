from PySide6.QtWidgets import QMainWindow, QTabWidget, QMessageBox
from ui.training_tab import TrainingTab
from ui.inference_tab import InferenceTab
from core.worker import TrainingWorker, InferenceWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 無程式碼訓練平台")
        self.resize(1000, 700)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.training_tab = TrainingTab()
        self.inference_tab = InferenceTab()

        self.tabs.addTab(self.training_tab, "訓練")
        self.tabs.addTab(self.inference_tab, "推論")

        # Connect Signals
        self.training_tab.train_requested.connect(self.start_training)
        self.inference_tab.inference_requested.connect(self.start_inference)

        self.train_worker = None
        self.inf_worker = None

    def start_training(self, config):
        if self.train_worker and self.train_worker.isRunning():
            QMessageBox.warning(self, "忙碌中", "訓練正在進行中。")
            return

        self.train_worker = TrainingWorker(config)
        self.train_worker.log_signal.connect(self.training_tab.append_log)
        self.train_worker.finished_signal.connect(self.on_training_finished)
        self.train_worker.error_signal.connect(self.on_training_error)
        
        self.train_worker.start()

    def on_training_finished(self):
        self.training_tab.training_finished()
        QMessageBox.information(self, "成功", "訓練成功完成！")

    def on_training_error(self, err_msg):
        self.training_tab.append_log(f"錯誤: {err_msg}")
        self.training_tab.train_btn.setEnabled(True) # Re-enable button
        QMessageBox.critical(self, "錯誤", f"訓練失敗: {err_msg}")

    def start_inference(self, model_path, image_folder):
        if self.inf_worker and self.inf_worker.isRunning():
            return

        self.inf_worker = InferenceWorker(model_path, image_folder)
        self.inf_worker.results_signal.connect(self.inference_tab.update_results)
        self.inf_worker.error_signal.connect(self.on_inference_error)
        
        self.inf_worker.start()

    def on_inference_error(self, err_msg):
        QMessageBox.critical(self, "錯誤", f"推論失敗: {err_msg}")
        self.inference_tab.run_btn.setEnabled(True)
