import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit, QGroupBox, QFormLayout, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from core.dataset_utils import split_dataset

class DatasetWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, source, output, ratio):
        super().__init__()
        self.source = source
        self.output = output
        self.ratio = ratio

    def run(self):
        try:
            split_dataset(
                self.source, 
                self.output, 
                self.ratio, 
                lambda msg: self.log_signal.emit(msg)
            )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Configuration
        config_group = QGroupBox("資料集製作設定")
        config_layout = QFormLayout()

        self.source_edit = self.create_file_selector(config_layout, "原始資料夾 (含圖片與txt):")
        self.output_edit = self.create_file_selector(config_layout, "輸出位置:", is_save=False)
        
        self.dataset_name_edit = QLineEdit("MyDataset")
        config_layout.addRow("資料集名稱:", self.dataset_name_edit)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.1, 0.9)
        self.ratio_spin.setSingleStep(0.1)
        self.ratio_spin.setValue(0.8)
        self.ratio_spin.setPrefix("訓練集比例: ")
        config_layout.addRow("分割比例:", self.ratio_spin)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Actions
        self.convert_btn = QPushButton("開始轉換")
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.clicked.connect(self.on_convert_clicked)
        layout.addWidget(self.convert_btn)

        # Logs
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def create_file_selector(self, layout, label_text, is_save=False):
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        line_edit = QLineEdit()
        btn = QPushButton("瀏覽")
        btn.clicked.connect(lambda: self.browse_folder(line_edit))
        
        h_layout.addWidget(line_edit)
        h_layout.addWidget(btn)
        
        layout.addRow(label_text, container)
        return line_edit

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            line_edit.setText(folder)

    def on_convert_clicked(self):
        source = self.source_edit.text()
        base_output = self.output_edit.text()
        name = self.dataset_name_edit.text()
        ratio = self.ratio_spin.value()

        if not source or not base_output or not name:
            QMessageBox.warning(self, "警告", "請填寫所有欄位")
            return

        final_output = os.path.join(base_output, name)
        
        self.log_output.append(f"開始轉換... 目標: {final_output}")
        self.convert_btn.setEnabled(False)

        self.worker = DatasetWorker(source, final_output, ratio)
        self.worker.log_signal.connect(self.log_output.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        self.convert_btn.setEnabled(True)
        QMessageBox.information(self, "成功", "資料集轉換完成！")
        self.log_output.append("完成。")

    def on_error(self, err):
        self.convert_btn.setEnabled(True)
        QMessageBox.critical(self, "錯誤", f"轉換失敗: {err}")
        self.log_output.append(f"錯誤: {err}")
