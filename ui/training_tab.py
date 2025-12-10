import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal

class TrainingTab(QWidget):
    train_requested = Signal(dict)  # Signal to send configuration to backend

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Project Configuration
        config_group = QGroupBox("專案設定")
        config_layout = QFormLayout()

        self.project_name_edit = QLineEdit("MyYOLOProject")
        self.model_name_edit = QLineEdit("yolov8n")
        self.version_combo = QComboBox()
        self.version_combo.addItems(["YOLOv8", "YOLOv11", "YOLOv5"])
        
        config_layout.addRow("專案名稱:", self.project_name_edit)
        config_layout.addRow("模型名稱:", self.model_name_edit)
        config_layout.addRow("YOLO 版本:", self.version_combo)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Dataset Selection
        dataset_group = QGroupBox("資料集選擇")
        dataset_layout = QFormLayout()

        self.train_images_edit = self.create_file_selector(dataset_layout, "訓練圖片路徑:")
        self.train_labels_edit = self.create_file_selector(dataset_layout, "訓練標籤路徑:")
        self.val_images_edit = self.create_file_selector(dataset_layout, "驗證圖片路徑 (選填):")
        self.val_labels_edit = self.create_file_selector(dataset_layout, "驗證標籤路徑 (選填):")
        
        # Class Names
        self.class_names_edit = QLineEdit()
        self.class_names_edit.setPlaceholderText("cat, dog, person (請用逗號分隔)")
        dataset_layout.addRow("類別名稱:", self.class_names_edit)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # Hyperparameters
        param_group = QGroupBox("超參數設定")
        param_layout = QFormLayout() # Changed to FormLayout for better alignment of many items

        # Row 1: Epochs, Batch, ImgSize
        row1_layout = QHBoxLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setPrefix("Epochs: ")
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.batch_spin.setPrefix("Batch: ")

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setPrefix("ImgSz: ")

        row1_layout.addWidget(self.epochs_spin)
        row1_layout.addWidget(self.batch_spin)
        row1_layout.addWidget(self.imgsz_spin)
        param_layout.addRow("基本設定:", row1_layout)

        # Row 2: Advanced Settings
        row2_layout = QHBoxLayout()

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "GPU (CUDA)", "GPU (MPS)"])
        self.device_combo.setToolTip("選擇訓練裝置")
        
        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(8)
        self.workers_spin.setPrefix("Workers: ")
        self.workers_spin.setToolTip("資料載入執行緒數量")

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        self.optimizer_combo.setToolTip("優化器選擇")

        # Patience
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(50)
        self.patience_spin.setPrefix("Patience: ")
        self.patience_spin.setToolTip("Early Stopping 耐心值")

        row2_layout.addWidget(QLabel("Device:"))
        row2_layout.addWidget(self.device_combo)
        row2_layout.addWidget(self.workers_spin)
        row2_layout.addWidget(QLabel("Opt:"))
        row2_layout.addWidget(self.optimizer_combo)
        row2_layout.addWidget(self.patience_spin)
        
        param_layout.addRow("進階設定:", row2_layout)

        # Row 3: Optimization Details
        row3_layout = QHBoxLayout()

        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0.0001, 0.1)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setDecimals(4)
        self.lr0_spin.setValue(0.01)
        self.lr0_spin.setPrefix("LR0: ")
        self.lr0_spin.setToolTip("初始學習率 (Initial Learning Rate)")

        self.cos_lr_check = QCheckBox("Cosine LR")
        self.cos_lr_check.setToolTip("使用餘弦退火調整學習率")
        
        self.rect_check = QCheckBox("Rect")
        self.rect_check.setToolTip("矩形訓練 (Rectangular Training)")
        
        self.cache_check = QCheckBox("Cache")
        self.cache_check.setToolTip("快取圖片至 RAM (Cache Images)")

        row3_layout.addWidget(self.lr0_spin)
        row3_layout.addWidget(self.cos_lr_check)
        row3_layout.addWidget(self.rect_check)
        row3_layout.addWidget(self.cache_check)
        
        param_layout.addRow("優化參數:", row3_layout)

        # Row 4: Augmentation
        row4_layout = QHBoxLayout()

        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(-180.0, 180.0)
        self.degrees_spin.setValue(0.0)
        self.degrees_spin.setPrefix("旋轉: ")
        self.degrees_spin.setSuffix("°")
        self.degrees_spin.setToolTip("隨機旋轉角度 (+/- degrees)")

        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.1)
        self.fliplr_spin.setValue(0.5)
        self.fliplr_spin.setPrefix("左右翻轉: ")
        self.fliplr_spin.setToolTip("隨機左右翻轉機率 (Flip Left-Right)")

        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setSingleStep(0.1)
        self.mosaic_spin.setValue(1.0)
        self.mosaic_spin.setPrefix("Mosaic: ")
        self.mosaic_spin.setToolTip("馬賽克增強機率")

        row4_layout.addWidget(self.degrees_spin)
        row4_layout.addWidget(self.fliplr_spin)
        row4_layout.addWidget(self.mosaic_spin)

        param_layout.addRow("資料增強:", row4_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Controls & Logs
        self.train_btn = QPushButton("開始訓練")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self.on_train_clicked)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        layout.addWidget(self.train_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_output)

    def create_file_selector(self, layout, label_text):
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

    def on_train_clicked(self):
        config = {
            "project_name": self.project_name_edit.text(),
            "model_name": self.model_name_edit.text(),
            "version": self.version_combo.currentText(),
            "train_images": self.train_images_edit.text(),
            "train_labels": self.train_labels_edit.text(),
            "val_images": self.val_images_edit.text(),
            "val_labels": self.val_labels_edit.text(),
            "classes": self.class_names_edit.text(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_combo.currentText(),
            "workers": self.workers_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "patience": self.patience_spin.value(),
            "lr0": self.lr0_spin.value(),
            "cos_lr": self.cos_lr_check.isChecked(),
            "rect": self.rect_check.isChecked(),
            "cache": self.cache_check.isChecked(),
            "degrees": self.degrees_spin.value(),
            "fliplr": self.fliplr_spin.value(),
            "mosaic": self.mosaic_spin.value()
        }
        self.train_requested.emit(config)
        self.log_output.append("請求訓練中...")
        self.train_btn.setEnabled(False)

    def append_log(self, message):
        self.log_output.append(message)
        # Auto scroll
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def training_finished(self):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.append_log("訓練完成！")

    def set_dataset_paths(self, root_path):
        """Auto-fill dataset paths based on standard structure"""
        import os
        self.train_images_edit.setText(os.path.join(root_path, "images", "train"))
        self.train_labels_edit.setText(os.path.join(root_path, "labels", "train"))
        self.val_images_edit.setText(os.path.join(root_path, "images", "val"))
        self.val_labels_edit.setText(os.path.join(root_path, "labels", "val"))
        self.append_log(f"已自動帶入資料集路徑: {root_path}")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
