import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QListWidget, QSplitter, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

class InferenceTab(QWidget):
    inference_requested = Signal(str, str) # model_path, image_folder

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_results = {} # Store results: {filename: {image: path, detections: []}}

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Configuration
        config_group = QGroupBox("Inference Configuration")
        config_layout = QVBoxLayout()

        # Model Selection
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to .pt or .onnx model")
        model_btn = QPushButton("Select Model")
        model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_btn)
        config_layout.addLayout(model_layout)

        # Image Folder Selection
        folder_layout = QHBoxLayout()
        self.image_folder_edit = QLineEdit()
        self.image_folder_edit.setPlaceholderText("Path to folder containing images")
        folder_btn = QPushButton("Select Images")
        folder_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(QLabel("Images:"))
        folder_layout.addWidget(self.image_folder_edit)
        folder_layout.addWidget(folder_btn)
        config_layout.addLayout(folder_layout)

        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self.on_run_clicked)
        config_layout.addWidget(self.run_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Results Viewer
        splitter = QSplitter(Qt.Horizontal)

        # File List
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        splitter.addWidget(self.file_list)

        # Image Viewer
        self.image_label = QLabel("Select an image to view results")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        splitter.addWidget(self.image_label)

        # Details
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumWidth(250)
        splitter.addWidget(self.details_text)

        splitter.setStretchFactor(1, 1) # Image viewer gets most space
        layout.addWidget(splitter)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt *.onnx)")
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder_edit.setText(folder)

    def on_run_clicked(self):
        model_path = self.model_path_edit.text()
        img_folder = self.image_folder_edit.text()
        if model_path and img_folder:
            self.run_btn.setEnabled(False)
            self.file_list.clear()
            self.current_results = {}
            self.inference_requested.emit(model_path, img_folder)
        else:
            self.details_text.setText("Please select both model and image folder.")

    def update_results(self, results):
        # results is a list of dicts: {'file': path, 'detections': [...], 'image_path': ...}
        for res in results:
            filename = os.path.basename(res['image_path'])
            self.current_results[filename] = res
            self.file_list.addItem(filename)
        
        self.run_btn.setEnabled(True)
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, current, previous):
        if not current:
            return
        
        filename = current.text()
        if filename in self.current_results:
            data = self.current_results[filename]
            self.display_image(data)
            self.display_details(data)

    def display_image(self, data):
        image_path = data['image_path']
        detections = data['detections']
        
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("Failed to load image")
            return

        # Draw bounding boxes
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0), 3)
        painter.setPen(pen)
        
        # detections: list of [x1, y1, x2, y2, conf, class_name]
        for det in detections:
            x1, y1, x2, y2, conf, cls_name = det
            w = x2 - x1
            h = y2 - y1
            painter.drawRect(x1, y1, w, h)
            painter.drawText(x1, y1 - 5, f"{cls_name} {conf:.2f}")
        
        painter.end()
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def display_details(self, data):
        detections = data['detections']
        text = f"File: {os.path.basename(data['image_path'])}\n"
        text += f"Detections: {len(detections)}\n\n"
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_name = det
            text += f"{i+1}. {cls_name}\n"
            text += f"   Conf: {conf:.2f}\n"
            text += f"   ROI: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]\n\n"
            
        self.details_text.setText(text)
