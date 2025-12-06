# YOLO 無程式碼訓練平台 (YOLO No-Code Training Platform)

這是一個基於 Python 與 PySide6 開發的圖形化介面應用程式，旨在讓使用者無需編寫任何程式碼，即可輕鬆訓練與測試 YOLO 物件偵測模型。支援 YOLOv5, YOLOv8, 與 YOLOv11。

## 功能特色

*   **圖形化操作介面**: 直觀的 GUI 設計，無需使用指令列。
*   **多模型支援**: 支援選擇 YOLOv5, YOLOv8, YOLOv11 進行訓練。
*   **自定義資料集**: 支援 LabelImg 格式 (YOLO .txt) 的資料集，可輕鬆指定圖片與標籤路徑。
*   **超參數調整**: 可在介面上直接調整訓練輪數 (Epochs)、批次大小 (Batch Size) 與圖片尺寸 (Image Size)。
*   **自動化設定**: 自動產生 YOLO 訓練所需的 `data.yaml` 設定檔。
*   **即時日誌**: 訓練過程中的日誌會即時顯示在介面上。
*   **模型匯出**: 訓練完成後自動匯出 ONNX 格式模型。
*   **推論與視覺化**: 內建推論測試功能，可載入模型並對資料夾內的圖片進行辨識，並直接在介面上顯示框選結果與信心度。

## 安裝說明

### 前置需求
*   Python 3.8 或以上版本
*   建議使用虛擬環境 (Virtual Environment)

### 安裝步驟

1.  **複製專案**
    ```bash
    git clone https://github.com/a23444452/YOLO_No_Code_Training.git
    cd YOLO_No_Code_Training
    ```

2.  **建立並啟動虛擬環境**
    ```bash
    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **安裝相依套件**
    ```bash
    pip install -r requirements.txt
    ```

## 使用說明

### 啟動程式
在終端機執行以下指令啟動主程式：
```bash
python main.py
```

### 1. 訓練模型 (Training Tab)

在「訓練」分頁中，您可以設定訓練參數：

*   **專案設定**:
    *   **專案名稱**: 訓練結果將儲存在 `runs/detect/<專案名稱>`。
    *   **模型名稱**: 此次實驗的名稱。
    *   **YOLO 版本**: 下拉選擇要使用的版本 (v5, v8, v11)。
*   **資料集選擇**:
    *   點擊「瀏覽」選擇 **訓練圖片路徑** 與 **訓練標籤路徑**。
    *   (選填) 選擇 **驗證圖片** 與 **驗證標籤** 路徑。
    *   **類別名稱**: 輸入物件類別名稱，以逗號分隔 (例如: `cat, dog, person`)。
*   **超參數設定**:
    *   調整 Epochs (訓練次數)、Batch Size (批次大小)、Img Size (圖片解析度)。
*   **開始訓練**:
    *   點擊「開始訓練」按鈕。程式會自動下載預訓練模型並開始訓練，下方視窗會顯示進度。

### 2. 測試模型 (Inference Tab)

在「推論」分頁中，您可以測試訓練好的模型：

*   **選擇模型**: 點擊「選擇模型」載入 `.pt` (PyTorch) 或 `.onnx` 檔案。
    *   訓練好的模型通常位於 `runs/detect/<專案名稱>/<模型名稱>/weights/best.pt`。
*   **選擇圖片**: 選擇包含測試圖片的資料夾。
*   **執行推論**:
    *   點擊「執行推論」。
    *   完成後，點擊左側列表中的檔名，右側將顯示辨識結果圖片 (繪製 Bounding Box)，以及詳細的類別、信心度與座標資訊。

## 輸出檔案

*   訓練結果 (權重檔、圖表) 預設存放於專案目錄下的 `runs/detect/`。
*   `data.yaml`: 程式會根據您的輸入自動在專案根目錄產生此檔案。

## 注意事項

*   請確保您的標籤檔案 (.txt) 格式符合 YOLO 標準格式。
*   若您的電腦有 NVIDIA 顯卡並安裝了 CUDA，程式會自動使用 GPU 加速訓練；否則將使用 CPU。
*   macOS 使用者若有 M1/M2/M3 晶片，PyTorch (MPS) 加速通常會自動啟用。
