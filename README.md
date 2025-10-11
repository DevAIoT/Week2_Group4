# 🏌️‍♂️ Golf Swing Analyzer IoT System

A comprehensive IoT solution combining **ESP32**, **Raspberry Pi**, and **MPU6050 IMU sensor** for **golf swing motion analysis** and **ball trajectory prediction**.  
It integrates hardware data collection, machine learning classification, and a 3D visualization frontend into one complete system.

---

## ⚙️ System Architecture

- **ESP32 Device** — Captures motion data using MPU6050 IMU and transmits it via BLE  
- **Raspberry Pi** — Central processing and machine learning inference unit  
- **Frontend Interface** — Web-based 3D mini golf simulator for real-time swing and trajectory visualization  

---

## 🚀 Quick Start

### 🧩 Prerequisites

- ESP32 development board with MPU6050 IMU sensor  
- Raspberry Pi 4 (with Bluetooth enabled)  
- Python 3.9+  
- [`uv`](https://github.com/astral-sh/uv) package manager  

---

### 🔧 Installation

1. **Clone the repository and install dependencies:**

   ```bash
   pip install uv
   uv sync
   ```

2. **Flash ESP32 firmware:**

   ```bash
   cd esp32_firmware
   # Follow ESP32 setup instructions
   ```

3. **Run Raspberry Pi data processor:**

   ```bash
   cd raspberry_pi
   uv run python src/main.py
   ```

4. **Start the frontend simulator:**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 🗂️ Project Structure

```
golf-swing-analyzer/
├── esp32_firmware/          # ESP32 IMU BLE firmware
├── raspberry_pi/            # Central data processing and ML inference
├── frontend/                # 3D visualization interface
├── tests/                   # Test scripts
└── docs/                    # Documentation and research notes
```

---

## 💡 Features

- Real-time IMU motion capture (100Hz sampling rate)  
- BLE communication with <50ms latency  
- Motion segmentation and swing phase detection  
- Physics-based ball trajectory prediction  
- Machine learning–powered swing classification (fast, slow, left, right, etc.)  
- Modular and extensible architecture  

---

# 🧠 IMU Swing Classifier

This subsystem trains a **machine learning model** to classify swing types such as  
`fast`, `slow`, `medium`, `left`, `right`, and `idle` using IMU data.  
It extracts **48 statistical and frequency features** from raw accelerometer and gyroscope readings.

---

## 📊 How It Works

### 1. Data Preparation

Each recorded swing is saved as a `.csv` file containing:

```
timestamp, ax, ay, az, gx, gy, gz, label
2025-10-09T11:30:57.548826, -12840, -2304, -10528, -257, 185, -124, right
```

- The script automatically extracts the label (e.g., “right”) from the filename  
- Expected file format: `label_YYYYMMDD_HHMMSS.csv`  

---

### 2. Feature Extraction

Each CSV file is condensed into a 48-dimensional feature vector composed of:

**Time-Domain Features**
- Mean  
- Standard Deviation (std)  
- Minimum & Maximum  
- Median  
- Root Mean Square (RMS)  

**Frequency-Domain Features**
- FFT Mean  
- FFT Peak  

Each of the 6 axes (`ax`, `ay`, `az`, `gx`, `gy`, `gz`) contributes 8 features → `6 × 8 = 48` total features per swing.

---

### 3. Model Training

- **Scaler:** Standardized using `StandardScaler`  
- **Encoder:** Label strings converted via `LabelEncoder`  
- **Model:** `RandomForestClassifier` with `class_weight='balanced'`  
- **Artifacts saved:**  
  - `swing_model.pkl`  
  - `scaler.pkl`  
  - `label_encoder.pkl`

---

### 4. Evaluation

- Prints a **classification report** (precision, recall, F1-score)  
- Displays a **confusion matrix**  
- Ensures generalization across multiple swing types  

---

## 🧩 Real-Time Swing Testing

Use this lightweight Python script to test the model in real time using your ESP32 and IMU.

```python
import serial, time, numpy as np, pandas as pd, joblib

# ===== CONFIG =====
port = '/dev/ttyUSB0'   # or 'COM6' on Windows
baud = 115200
duration = 2.0  # seconds to record
# ==================

# Load model and scaler
clf = joblib.load('swing_model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(df):
    feats = []
    for col in ['ax','ay','az','gx','gy','gz']:
        feats += [df[col].mean(), df[col].std(), df[col].min(), df[col].max(), df[col].max() - df[col].min()]
    return np.array(feats).reshape(1, -1)

print("Connecting to ESP32...")
with serial.Serial(port, baud, timeout=1) as ser:
    time.sleep(2)
    print("Recording...")
    start = time.time()
    data = []
    while time.time() - start < duration:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 6:
            try:
                ax, ay, az, gx, gy, gz = map(float, parts)
                data.append([ax, ay, az, gx, gy, gz])
            except ValueError:
                continue

print(f"Captured {len(data)} samples.")
df = pd.DataFrame(data, columns=['ax','ay','az','gx','gy','gz'])
X = extract_features(df)
X_scaled = scaler.transform(X)

pred = clf.predict(X_scaled)[0]
print("🏌️‍♂️ Predicted Swing:", pred)
```

---

## 🧰 Development

### Run Tests

```bash
uv run pytest
```

### Format Code

```bash
uv run black .
uv run isort .
```

---

## 🌐 Contributing

Pull requests are welcome!  
Follow **conventional commit** format and ensure pre-commit hooks pass before submitting.

---

## 📜 License

**MIT License** © 2025 Golf Swing Analyzer Contributors
