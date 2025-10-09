import serial
import time
import numpy as np
import joblib
import pandas as pd

# ======== CONFIG ========
port = '/dev/ttyUSB0'  # or 'COM6' on Windows
baud = 115200
duration = 2.0  # seconds to record one swing
window_size = 100  # 50 Hz × 2 s
# =========================

# Load model + scaler
clf = joblib.load('swing_model.pkl')
scaler = joblib.load('scaler.pkl')

# Helper: extract same features as during training
def extract_features_from_df(df):
    feats = []
    for col in ['ax','ay','az','gx','gy','gz']:
        feats += [
            df[col].mean(),
            df[col].std(),
            df[col].min(),
            df[col].max(),
            df[col].max() - df[col].min()
        ]
    return np.array(feats).reshape(1, -1)

# ======== RECORD ========
print("Connecting to ESP32...")
with serial.Serial(port, baud, timeout=1) as ser:
    time.sleep(2)
    print("Ready! Swing when you see 'Recording...'")
    time.sleep(1)

    print("\nRecording...")
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
if len(data) < 10:
    print("⚠️ Not enough data collected. Try again.")
    raise SystemExit

# ======== FEATURE EXTRACTION ========
df = pd.DataFrame(data, columns=['ax','ay','az','gx','gy','gz'])
X = df.mean().to_numpy().reshape(1, -1)
X_scaled = scaler.transform(X)

# ======== PREDICTION ========
pred = clf.predict(X_scaled)[0]
proba = clf.predict_proba(X_scaled)[0]
print("🏌️‍♂️ Predicted Swing:", pred)
print("📊 Confidence:", round(max(proba)*100, 2), "%")

