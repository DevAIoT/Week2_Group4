import serial
import joblib
import numpy as np
import time

# Load trained model and scaler
clf = joblib.load("swing_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Adjust your serial port name below ---
port = '/dev/ttyUSB0'
baud = 115200

# --- Serial reading and live prediction ---
with serial.Serial(port, baud, timeout=1) as ser:
    print("Waiting for ESP32 to start...")
    time.sleep(3)
    print("🔍 Live swing prediction started (Ctrl+C to stop)\n")

    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) == 7:
                ax, ay, az, gx, gy, gz, _ = parts

                # Convert to float and reshape for model input
                X_live = np.array([[float(ax), float(ay), float(az),
                                    float(gx), float(gy), float(gz)]])
                X_live_scaled = scaler.transform(X_live)

                # Predict swing type
                pred = clf.predict(X_live_scaled)[0]
                print(f"🏌️ Swing detected: {pred}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print("Error:", e)
