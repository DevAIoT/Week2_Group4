import serial
import csv
from datetime import datetime
import time

# ===== CONFIG =====
port = '/dev/ttyUSB0'   # or COM6 on Windows
baud = 115200
label = input("Enter swing label (idle, slow, medium, fast, left, right): ").strip()
duration = float(input("Enter recording duration (seconds): "))

filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ===== START =====
print(f"\nPreparing to record {label} swings for {duration} seconds...")
time.sleep(2)
print("Recording started... SWING NOW!\n")

with serial.Serial(port, baud, timeout=1) as ser, open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])
    
    start_time = time.time()
    while (time.time() - start_time) < duration:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 6:
            ts = datetime.now().isoformat()
            writer.writerow([ts] + parts + [label])
    
print(f"\n✅ Data saved to {filename}")
