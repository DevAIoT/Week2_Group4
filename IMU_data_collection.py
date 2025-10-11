import serial
import csv
from datetime import datetime

port = 'COM6'  # change to your port
baud = 115200
label = 'idle' # change per session

with serial.Serial(port, baud, timeout=1) as ser, open(f'{label}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])
    while True:
        line = ser.readline().decode().strip()
        if line:
            parts = line.split(',')
            if len(parts) == 6:
                ts = datetime.now().isoformat()
                writer.writerow([ts] + parts + [label])
                print(parts)
