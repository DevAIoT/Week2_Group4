from bluepy.btle import Peripheral, UUID, Characteristic, DefaultDelegate
import time
import csv
from datetime import datetime

# ESP32 BLE Configuration - match with ESP32 code
SERVICE_UUID = UUID("4fafc201-1fb5-459e-8fcc-c5c9c331914b")
CHARACTERISTIC_UUID = UUID("beb5483e-36e1-4688-b7f5-ea07361b26a8")

# Global variables
data_buffer = []
recording = False
output_file = None

class IMUDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)

    def handleNotification(self, cHandle, data):
        global data_buffer, recording, output_file
        
        # Decode received data
        imu_data = data.decode('utf-8').split(',')
        
        if len(imu_data) == 9:  # Verify we received all 9 values
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Create data dictionary
            data_dict = {
                'timestamp': timestamp,
                'accel_x': float(imu_data[0]),
                'accel_y': float(imu_data[1]),
                'accel_z': float(imu_data[2]),
                'gyro_x': float(imu_data[3]),
                'gyro_y': float(imu_data[4]),
                'gyro_z': float(imu_data[5]),
                'angle_x': float(imu_data[6]),
                'angle_y': float(imu_data[7]),
                'angle_z': float(imu_data[8])
            }
            
            # Print to console
            print(f"[{timestamp}] Angles: X={data_dict['angle_x']:.2f}, Y={data_dict['angle_y']:.2f}, Z={data_dict['angle_z']:.2f}")
            
            # Save if recording
            if recording and output_file:
                writer = csv.DictWriter(output_file, fieldnames=data_dict.keys())
                writer.writerow(data_dict)
                output_file.flush()

def main():
    global recording, output_file
    esp32_address = None  # Will be discovered
    
    print("Scanning for ESP32-IMU...")
    try:
        # Connect to ESP32
        p = Peripheral("ESP32-IMU", "random")
        print("Connected to ESP32-IMU")
        
        # Set up delegate
        p.setDelegate(IMUDelegate())
        
        # Get service and characteristic
        service = p.getServiceByUUID(SERVICE_UUID)
        chara = service.getCharacteristics(CHARACTERISTIC_UUID)[0]
        
        # Enable notifications
        setup_data = b"\x01\x00"
        p.writeCharacteristic(chara.getHandle() + 1, setup_data)
        
        print("Waiting for data... (Press 'r' to start/stop recording, 'q' to quit)")
        
        while True:
            if p.waitForNotifications(1.0):
                continue
                
            # Check for user input
            import sys, select
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line.strip().lower() == 'r':
                    recording = not recording
                    if recording:
                        # Create new CSV file
                        filename = f"golf_swing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        output_file = open(filename, 'w', newline='')
                        fieldnames = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 
                                     'gyro_x', 'gyro_y', 'gyro_z', 
                                     'angle_x', 'angle_y', 'angle_z']
                        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                        writer.writeheader()
                        print(f"Started recording to {filename}")
                    else:
                        if output_file:
                            output_file.close()
                            output_file = None
                        print("Stopped recording")
                elif line.strip().lower() == 'q':
                    break
        
        # Cleanup
        p.disconnect()
        if output_file:
            output_file.close()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
