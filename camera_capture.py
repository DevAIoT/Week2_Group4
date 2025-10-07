from picamera2 import Picamera2
import time
from datetime import datetime
import threading
import os

class GolfSwingCamera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.recording = False
        self.output_path = "swing_videos"
        os.makedirs(self.output_path, exist_ok=True)
        
        # Configure camera
        video_config = self.picam2.create_video_configuration(main={"size": (1920, 1080)})
        self.picam2.configure(video_config)
        self.picam2.start()
        
        print("Camera initialized. Press 'r' to start/stop recording, 'q' to quit.")

    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"{self.output_path}/swing_{timestamp}.h264"
            self.start_time = time.time()
            self.picam2.start_recording(self.filename)
            self.recording = True
            print(f"Started recording to {self.filename}")
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_recording(self):
        if self.recording:
            self.picam2.stop_recording()
            self.recording = False
            duration = time.time() - self.start_time
            print(f"Stopped recording. Duration: {duration:.2f} seconds")

    def run(self):
        try:
            while True:
                command = input().strip().lower()
                if command == 'r':
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif command == 'q':
                    if self.recording:
                        self.stop_recording()
                    self.picam2.stop()
                    print("Exiting...")
                    break
        except KeyboardInterrupt:
            if self.recording:
                self.stop_recording()
            self.picam2.stop()

if __name__ == "__main__":
    camera = GolfSwingCamera()
    camera.run()
