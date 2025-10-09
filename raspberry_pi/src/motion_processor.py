import numpy as np
from typing import Deque, Optional
from collections import deque
import time
from loguru import logger
from models import IMUData, ProcessedMotionData, CalibrationData

class MotionProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['data_processing']['sample_rate']
        self.filter_alpha = config['data_processing']['filter_alpha']
        
        # Buffers for sensor fusion
        self.accel_buffer: Deque[np.ndarray] = deque(maxlen=10)
        self.gyro_buffer: Deque[np.ndarray] = deque(maxlen=10)
        self.timestamp_buffer: Deque[float] = deque(maxlen=10)
        
        # State variables for complementary filter
        self.orientation = np.array([0.0, 0.0, 0.0])  # Roll, Pitch, Yaw
        self.last_timestamp = None
        
        # Calibration data
        self.calibration: Optional[CalibrationData] = None
        self.is_calibrated = False
        
        # Statistics for auto-calibration
        self.stationary_buffer: Deque[np.ndarray] = deque(maxlen=int(self.sample_rate * 2))
        
    def set_calibration(self, calibration: CalibrationData):
        self.calibration = calibration
        self.is_calibrated = calibration.is_calibrated
        logger.info("Calibration data loaded")
    
    def process_raw_data(self, raw_data: IMUData) -> ProcessedMotionData:
        current_time = time.time()
        
        # Convert to numpy arrays
        accel = np.array([raw_data.accel_x, raw_data.accel_y, raw_data.accel_z])
        gyro = np.array([raw_data.gyro_x, raw_data.gyro_y, raw_data.gyro_z])
        
        # Apply calibration if available
        if self.is_calibrated and self.calibration:
            accel -= self.calibration.accel_offset
            gyro -= self.calibration.gyro_offset
        
        # Add to buffers
        self.accel_buffer.append(accel)
        self.gyro_buffer.append(gyro)
        self.timestamp_buffer.append(current_time)
        
        # Check for auto-calibration
        if not self.is_calibrated:
            self._check_auto_calibration(accel, gyro)
        
        # Apply complementary filter for orientation estimation
        orientation = self._complementary_filter(accel, gyro, current_time)
        
        return ProcessedMotionData(
            acceleration=accel,
            angular_velocity=gyro,
            orientation=orientation,
            timestamp=current_time
        )
    
    def _complementary_filter(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float) -> np.ndarray:
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return self.orientation
        
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Normalize accelerometer data
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0:
            accel_normalized = accel / accel_norm
        else:
            accel_normalized = accel
        
        # Calculate roll and pitch from accelerometer
        accel_roll = np.arctan2(accel_normalized[1], accel_normalized[2])
        accel_pitch = np.arctan2(-accel_normalized[0], 
                                 np.sqrt(accel_normalized[1]**2 + accel_normalized[2]**2))
        
        # Integrate gyroscope for orientation change
        gyro_roll = self.orientation[0] + gyro[0] * dt
        gyro_pitch = self.orientation[1] + gyro[1] * dt
        gyro_yaw = self.orientation[2] + gyro[2] * dt
        
        # Complementary filter
        alpha = self.filter_alpha
        
        # Only use accelerometer for roll and pitch (not affected by linear acceleration)
        if accel_norm > 8.0 and accel_norm < 12.0:  # Filter out high acceleration
            self.orientation[0] = alpha * gyro_roll + (1 - alpha) * accel_roll
            self.orientation[1] = alpha * gyro_pitch + (1 - alpha) * accel_pitch
        else:
            self.orientation[0] = gyro_roll
            self.orientation[1] = gyro_pitch
        
        # Yaw only from gyroscope (no magnetometer)
        self.orientation[2] = gyro_yaw
        
        return self.orientation.copy()
    
    def _check_auto_calibration(self, accel: np.ndarray, gyro: np.ndarray):
        if not self.config['calibration']['auto_calibrate']:
            return
        
        # Add current readings to stationary buffer
        combined_data = np.concatenate([accel, gyro])
        self.stationary_buffer.append(combined_data)
        
        if len(self.stationary_buffer) < int(self.sample_rate * self.config['calibration']['stationary_time']):
            return
        
        # Check if sensor has been stationary
        buffer_array = np.array(self.stationary_buffer)
        accel_std = np.std(buffer_array[:, :3], axis=0)
        gyro_std = np.std(buffer_array[:, 3:], axis=0)
        
        gravity_threshold = self.config['calibration']['gravity_threshold']
        
        if (np.all(accel_std < gravity_threshold) and 
            np.all(gyro_std < 0.1)):  # Low gyro movement
            
            # Perform auto-calibration
            accel_mean = np.mean(buffer_array[:, :3], axis=0)
            gyro_mean = np.mean(buffer_array[:, 3:], axis=0)
            
            # Gravity vector should point down (negative Z in typical orientation)
            gravity_vector = accel_mean / np.linalg.norm(accel_mean) * 9.81
            
            self.calibration = CalibrationData(
                accel_offset=accel_mean - gravity_vector,
                gyro_offset=gyro_mean,
                gravity_vector=gravity_vector,
                is_calibrated=True,
                calibration_time=time.time()
            )
            
            self.is_calibrated = True
            logger.info("Auto-calibration completed")
            logger.info(f"Accel offset: {self.calibration.accel_offset}")
            logger.info(f"Gyro offset: {self.calibration.gyro_offset}")
    
    def get_linear_acceleration(self, processed_data: ProcessedMotionData) -> np.ndarray:
        # Remove gravity component from acceleration
        if not self.is_calibrated or not self.calibration:
            return processed_data.acceleration
        
        # Create rotation matrix from orientation
        roll, pitch, yaw = processed_data.orientation
        
        # Gravity vector in body frame
        gravity_body = np.array([
            -np.sin(pitch),
            np.sin(roll) * np.cos(pitch),
            np.cos(roll) * np.cos(pitch)
        ]) * 9.81
        
        # Linear acceleration = total acceleration - gravity
        linear_accel = processed_data.acceleration - gravity_body
        
        return linear_accel
    
    def calculate_speed(self, window_size: int = 10) -> float:
        if len(self.accel_buffer) < window_size:
            return 0.0
        
        # Calculate speed from recent acceleration data
        recent_accels = list(self.accel_buffer)[-window_size:]
        recent_times = list(self.timestamp_buffer)[-window_size:]
        
        velocity = 0.0
        for i in range(1, len(recent_accels)):
            dt = recent_times[i] - recent_times[i-1]
            accel_mag = np.linalg.norm(recent_accels[i])
            velocity += accel_mag * dt
        
        return velocity
    
    def get_orientation_degrees(self) -> np.ndarray:
        return np.degrees(self.orientation)