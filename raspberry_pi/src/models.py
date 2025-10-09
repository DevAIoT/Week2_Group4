from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np

@dataclass
class IMUData:
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    timestamp: int
    checksum: int

@dataclass
class ProcessedMotionData:
    acceleration: np.ndarray  # 3D acceleration vector
    angular_velocity: np.ndarray  # 3D angular velocity vector
    orientation: np.ndarray  # Quaternion or Euler angles
    timestamp: float
    
class SwingPhase(Enum):
    IDLE = "idle"
    ADDRESS = "address"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"

@dataclass
class SwingMetrics:
    max_speed: float  # m/s
    impact_speed: float  # m/s
    swing_angle: float  # degrees
    attack_angle: float  # degrees
    swing_plane: float  # degrees
    tempo: float  # seconds
    swing_path: str  # "in-to-out", "out-to-in", "square"
    
@dataclass
class LaunchConditions:
    ball_speed: float  # m/s
    launch_angle: float  # degrees
    spin_rate: float  # rpm
    azimuth: float  # degrees (direction)
    
@dataclass
class TrajectoryPoint:
    x: float  # meters
    y: float  # meters (height)
    z: float  # meters (lateral)
    time: float  # seconds
    velocity: np.ndarray  # 3D velocity vector
    
@dataclass
class TrajectoryData:
    points: List[TrajectoryPoint]
    total_distance: float
    max_height: float
    flight_time: float
    landing_angle: float
    
@dataclass
class SystemStatus:
    ble_connected: bool
    imu_active: bool
    websocket_clients: int
    swing_phase: SwingPhase
    last_data_time: Optional[float]
    error_messages: List[str]
    
@dataclass
class CalibrationData:
    accel_offset: np.ndarray
    gyro_offset: np.ndarray
    gravity_vector: np.ndarray
    is_calibrated: bool
    calibration_time: float