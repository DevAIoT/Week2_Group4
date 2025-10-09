import numpy as np
from typing import List, Optional, Tuple
from collections import deque
import time
from loguru import logger
from models import ProcessedMotionData, SwingPhase, SwingMetrics, LaunchConditions

class SwingAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.swing_threshold = config['data_processing']['swing_detection_threshold']
        self.min_swing_duration = config['data_processing']['min_swing_duration']
        self.max_swing_duration = config['data_processing']['max_swing_duration']
        
        # State tracking
        self.current_phase = SwingPhase.IDLE
        self.swing_start_time: Optional[float] = None
        self.phase_start_time: Optional[float] = None
        
        # Data buffers for swing analysis
        self.motion_buffer: deque = deque(maxlen=500)  # ~5 seconds at 100Hz
        self.speed_buffer: deque = deque(maxlen=100)   # ~1 second
        
        # Swing metrics tracking
        self.current_swing_data: List[ProcessedMotionData] = []
        self.peak_speed = 0.0
        self.peak_acceleration = 0.0
        
        # Phase detection parameters
        self.backswing_threshold = 1.5  # m/s²
        self.downswing_threshold = 3.0  # m/s²
        self.impact_threshold = 8.0     # m/s²
        
    def analyze_motion(self, motion_data: ProcessedMotionData) -> Tuple[SwingPhase, Optional[SwingMetrics]]:
        # Add to buffers
        self.motion_buffer.append(motion_data)
        
        # Calculate current speed
        speed = np.linalg.norm(motion_data.acceleration)
        self.speed_buffer.append(speed)
        
        # Update peak values
        self.peak_speed = max(self.peak_speed, speed)
        self.peak_acceleration = max(self.peak_acceleration, np.linalg.norm(motion_data.acceleration))
        
        # Detect swing phase transitions
        new_phase = self._detect_swing_phase(motion_data, speed)
        
        # Handle phase transitions
        if new_phase != self.current_phase:
            logger.info(f"Swing phase transition: {self.current_phase.value} -> {new_phase.value}")
            self.current_phase = new_phase
            self.phase_start_time = motion_data.timestamp
            
            # Start new swing
            if new_phase == SwingPhase.ADDRESS:
                self._start_new_swing(motion_data.timestamp)
            
            # End swing and calculate metrics
            elif new_phase == SwingPhase.IDLE and self.swing_start_time:
                return new_phase, self._calculate_swing_metrics()
        
        # Add data to current swing if active
        if self.current_phase != SwingPhase.IDLE:
            self.current_swing_data.append(motion_data)
        
        return self.current_phase, None
    
    def _detect_swing_phase(self, motion_data: ProcessedMotionData, speed: float) -> SwingPhase:
        current_time = motion_data.timestamp
        
        # Check for timeout conditions
        if self.swing_start_time and (current_time - self.swing_start_time) > self.max_swing_duration:
            logger.warning("Swing timeout - returning to idle")
            return SwingPhase.IDLE
        
        # State machine for swing phase detection
        if self.current_phase == SwingPhase.IDLE:
            # Look for address position (stationary period before movement)
            if self._is_stationary(window_size=20) and len(self.speed_buffer) >= 20:
                return SwingPhase.ADDRESS
                
        elif self.current_phase == SwingPhase.ADDRESS:
            # Look for backswing start (initial acceleration)
            if speed > self.backswing_threshold:
                return SwingPhase.BACKSWING
            
            # Timeout in address
            if self.phase_start_time and (current_time - self.phase_start_time) > 5.0:
                return SwingPhase.IDLE
                
        elif self.current_phase == SwingPhase.BACKSWING:
            # Look for peak of backswing (deceleration then acceleration)
            if self._detect_backswing_peak():
                return SwingPhase.DOWNSWING
                
        elif self.current_phase == SwingPhase.DOWNSWING:
            # Look for impact (peak acceleration)
            if speed > self.impact_threshold:
                return SwingPhase.IMPACT
                
        elif self.current_phase == SwingPhase.IMPACT:
            # Look for follow-through (continued high speed)
            if self.phase_start_time and (current_time - self.phase_start_time) > 0.1:
                return SwingPhase.FOLLOW_THROUGH
                
        elif self.current_phase == SwingPhase.FOLLOW_THROUGH:
            # Look for swing completion (return to low speed)
            if speed < self.swing_threshold and self._is_decelerating():
                return SwingPhase.IDLE
        
        return self.current_phase
    
    def _is_stationary(self, window_size: int = 20) -> bool:
        if len(self.speed_buffer) < window_size:
            return False
        
        recent_speeds = list(self.speed_buffer)[-window_size:]
        avg_speed = np.mean(recent_speeds)
        std_speed = np.std(recent_speeds)
        
        return avg_speed < self.swing_threshold and std_speed < 0.5
    
    def _is_decelerating(self, window_size: int = 10) -> bool:
        if len(self.speed_buffer) < window_size:
            return False
        
        recent_speeds = list(self.speed_buffer)[-window_size:]
        
        # Check if speeds are generally decreasing
        decreasing_count = 0
        for i in range(1, len(recent_speeds)):
            if recent_speeds[i] < recent_speeds[i-1]:
                decreasing_count += 1
        
        return decreasing_count > (window_size * 0.6)  # 60% decreasing
    
    def _detect_backswing_peak(self) -> bool:
        if len(self.current_swing_data) < 20:
            return False
        
        # Look for peak in gyroscope data (rotation reversal)
        recent_data = self.current_swing_data[-20:]
        
        # Check for direction change in angular velocity
        angular_velocities = [np.linalg.norm(data.angular_velocity) for data in recent_data]
        
        # Find peak and subsequent decrease
        if len(angular_velocities) >= 10:
            peak_idx = np.argmax(angular_velocities[:10])
            if peak_idx < 8 and angular_velocities[peak_idx] > 2.0:
                # Check for decrease after peak
                post_peak = angular_velocities[peak_idx:]
                if len(post_peak) > 5 and np.mean(post_peak[-5:]) < angular_velocities[peak_idx] * 0.7:
                    return True
        
        return False
    
    def _start_new_swing(self, timestamp: float):
        self.swing_start_time = timestamp
        self.current_swing_data = []
        self.peak_speed = 0.0
        self.peak_acceleration = 0.0
        logger.info("New swing started")
    
    def _calculate_swing_metrics(self) -> SwingMetrics:
        if not self.current_swing_data:
            return SwingMetrics(0, 0, 0, 0, 0, 0, "unknown")
        
        # Calculate swing duration
        swing_duration = self.current_swing_data[-1].timestamp - self.current_swing_data[0].timestamp
        
        # Find impact point (highest acceleration)
        impact_idx = 0
        max_accel = 0
        for i, data in enumerate(self.current_swing_data):
            accel_mag = np.linalg.norm(data.acceleration)
            if accel_mag > max_accel:
                max_accel = accel_mag
                impact_idx = i
        
        impact_data = self.current_swing_data[impact_idx] if impact_idx < len(self.current_swing_data) else self.current_swing_data[-1]
        
        # Calculate swing metrics
        max_speed = self.peak_speed
        impact_speed = np.linalg.norm(impact_data.acceleration)
        
        # Calculate swing angles from orientation data
        orientations = [data.orientation for data in self.current_swing_data]
        if orientations:
            swing_angle = np.degrees(np.max([abs(o[0]) for o in orientations]))  # Max roll
            attack_angle = np.degrees(orientations[impact_idx][1]) if impact_idx < len(orientations) else 0  # Pitch at impact
            swing_plane = np.degrees(np.std([o[2] for o in orientations]))  # Yaw consistency
        else:
            swing_angle = attack_angle = swing_plane = 0
        
        # Determine swing path
        swing_path = self._determine_swing_path()
        
        metrics = SwingMetrics(
            max_speed=max_speed,
            impact_speed=impact_speed,
            swing_angle=swing_angle,
            attack_angle=attack_angle,
            swing_plane=swing_plane,
            tempo=swing_duration,
            swing_path=swing_path
        )
        
        logger.info(f"Swing completed - Max Speed: {max_speed:.1f} m/s, Duration: {swing_duration:.2f}s")
        
        # Reset for next swing
        self.swing_start_time = None
        self.current_swing_data = []
        
        return metrics
    
    def _determine_swing_path(self) -> str:
        if not self.current_swing_data:
            return "unknown"
        
        # Analyze the yaw (rotation) pattern during swing
        yaw_values = [data.orientation[2] for data in self.current_swing_data]
        
        if len(yaw_values) < 10:
            return "unknown"
        
        # Calculate overall yaw change
        yaw_change = yaw_values[-1] - yaw_values[0]
        
        if yaw_change > 0.2:
            return "in-to-out"
        elif yaw_change < -0.2:
            return "out-to-in"
        else:
            return "square"
    
    def calculate_launch_conditions(self, swing_metrics: SwingMetrics) -> LaunchConditions:
        # Convert swing speed to ball speed using club head efficiency
        club_efficiency = self.config['physics']['club_head_efficiency']
        ball_speed = swing_metrics.impact_speed * club_efficiency
        
        # Estimate launch angle from attack angle and swing characteristics
        launch_angle = max(8.0, min(20.0, swing_metrics.attack_angle + 12.0))
        
        # Estimate spin rate based on swing speed and attack angle
        spin_rate = 2000 + (swing_metrics.impact_speed * 100) + (abs(swing_metrics.attack_angle) * 50)
        
        # Azimuth from swing path
        azimuth_map = {
            "in-to-out": 5.0,
            "out-to-in": -5.0,
            "square": 0.0,
            "unknown": 0.0
        }
        azimuth = azimuth_map.get(swing_metrics.swing_path, 0.0)
        
        return LaunchConditions(
            ball_speed=ball_speed,
            launch_angle=launch_angle,
            spin_rate=spin_rate,
            azimuth=azimuth
        )