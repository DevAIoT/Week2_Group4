import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "raspberry_pi" / "src"))

from swing_analyzer import SwingAnalyzer
from motion_processor import MotionProcessor
from models import ProcessedMotionData, SwingPhase

@pytest.fixture
def analyzer_config():
    return {
        'data_processing': {
            'swing_detection_threshold': 2.0,
            'min_swing_duration': 0.5,
            'max_swing_duration': 3.0,
            'sample_rate': 100,
            'filter_alpha': 0.98
        },
        'physics': {
            'club_head_efficiency': 1.4
        }
    }

@pytest.fixture
def swing_analyzer(analyzer_config):
    return SwingAnalyzer(analyzer_config)

def create_motion_data(accel_magnitude, angular_vel_magnitude, timestamp):
    """Helper function to create test motion data"""
    return ProcessedMotionData(
        acceleration=np.array([accel_magnitude, 0, 0]),
        angular_velocity=np.array([angular_vel_magnitude, 0, 0]),
        orientation=np.array([0, 0, 0]),
        timestamp=timestamp
    )

def test_swing_analyzer_initialization(swing_analyzer):
    assert swing_analyzer.current_phase == SwingPhase.IDLE
    assert swing_analyzer.swing_start_time is None
    assert len(swing_analyzer.current_swing_data) == 0

def test_idle_to_address_transition(swing_analyzer):
    # Simulate stationary period (address position)
    timestamp = time.time()
    
    # Send several low-motion data points
    for i in range(25):
        motion_data = create_motion_data(0.5, 0.1, timestamp + i * 0.01)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    # Should transition to address phase
    assert swing_analyzer.current_phase == SwingPhase.ADDRESS

def test_address_to_backswing_transition(swing_analyzer):
    # First establish address position
    timestamp = time.time()
    
    # Stationary period
    for i in range(25):
        motion_data = create_motion_data(0.5, 0.1, timestamp + i * 0.01)
        swing_analyzer.analyze_motion(motion_data)
    
    # Now start backswing with increased acceleration
    motion_data = create_motion_data(2.5, 1.0, timestamp + 0.5)
    phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    assert swing_analyzer.current_phase == SwingPhase.BACKSWING

def test_complete_swing_simulation(swing_analyzer):
    """Test a complete swing sequence"""
    timestamp = time.time()
    dt = 0.01  # 10ms intervals
    
    # Phase 1: Address (stationary)
    for i in range(25):
        motion_data = create_motion_data(0.5, 0.1, timestamp + i * dt)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    assert swing_analyzer.current_phase == SwingPhase.ADDRESS
    
    # Phase 2: Backswing (moderate acceleration)
    for i in range(25, 50):
        motion_data = create_motion_data(2.0, 2.0, timestamp + i * dt)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    assert swing_analyzer.current_phase == SwingPhase.BACKSWING
    
    # Phase 3: Top of backswing and start of downswing
    # Add some data to trigger backswing peak detection
    for i in range(50, 70):
        # Simulate changing angular velocity pattern
        angular_vel = 3.0 if i < 60 else 1.5  # Peak then decrease
        motion_data = create_motion_data(3.0, angular_vel, timestamp + i * dt)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    # Phase 4: Impact (high acceleration)
    for i in range(70, 75):
        motion_data = create_motion_data(10.0, 5.0, timestamp + i * dt)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    assert swing_analyzer.current_phase in [SwingPhase.IMPACT, SwingPhase.FOLLOW_THROUGH]
    
    # Phase 5: Follow through and return to idle
    for i in range(75, 100):
        # Gradually decrease acceleration
        accel = max(1.0, 10.0 - (i - 75) * 0.5)
        motion_data = create_motion_data(accel, 2.0, timestamp + i * dt)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    # Final deceleration to idle
    for i in range(100, 120):
        motion_data = create_motion_data(0.5, 0.1, timestamp + i * dt)
        phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    # Should eventually return to idle and produce metrics
    assert swing_analyzer.current_phase == SwingPhase.IDLE or metrics is not None

def test_swing_timeout(swing_analyzer):
    """Test that swing times out if it takes too long"""
    timestamp = time.time()
    
    # Start a swing
    for i in range(25):
        motion_data = create_motion_data(0.5, 0.1, timestamp + i * 0.01)
        swing_analyzer.analyze_motion(motion_data)
    
    # Enter backswing
    motion_data = create_motion_data(2.5, 1.0, timestamp + 0.5)
    swing_analyzer.analyze_motion(motion_data)
    
    # Simulate time passing beyond max_swing_duration
    long_time_later = timestamp + 5.0  # 5 seconds later
    motion_data = create_motion_data(1.0, 0.5, long_time_later)
    phase, metrics = swing_analyzer.analyze_motion(motion_data)
    
    # Should return to idle due to timeout
    assert phase == SwingPhase.IDLE

def test_launch_conditions_calculation(swing_analyzer):
    """Test launch conditions calculation from swing metrics"""
    from models import SwingMetrics
    
    test_metrics = SwingMetrics(
        max_speed=30.0,
        impact_speed=25.0,
        swing_angle=15.0,
        attack_angle=5.0,
        swing_plane=10.0,
        tempo=1.2,
        swing_path="square"
    )
    
    launch_conditions = swing_analyzer.calculate_launch_conditions(test_metrics)
    
    assert launch_conditions.ball_speed > 0
    assert 0 < launch_conditions.launch_angle < 30
    assert launch_conditions.spin_rate > 0
    assert launch_conditions.azimuth == 0.0  # Square path should be 0

def test_swing_path_determination(swing_analyzer):
    """Test swing path determination from motion data"""
    timestamp = time.time()
    
    # Simulate in-to-out swing path (positive yaw change)
    swing_analyzer.current_swing_data = [
        ProcessedMotionData(
            acceleration=np.array([5.0, 0, 0]),
            angular_velocity=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0.0]),  # Start yaw
            timestamp=timestamp
        ),
        ProcessedMotionData(
            acceleration=np.array([5.0, 0, 0]),
            angular_velocity=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0.3]),  # End yaw
            timestamp=timestamp + 1.0
        )
    ]
    
    swing_path = swing_analyzer._determine_swing_path()
    assert swing_path == "in-to-out"

def test_peak_speed_tracking(swing_analyzer):
    """Test that peak speed is tracked correctly"""
    timestamp = time.time()
    
    # Send progressively higher speed data
    speeds = [1.0, 5.0, 10.0, 8.0, 3.0]
    
    for i, speed in enumerate(speeds):
        motion_data = create_motion_data(speed, 1.0, timestamp + i * 0.1)
        swing_analyzer.analyze_motion(motion_data)
    
    # Peak speed should be 10.0
    assert swing_analyzer.peak_speed == 10.0

def test_stationary_detection(swing_analyzer):
    """Test stationary detection logic"""
    # Fill speed buffer with low values
    for _ in range(30):
        swing_analyzer.speed_buffer.append(0.5)
    
    assert swing_analyzer._is_stationary()
    
    # Add high speed value
    swing_analyzer.speed_buffer.append(5.0)
    assert not swing_analyzer._is_stationary()

def test_deceleration_detection(swing_analyzer):
    """Test deceleration detection logic"""
    # Fill speed buffer with decreasing values
    for speed in [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]:
        swing_analyzer.speed_buffer.append(speed)
    
    assert swing_analyzer._is_decelerating()
    
    # Clear and add increasing values
    swing_analyzer.speed_buffer.clear()
    for speed in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        swing_analyzer.speed_buffer.append(speed)
    
    assert not swing_analyzer._is_decelerating()

if __name__ == "__main__":
    pytest.main([__file__])