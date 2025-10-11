import pytest
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "raspberry_pi" / "src"))

from physics_engine import PhysicsEngine
from models import LaunchConditions

@pytest.fixture
def physics_config():
    return {
        'physics': {
            'ball_mass': 0.04593,
            'ball_diameter': 0.04267,
            'air_density': 1.225,
            'gravity': 9.81,
            'drag_coefficient': 0.47,
            'magnus_coefficient': 0.5,
            'club_head_efficiency': 1.4
        }
    }

@pytest.fixture
def physics_engine(physics_config):
    return PhysicsEngine(physics_config)

@pytest.fixture
def test_launch_conditions():
    return LaunchConditions(
        ball_speed=50.0,  # m/s
        launch_angle=12.0,  # degrees
        spin_rate=2500,  # rpm
        azimuth=0.0  # degrees
    )

def test_physics_engine_initialization(physics_engine):
    assert physics_engine.ball_mass == 0.04593
    assert physics_engine.gravity == 9.81
    assert physics_engine.ball_radius == 0.04267 / 2

def test_trajectory_calculation(physics_engine, test_launch_conditions):
    trajectory = physics_engine.calculate_trajectory(test_launch_conditions)
    
    # Basic trajectory validation
    assert trajectory.total_distance > 0
    assert trajectory.max_height > 0
    assert trajectory.flight_time > 0
    assert len(trajectory.points) > 0
    
    # Reasonable golf shot expectations
    assert 50 < trajectory.total_distance < 300  # Reasonable distance
    assert 1 < trajectory.max_height < 100  # Reasonable height
    assert 1 < trajectory.flight_time < 10  # Reasonable flight time

def test_carry_distance_calculation(physics_engine, test_launch_conditions):
    carry_distance = physics_engine.calculate_carry_distance(test_launch_conditions)
    assert carry_distance > 0
    assert 50 < carry_distance < 300

def test_optimal_launch_angle(physics_engine):
    # Test different ball speeds
    low_speed_angle = physics_engine.calculate_optimal_launch_angle(25)
    medium_speed_angle = physics_engine.calculate_optimal_launch_angle(40)
    high_speed_angle = physics_engine.calculate_optimal_launch_angle(60)
    
    # Higher speeds should have lower optimal angles
    assert low_speed_angle > medium_speed_angle
    assert medium_speed_angle > high_speed_angle

def test_trajectory_quality_analysis(physics_engine, test_launch_conditions):
    trajectory = physics_engine.calculate_trajectory(test_launch_conditions)
    quality = physics_engine.analyze_trajectory_quality(trajectory, target_distance=150)
    
    assert 'overall_score' in quality
    assert 'distance_score' in quality
    assert 'height_score' in quality
    assert 'landing_score' in quality
    assert 0 <= quality['overall_score'] <= 100

def test_zero_speed_trajectory(physics_engine):
    launch_conditions = LaunchConditions(
        ball_speed=0.0,
        launch_angle=12.0,
        spin_rate=0,
        azimuth=0.0
    )
    
    trajectory = physics_engine.calculate_trajectory(launch_conditions)
    
    # Ball with no initial speed should not travel far
    assert trajectory.total_distance < 1.0
    assert trajectory.max_height < 1.0

def test_high_launch_angle(physics_engine):
    launch_conditions = LaunchConditions(
        ball_speed=50.0,
        launch_angle=45.0,  # Very high launch angle
        spin_rate=2500,
        azimuth=0.0
    )
    
    trajectory = physics_engine.calculate_trajectory(launch_conditions)
    
    # High launch angle should produce high trajectory
    assert trajectory.max_height > 20.0  # Should be quite high

def test_side_spin_effect(physics_engine):
    # Test ball with side spin (azimuth)
    straight_shot = LaunchConditions(
        ball_speed=50.0,
        launch_angle=12.0,
        spin_rate=2500,
        azimuth=0.0
    )
    
    fade_shot = LaunchConditions(
        ball_speed=50.0,
        launch_angle=12.0,
        spin_rate=2500,
        azimuth=5.0  # Side spin
    )
    
    straight_trajectory = physics_engine.calculate_trajectory(straight_shot)
    fade_trajectory = physics_engine.calculate_trajectory(fade_shot)
    
    # Side spin should affect the trajectory
    # Check if final Z position differs
    straight_final_z = straight_trajectory.points[-1].z if straight_trajectory.points else 0
    fade_final_z = fade_trajectory.points[-1].z if fade_trajectory.points else 0
    
    assert abs(straight_final_z - fade_final_z) > 0.1  # Should be different

if __name__ == "__main__":
    pytest.main([__file__])