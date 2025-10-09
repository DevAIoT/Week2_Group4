import numpy as np
from typing import List, Tuple
import math
from loguru import logger
from models import LaunchConditions, TrajectoryPoint, TrajectoryData

class PhysicsEngine:
    def __init__(self, config: dict):
        self.config = config
        
        # Physical constants
        self.ball_mass = config['physics']['ball_mass']
        self.ball_diameter = config['physics']['ball_diameter']
        self.air_density = config['physics']['air_density']
        self.gravity = config['physics']['gravity']
        self.drag_coefficient = config['physics']['drag_coefficient']
        self.magnus_coefficient = config['physics']['magnus_coefficient']
        
        # Calculated values
        self.ball_radius = self.ball_diameter / 2
        self.ball_area = math.pi * self.ball_radius ** 2
        
    def calculate_trajectory(self, launch_conditions: LaunchConditions, dt: float = 0.01) -> TrajectoryData:
        # Initial conditions
        v0 = launch_conditions.ball_speed
        launch_angle_rad = math.radians(launch_conditions.launch_angle)
        azimuth_rad = math.radians(launch_conditions.azimuth)
        spin_rate = launch_conditions.spin_rate
        
        # Initial velocity components
        vx = v0 * math.cos(launch_angle_rad) * math.cos(azimuth_rad)
        vy = v0 * math.sin(launch_angle_rad)
        vz = v0 * math.cos(launch_angle_rad) * math.sin(azimuth_rad)
        
        # Initial position
        x, y, z = 0.0, 0.0, 0.0
        
        # Trajectory points
        points: List[TrajectoryPoint] = []
        t = 0.0
        
        # Simulation parameters
        max_time = 20.0  # Maximum flight time
        ground_threshold = -0.1  # Below ground level
        
        while t < max_time and y > ground_threshold:
            # Current velocity magnitude
            velocity = np.array([vx, vy, vz])
            v_mag = np.linalg.norm(velocity)
            
            if v_mag < 0.1:  # Ball has essentially stopped
                break
            
            # Calculate forces
            drag_force = self._calculate_drag_force(velocity, v_mag)
            magnus_force = self._calculate_magnus_force(velocity, spin_rate, azimuth_rad)
            gravity_force = np.array([0, -self.gravity * self.ball_mass, 0])
            
            # Total force
            total_force = drag_force + magnus_force + gravity_force
            
            # Acceleration (F = ma)
            acceleration = total_force / self.ball_mass
            
            # Store current point
            points.append(TrajectoryPoint(
                x=x, y=y, z=z, time=t,
                velocity=velocity.copy()
            ))
            
            # Update velocity (Euler integration)
            vx += acceleration[0] * dt
            vy += acceleration[1] * dt
            vz += acceleration[2] * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            z += vz * dt
            
            # Update time
            t += dt
            
            # Spin decay (ball loses spin over time)
            spin_rate *= (1 - 0.001 * dt)  # 0.1% decay per 0.01s
        
        # Calculate trajectory statistics
        total_distance = x
        max_height = max(point.y for point in points) if points else 0
        flight_time = t
        
        # Calculate landing angle
        if len(points) > 1:
            final_velocity = points[-1].velocity
            landing_angle = math.degrees(math.atan2(-final_velocity[1], final_velocity[0]))
        else:
            landing_angle = 0
        
        logger.info(f"Trajectory calculated: Distance={total_distance:.1f}m, Height={max_height:.1f}m, Time={flight_time:.1f}s")
        
        return TrajectoryData(
            points=points,
            total_distance=total_distance,
            max_height=max_height,
            flight_time=flight_time,
            landing_angle=landing_angle
        )
    
    def _calculate_drag_force(self, velocity: np.ndarray, v_mag: float) -> np.ndarray:
        if v_mag == 0:
            return np.zeros(3)
        
        # Drag force opposes motion
        drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * self.ball_area * v_mag ** 2
        drag_direction = -velocity / v_mag
        
        return drag_magnitude * drag_direction
    
    def _calculate_magnus_force(self, velocity: np.ndarray, spin_rate: float, azimuth_rad: float) -> np.ndarray:
        if spin_rate == 0:
            return np.zeros(3)
        
        # Convert spin rate to angular velocity (rad/s)
        omega = (spin_rate * 2 * math.pi) / 60  # RPM to rad/s
        
        # Spin axis (assuming backspin primarily, with side spin from azimuth)
        spin_axis = np.array([
            -math.sin(azimuth_rad),  # Side spin component
            0,                       # No vertical spin axis
            math.cos(azimuth_rad)    # Primary backspin axis
        ])
        
        # Magnus force = k * (ω × v) where k is a constant
        # This creates lift for backspin and side force for side spin
        omega_vector = omega * spin_axis
        
        # Cross product: ω × v
        magnus_direction = np.cross(omega_vector, velocity)
        
        # Magnus force magnitude (simplified model)
        v_mag = np.linalg.norm(velocity)
        magnus_magnitude = self.magnus_coefficient * self.air_density * self.ball_area * v_mag * omega
        
        if np.linalg.norm(magnus_direction) > 0:
            magnus_direction = magnus_direction / np.linalg.norm(magnus_direction)
            return magnus_magnitude * magnus_direction
        else:
            return np.zeros(3)
    
    def calculate_carry_distance(self, launch_conditions: LaunchConditions) -> float:
        # Simplified carry distance calculation for quick estimates
        v0 = launch_conditions.ball_speed
        angle_rad = math.radians(launch_conditions.launch_angle)
        
        # Basic projectile motion with drag approximation
        # This is a simplified model for quick calculations
        range_no_air = (v0 ** 2 * math.sin(2 * angle_rad)) / self.gravity
        
        # Apply drag reduction factor (empirical)
        drag_factor = 1 - (0.001 * v0)  # Higher speeds lose more distance to drag
        
        # Apply Magnus effect (backspin increases carry)
        spin_factor = 1 + (launch_conditions.spin_rate / 10000)  # Simplified spin benefit
        
        estimated_carry = range_no_air * drag_factor * spin_factor
        
        return max(0, estimated_carry)
    
    def calculate_optimal_launch_angle(self, ball_speed: float) -> float:
        # Calculate optimal launch angle for maximum distance
        # This varies with ball speed due to air resistance
        
        if ball_speed < 30:  # Low speed
            return 15.0
        elif ball_speed < 50:  # Medium speed
            return 12.0
        else:  # High speed
            return 10.0
    
    def analyze_trajectory_quality(self, trajectory: TrajectoryData, target_distance: float = 150) -> dict:
        # Analyze trajectory quality metrics
        actual_distance = trajectory.total_distance
        distance_error = abs(actual_distance - target_distance)
        
        # Height quality (not too high, not too low)
        optimal_height = target_distance * 0.15  # Rough guideline: 15% of distance
        height_error = abs(trajectory.max_height - optimal_height)
        
        # Landing angle quality (not too steep)
        ideal_landing_angle = 45  # degrees
        landing_error = abs(trajectory.landing_angle - ideal_landing_angle)
        
        # Overall quality score (0-100)
        distance_score = max(0, 100 - (distance_error / target_distance * 100))
        height_score = max(0, 100 - (height_error / optimal_height * 100)) if optimal_height > 0 else 50
        landing_score = max(0, 100 - (landing_error / ideal_landing_angle * 100))
        
        overall_score = (distance_score + height_score + landing_score) / 3
        
        return {
            "overall_score": overall_score,
            "distance_score": distance_score,
            "height_score": height_score,
            "landing_score": landing_score,
            "distance_error": distance_error,
            "height_error": height_error,
            "landing_error": landing_error
        }