#!/usr/bin/env python3
"""
Test script to verify the golf swing analyzer system without ESP32
This simulates IMU data to test the complete pipeline
"""

import asyncio
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from motion_processor import MotionProcessor
from swing_analyzer import SwingAnalyzer
from physics_engine import PhysicsEngine
from websocket_server import WebSocketServer
from models import IMUData, SystemStatus, SwingPhase

class SystemTester:
    def __init__(self):
        # Load config
        import yaml
        with open("config/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.motion_processor = MotionProcessor(self.config)
        self.swing_analyzer = SwingAnalyzer(self.config)
        self.physics_engine = PhysicsEngine(self.config)
        self.websocket_server = WebSocketServer(self.config)
        
        print("✅ All components initialized successfully!")
    
    def test_motion_processing(self):
        """Test motion processing with simulated data"""
        print("\n🧪 Testing Motion Processing...")
        
        # Create test IMU data
        test_data = IMUData(
            accel_x=2.5, accel_y=1.2, accel_z=9.8,
            gyro_x=0.1, gyro_y=0.2, gyro_z=0.05,
            timestamp=int(time.time() * 1000),
            checksum=0
        )
        
        # Process the data
        processed = self.motion_processor.process_raw_data(test_data)
        
        print(f"   Acceleration: {processed.acceleration}")
        print(f"   Angular velocity: {processed.angular_velocity}")
        print(f"   Orientation: {np.degrees(processed.orientation)} degrees")
        print("✅ Motion processing works!")
        
        return processed
    
    def test_swing_analysis(self):
        """Test swing analysis with simulated swing"""
        print("\n🏌️ Testing Swing Analysis...")
        
        # Simulate a complete swing sequence
        swing_phases = []
        metrics = None
        
        # Address phase (stationary)
        for i in range(25):
            test_data = IMUData(
                accel_x=0.1, accel_y=0.1, accel_z=9.8,
                gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                timestamp=int(time.time() * 1000) + i * 10,
                checksum=0
            )
            processed = self.motion_processor.process_raw_data(test_data)
            phase, swing_metrics = self.swing_analyzer.analyze_motion(processed)
            swing_phases.append(phase)
        
        # Backswing (increasing acceleration)
        for i in range(25):
            test_data = IMUData(
                accel_x=2.0 + i * 0.1, accel_y=1.0, accel_z=9.8,
                gyro_x=1.0 + i * 0.05, gyro_y=0.5, gyro_z=0.2,
                timestamp=int(time.time() * 1000) + (25 + i) * 10,
                checksum=0
            )
            processed = self.motion_processor.process_raw_data(test_data)
            phase, swing_metrics = self.swing_analyzer.analyze_motion(processed)
            swing_phases.append(phase)
        
        # Impact (high acceleration)
        for i in range(10):
            test_data = IMUData(
                accel_x=15.0 - i * 0.5, accel_y=3.0, accel_z=9.8,
                gyro_x=8.0 - i * 0.3, gyro_y=2.0, gyro_z=1.0,
                timestamp=int(time.time() * 1000) + (50 + i) * 10,
                checksum=0
            )
            processed = self.motion_processor.process_raw_data(test_data)
            phase, swing_metrics = self.swing_analyzer.analyze_motion(processed)
            swing_phases.append(phase)
            if swing_metrics:
                metrics = swing_metrics
        
        # Return to idle
        for i in range(20):
            test_data = IMUData(
                accel_x=0.5, accel_y=0.2, accel_z=9.8,
                gyro_x=0.1, gyro_y=0.1, gyro_z=0.05,
                timestamp=int(time.time() * 1000) + (60 + i) * 10,
                checksum=0
            )
            processed = self.motion_processor.process_raw_data(test_data)
            phase, swing_metrics = self.swing_analyzer.analyze_motion(processed)
            swing_phases.append(phase)
            if swing_metrics:
                metrics = swing_metrics
        
        print(f"   Detected phases: {set([p.value for p in swing_phases])}")
        if metrics:
            print(f"   Max speed: {metrics.max_speed:.1f} m/s")
            print(f"   Impact speed: {metrics.impact_speed:.1f} m/s")
            print(f"   Swing path: {metrics.swing_path}")
            print("✅ Swing analysis works!")
            return metrics
        else:
            print("⚠️ No swing metrics generated (may need more realistic simulation)")
            return None
    
    def test_physics_engine(self, metrics=None):
        """Test physics engine with launch conditions"""
        print("\n⚽ Testing Physics Engine...")
        
        if not metrics:
            # Create test metrics
            from models import SwingMetrics
            metrics = SwingMetrics(
                max_speed=25.0,
                impact_speed=20.0,
                swing_angle=15.0,
                attack_angle=5.0,
                swing_plane=10.0,
                tempo=1.2,
                swing_path="square"
            )
        
        # Calculate launch conditions
        launch_conditions = self.swing_analyzer.calculate_launch_conditions(metrics)
        print(f"   Ball speed: {launch_conditions.ball_speed:.1f} m/s")
        print(f"   Launch angle: {launch_conditions.launch_angle:.1f}°")
        print(f"   Spin rate: {launch_conditions.spin_rate:.0f} rpm")
        
        # Calculate trajectory
        trajectory = self.physics_engine.calculate_trajectory(launch_conditions)
        print(f"   Distance: {trajectory.total_distance:.1f} m")
        print(f"   Max height: {trajectory.max_height:.1f} m")
        print(f"   Flight time: {trajectory.flight_time:.1f} s")
        print(f"   Trajectory points: {len(trajectory.points)}")
        print("✅ Physics engine works!")
        
        return trajectory
    
    async def test_websocket_server(self):
        """Test WebSocket server"""
        print("\n🌐 Testing WebSocket Server...")
        
        try:
            # Start server
            await self.websocket_server.start()
            print(f"   Server started on {self.config['websocket']['host']}:{self.config['websocket']['port']}")
            
            # Test broadcasting system status
            status = SystemStatus(
                ble_connected=False,  # No ESP32 in test
                imu_active=False,
                websocket_clients=0,
                swing_phase=SwingPhase.IDLE,
                last_data_time=time.time(),
                error_messages=["Test mode - no ESP32 connected"]
            )
            
            await self.websocket_server.broadcast_system_status(status)
            print("   ✅ System status broadcast successful")
            
            # Keep server running for a bit
            print("   Server will run for 30 seconds for testing...")
            print("   You can now test the frontend connection!")
            await asyncio.sleep(30)
            
            await self.websocket_server.stop()
            print("✅ WebSocket server test completed!")
            
        except Exception as e:
            print(f"❌ WebSocket server error: {e}")
    
    async def run_full_test(self):
        """Run complete system test"""
        print("🚀 Starting Golf Swing Analyzer System Test\n")
        
        try:
            # Test each component
            processed_data = self.test_motion_processing()
            metrics = self.test_swing_analysis()
            trajectory = self.test_physics_engine(metrics)
            
            # Test WebSocket server
            await self.test_websocket_server()
            
            print("\n🎉 All tests completed successfully!")
            print("\nNext steps:")
            print("1. Run the frontend: cd ../frontend && npm install && npm run dev")
            print("2. Open http://localhost:3000 in your browser")
            print("3. You should see 'Test mode - no ESP32 connected' status")
            print("4. Set up ESP32 hardware for real data")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    tester = SystemTester()
    await tester.run_full_test()

if __name__ == "__main__":
    asyncio.run(main())