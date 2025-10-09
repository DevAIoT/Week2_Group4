import asyncio
import yaml
import time
from pathlib import Path
from loguru import logger
import signal
import sys

from serial_client import SerialClient
from motion_processor import MotionProcessor
from swing_analyzer import SwingAnalyzer
from physics_engine import PhysicsEngine
from websocket_server import WebSocketServer
from models import SystemStatus, SwingPhase, IMUData

class GolfSwingAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.is_running = False
        
        # Initialize components
        self.serial_client = SerialClient(self.config)
        self.motion_processor = MotionProcessor(self.config)
        self.swing_analyzer = SwingAnalyzer(self.config)
        self.physics_engine = PhysicsEngine(self.config)
        self.websocket_server = WebSocketServer(self.config)
        
        # System state
        self.current_phase = SwingPhase.IDLE
        self.last_imu_data_time = None
        self.error_messages = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str) -> dict:
        config_file = Path(__file__).parent.parent / config_path
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_logging(self):
        log_level = self.config['system']['log_level']
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            "logs/golf_analyzer.log",
            rotation="10 MB",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def start(self):
        logger.info("Starting Golf Swing Analyzer System")
        logger.info(f"Version: {self.config['system']['version']}")
        
        try:
            # Start WebSocket server
            await self.websocket_server.start()
            
            # Setup Serial client callbacks
            self.serial_client.set_data_callback(self._handle_imu_data)
            self.serial_client.set_status_callback(self._handle_status_update)
            
            # Connect to ESP32 via USB
            logger.info("Attempting to connect to ESP32 via USB...")
            if await self.serial_client.connect():
                logger.info("ESP32 USB connection established")
            else:
                logger.warning("Failed to connect to ESP32 - will retry periodically")
            
            self.is_running = True
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
    
    async def _main_loop(self):
        logger.info("Starting main processing loop")
        
        status_update_interval = 5.0  # seconds
        last_status_update = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Periodic status updates
                if current_time - last_status_update >= status_update_interval:
                    await self._broadcast_system_status()
                    last_status_update = current_time
                
                # Check Serial connection
                if not self.serial_client.is_connected:
                    logger.info("ESP32 disconnected, attempting to reconnect...")
                    await self.serial_client.reconnect_loop()
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.error_messages.append(str(e))
                await asyncio.sleep(1)
    
    def _handle_imu_data(self, imu_data: IMUData):
        try:
            self.last_imu_data_time = time.time()
            
            # Process motion data
            processed_data = self.motion_processor.process_raw_data(imu_data)
            
            # Analyze swing
            phase, metrics = self.swing_analyzer.analyze_motion(processed_data)
            self.current_phase = phase
            
            # If swing completed, calculate trajectory
            if metrics:
                asyncio.create_task(self._handle_swing_completion(metrics))
            
            # Broadcast real-time data occasionally (not every packet to avoid spam)
            if int(time.time() * 10) % 5 == 0:  # Every 0.5 seconds
                imu_dict = {
                    'accel_x': imu_data.accel_x,
                    'accel_y': imu_data.accel_y,
                    'accel_z': imu_data.accel_z,
                    'gyro_x': imu_data.gyro_x,
                    'gyro_y': imu_data.gyro_y,
                    'gyro_z': imu_data.gyro_z,
                    'timestamp': imu_data.timestamp
                }
                asyncio.create_task(self.websocket_server.broadcast_imu_data(imu_dict))
            
        except Exception as e:
            logger.error(f"Error processing IMU data: {e}")
            self.error_messages.append(f"IMU processing error: {str(e)}")
    
    async def _handle_swing_completion(self, metrics):
        try:
            logger.info("Swing completed, calculating trajectory...")
            
            # Broadcast swing metrics
            await self.websocket_server.broadcast_swing_metrics(metrics)
            
            # Calculate launch conditions
            launch_conditions = self.swing_analyzer.calculate_launch_conditions(metrics)
            
            # Calculate trajectory
            trajectory = self.physics_engine.calculate_trajectory(launch_conditions)
            
            # Broadcast trajectory data
            await self.websocket_server.broadcast_trajectory_data(trajectory)
            
            logger.info(f"Trajectory calculated: {trajectory.total_distance:.1f}m distance, "
                       f"{trajectory.max_height:.1f}m height, {trajectory.flight_time:.1f}s flight time")
            
        except Exception as e:
            logger.error(f"Error handling swing completion: {e}")
            self.error_messages.append(f"Trajectory calculation error: {str(e)}")
    
    def _handle_status_update(self, status):
        # Handle status updates from BLE client
        pass
    
    async def _broadcast_system_status(self):
        try:
            status = SystemStatus(
                ble_connected=self.serial_client.is_connected,
                imu_active=self.serial_client.is_connected and self.last_imu_data_time is not None,
                websocket_clients=self.websocket_server.get_client_count(),
                swing_phase=self.current_phase,
                last_data_time=self.last_imu_data_time,
                error_messages=self.error_messages[-5:]  # Last 5 errors
            )
            
            await self.websocket_server.broadcast_system_status(status)
            
        except Exception as e:
            logger.error(f"Error broadcasting system status: {e}")
    
    async def stop(self):
        logger.info("Stopping Golf Swing Analyzer System")
        
        self.is_running = False
        
        # Disconnect Serial
        if self.serial_client:
            await self.serial_client.disconnect()
        
        # Stop WebSocket server
        if self.websocket_server:
            await self.websocket_server.stop()
        
        logger.info("System stopped")

async def main():
    analyzer = None
    
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Create and start analyzer
        analyzer = GolfSwingAnalyzer()
        await analyzer.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if analyzer:
            await analyzer.stop()

if __name__ == "__main__":
    asyncio.run(main())