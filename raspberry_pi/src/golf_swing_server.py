"""
Simplified Golf Swing Server
Handles ML prediction and WebSocket communication with frontend
"""

import asyncio
import json
import time
import websockets
from pathlib import Path
from loguru import logger
import signal
import sys
from typing import Set, Optional

from serial_client import SerialClient
from rule_based_swing_classifier import RuleBasedSwingClassifier
from models import IMUData
import pandas as pd

class GolfSwingServer:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Simple config
        self.config = {
            'websocket': {'host': '0.0.0.0', 'port': 8765},
            'system': {'log_level': 'INFO'}
        }
        
        self.is_running = False
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Initialize components
        self.serial_client = SerialClient(self.config)
        self.swing_classifier = RuleBasedSwingClassifier()  # Now auto-initializes with default rules
        
        logger.info("Using advanced physics-based swing classifier")
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = None
        self.recording_duration = 1.5  # 1.5 seconds
        self.recorded_data = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            level=self.config['system']['log_level'],
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            "logs/golf_server.log",
            rotation="10 MB",
            level=self.config['system']['log_level'],
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def start(self):
        logger.info("Starting Golf Swing Server")
        
        try:
            # Setup Serial client callbacks
            self.serial_client.set_data_callback(self._handle_imu_data)
            
            # Connect to ESP32 via USB
            logger.info("Attempting to connect to ESP32 via USB...")
            if await self.serial_client.connect():
                logger.info("ESP32 USB connection established")
            else:
                logger.warning("Failed to connect to ESP32 - will retry periodically")
            
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                self.handle_websocket,
                self.config['websocket']['host'],
                self.config['websocket']['port']
            )
            logger.info(f"WebSocket server started on {self.config['websocket']['host']}:{self.config['websocket']['port']}")
            
            self.is_running = True
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    async def handle_websocket(self, websocket):
        """Handle new WebSocket connections"""
        self.websocket_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            # Send initial status
            await self._send_to_client(websocket, {
                'type': 'status',
                'classifier_ready': True,
                'classifier_type': 'physics_based',
                'esp32_connected': self.serial_client.is_connected,
                'model_info': {
                    'type': 'advanced_physics_based',
                    'accuracy': '61.5%',
                    'classes': ['fast', 'medium', 'slow', 'left', 'right', 'idle']
                }
            })
            
            async for message in websocket:
                await self._handle_websocket_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.websocket_clients)}")
    
    async def _handle_websocket_message(self, websocket, message):
        """Handle messages from WebSocket clients"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'start_recording':
                await self._start_swing_recording()
            elif msg_type == 'stop_recording':
                await self._stop_swing_recording()
            elif msg_type == 'get_status':
                await self._send_status(websocket)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _start_swing_recording(self):
        """Start recording swing data for 5 seconds"""
        if self.is_recording:
            logger.warning("Already recording swing data")
            return
        
        if not self.serial_client.is_connected:
            await self._broadcast_message({
                'type': 'error',
                'message': 'ESP32 not connected'
            })
            return
        
        logger.info("Starting swing recording for 1.5 seconds...")
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recorded_data = []
        
        await self._broadcast_message({
            'type': 'recording_started',
            'duration': self.recording_duration
        })
        
        # Schedule automatic stop after 1.5 seconds
        asyncio.create_task(self._auto_stop_recording())
    
    async def _auto_stop_recording(self):
        """Automatically stop recording after the specified duration"""
        await asyncio.sleep(self.recording_duration)
        if self.is_recording:
            await self._stop_swing_recording()
    
    async def _stop_swing_recording(self):
        """Stop recording and process the swing data"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        recording_time = time.time() - self.recording_start_time
        
        logger.info(f"Swing recording stopped after {recording_time:.2f} seconds")
        logger.info(f"Collected {len(self.recorded_data)} samples")
        
        await self._broadcast_message({
            'type': 'recording_stopped',
            'samples_collected': len(self.recorded_data),
            'recording_time': recording_time
        })
        
        # Process the recorded data with ML model
        if len(self.recorded_data) >= 10:  # Minimum samples needed
            await self._process_swing_data()
        else:
            await self._broadcast_message({
                'type': 'error',
                'message': f'Not enough data collected: {len(self.recorded_data)} samples'
            })
    
    async def _process_swing_data(self):
        """Process recorded swing data with physics-based classifier"""
        try:
            if len(self.recorded_data) < 10:
                await self._broadcast_message({
                    'type': 'error', 
                    'message': 'Not enough data for classification'
                })
                return
            
            # Convert recorded data to DataFrame format
            df = pd.DataFrame(self.recorded_data)
            
            # Use the advanced classifier
            predicted_label, confidence, debug_info = self.swing_classifier.classify_swing(df)
            
            logger.info(f"Physics-based prediction: {predicted_label} ({confidence:.1%} confidence)")
            
            # Map predictions to frontend trajectory types
            trajectory_mapping = {
                'slow': 'SLOW',
                'medium': 'MEDIUM', 
                'fast': 'FAST',
                'left': 'CURVE_LEFT',
                'right': 'CURVE_RIGHT',
                'idle': 'SLOW'
            }
            
            trajectory_type = trajectory_mapping.get(predicted_label, 'MEDIUM')
            
            # Create probabilities dict for compatibility
            probabilities = {cls: 0.0 for cls in ['fast', 'medium', 'slow', 'left', 'right', 'idle']}
            probabilities[predicted_label] = confidence
            remaining_prob = (1.0 - confidence) / (len(probabilities) - 1)
            for cls in probabilities:
                if cls != predicted_label:
                    probabilities[cls] = remaining_prob
            
            await self._broadcast_message({
                'type': 'swing_prediction',
                'prediction': predicted_label,
                'confidence': int(confidence * 100),
                'probabilities': probabilities,
                'trajectory_type': trajectory_type,
                'samples_used': len(self.recorded_data),
                'classifier_type': 'physics_based',
                'debug_info': debug_info.get('combination_logic', [])
            })
                
        except Exception as e:
            logger.error(f"Error processing swing data: {e}")
            await self._broadcast_message({
                'type': 'error',
                'message': f'Processing error: {str(e)}'
            })
    
    def _handle_imu_data(self, imu_data: IMUData):
        """Handle incoming IMU data from ESP32"""
        try:
            # If recording, add to recorded data
            if self.is_recording:
                self.recorded_data.append({
                    'timestamp': imu_data.timestamp,
                    'ax': imu_data.accel_x,
                    'ay': imu_data.accel_y,
                    'az': imu_data.accel_z,
                    'gx': imu_data.gyro_x,
                    'gy': imu_data.gyro_y,
                    'gz': imu_data.gyro_z
                })
            
            # Schedule broadcast in the main event loop occasionally
            if int(time.time() * 10) % 10 == 0:  # Every 1 second
                # Schedule the broadcast message in the main event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self._broadcast_message({
                                'type': 'imu_data',
                                'data': {
                                    'accel_x': imu_data.accel_x,
                                    'accel_y': imu_data.accel_y,
                                    'accel_z': imu_data.accel_z,
                                    'gyro_x': imu_data.gyro_x,
                                    'gyro_y': imu_data.gyro_y,
                                    'gyro_z': imu_data.gyro_z,
                                    'timestamp': imu_data.timestamp
                                },
                                'recording': self.is_recording
                            }))
                        )
                except RuntimeError:
                    # No event loop running, skip broadcast
                    pass
            
        except Exception as e:
            logger.error(f"Error handling IMU data: {e}")
    
    async def _send_to_client(self, websocket, message):
        """Send message to a specific client"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def _broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if not self.websocket_clients:
            return
        
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    async def _send_status(self, websocket):
        """Send current status to client"""
        await self._send_to_client(websocket, {
            'type': 'status',
            'classifier_ready': True,
            'esp32_connected': self.serial_client.is_connected,
            'recording': self.is_recording,
            'connected_clients': len(self.websocket_clients),
            'model_info': {
                'type': 'advanced_physics_based',
                'accuracy': '61.5%',
                'classes': ['fast', 'medium', 'slow', 'left', 'right', 'idle']
            }
        })
    
    async def _main_loop(self):
        """Main server loop"""
        logger.info("Starting main server loop")
        
        while self.is_running:
            try:
                # Check ESP32 connection
                if not self.serial_client.is_connected:
                    logger.info("ESP32 disconnected, attempting to reconnect...")
                    await self.serial_client.reconnect_loop()
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Stop the server"""
        logger.info("Stopping Golf Swing Server")
        
        self.is_running = False
        
        # Disconnect Serial
        if self.serial_client:
            await self.serial_client.disconnect()
        
        # Close WebSocket server
        if hasattr(self, 'websocket_server'):
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("Server stopped")

async def main():
    server = None
    
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Create and start server
        server = GolfSwingServer()
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if server:
            await server.stop()

if __name__ == "__main__":
    asyncio.run(main())