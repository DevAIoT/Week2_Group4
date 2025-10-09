import asyncio
import json
import websockets
import logging
from typing import Set, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from models import SystemStatus, SwingMetrics, TrajectoryData, SwingPhase

class WebSocketServer:
    def __init__(self, config: dict):
        self.config = config
        self.host = config['websocket']['host']
        self.port = config['websocket']['port']
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        
        # Data storage for broadcasting
        self.latest_system_status: Optional[SystemStatus] = None
        self.latest_swing_metrics: Optional[SwingMetrics] = None
        self.latest_trajectory: Optional[TrajectoryData] = None
        
        # Message handlers
        self.message_handlers = {
            'handshake': self._handle_handshake,
            'request_system_status': self._handle_status_request,
            'request_calibration': self._handle_calibration_request,
            'clear_trajectory': self._handle_clear_trajectory,
            'pong': self._handle_pong
        }
    
    async def start(self):
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=self.config['websocket']['ping_interval'],
            ping_timeout=self.config['websocket']['ping_timeout']
        )
        
        logger.info("WebSocket server started successfully")
    
    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket, path):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New client connected: {client_id}")
        
        self.clients.add(websocket)
        
        try:
            # Send initial system status if available
            if self.latest_system_status:
                await self._send_to_client(websocket, 'system_status', self.latest_system_status.__dict__)
            
            async for message in websocket:
                try:
                    await self._handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    await self._send_error(websocket, f"Error processing message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Unexpected error with client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def _handle_message(self, websocket, message: str):
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type not in self.message_handlers:
                logger.warning(f"Unknown message type: {message_type}")
                await self._send_error(websocket, f"Unknown message type: {message_type}")
                return
            
            handler = self.message_handlers[message_type]
            await handler(websocket, data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            await self._send_error(websocket, "Invalid JSON format")
    
    async def _handle_handshake(self, websocket, data: Dict[str, Any]):
        client_info = {
            'client': data.get('client', 'unknown'),
            'version': data.get('version', 'unknown')
        }
        logger.info(f"Client handshake: {client_info}")
        
        # Send welcome message
        await self._send_to_client(websocket, 'handshake_ack', {
            'server': 'golf-swing-analyzer',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_status_request(self, websocket, data: Dict[str, Any]):
        if self.latest_system_status:
            await self._send_to_client(websocket, 'system_status', self.latest_system_status.__dict__)
        else:
            await self._send_to_client(websocket, 'system_status', {
                'ble_connected': False,
                'imu_active': False,
                'websocket_clients': len(self.clients),
                'swing_phase': SwingPhase.IDLE.value,
                'last_data_time': None,
                'error_messages': ['No system status available']
            })
    
    async def _handle_calibration_request(self, websocket, data: Dict[str, Any]):
        # This would trigger calibration on the main system
        logger.info("Calibration requested from client")
        await self._send_to_client(websocket, 'calibration_status', {
            'status': 'requested',
            'message': 'Calibration request received'
        })
    
    async def _handle_clear_trajectory(self, websocket, data: Dict[str, Any]):
        # Clear trajectory data
        self.latest_trajectory = None
        await self.broadcast_message('clear_trajectory', {'status': 'cleared'})
        logger.info("Trajectory cleared by client request")
    
    async def _handle_pong(self, websocket, data: Dict[str, Any]):
        # Handle pong response (for connection keep-alive)
        pass
    
    async def _send_to_client(self, websocket, message_type: str, payload: Any):
        try:
            message = {
                'type': message_type,
                'payload': payload,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Attempted to send to closed connection")
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
    
    async def _send_error(self, websocket, error_message: str):
        await self._send_to_client(websocket, 'error', {'message': error_message})
    
    async def broadcast_message(self, message_type: str, payload: Any):
        if not self.clients:
            return
        
        message = {
            'type': message_type,
            'payload': payload,
            'timestamp': datetime.now().isoformat()
        }
        
        message_json = json.dumps(message)
        
        # Send to all connected clients
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
        
        if disconnected_clients:
            logger.info(f"Removed {len(disconnected_clients)} disconnected clients")
    
    # Public methods for broadcasting data
    async def broadcast_system_status(self, status: SystemStatus):
        self.latest_system_status = status
        status_dict = status.__dict__.copy()
        
        # Convert enum to string
        if hasattr(status_dict['swing_phase'], 'value'):
            status_dict['swing_phase'] = status_dict['swing_phase'].value
        
        # Add client count
        status_dict['websocket_clients'] = len(self.clients)
        
        await self.broadcast_message('system_status', status_dict)
    
    async def broadcast_swing_metrics(self, metrics: SwingMetrics):
        self.latest_swing_metrics = metrics
        await self.broadcast_message('swing_metrics', metrics.__dict__)
    
    async def broadcast_trajectory_data(self, trajectory: TrajectoryData):
        self.latest_trajectory = trajectory
        
        # Convert trajectory data to JSON-serializable format
        trajectory_dict = {
            'points': [
                {
                    'x': point.x,
                    'y': point.y,
                    'z': point.z,
                    'time': point.time,
                    'velocity': point.velocity.tolist() if hasattr(point.velocity, 'tolist') else list(point.velocity)
                }
                for point in trajectory.points
            ],
            'total_distance': trajectory.total_distance,
            'max_height': trajectory.max_height,
            'flight_time': trajectory.flight_time,
            'landing_angle': trajectory.landing_angle
        }
        
        await self.broadcast_message('trajectory_data', trajectory_dict)
    
    async def broadcast_imu_data(self, imu_data: dict):
        # Only broadcast if there are clients and not too frequently
        if self.clients:
            await self.broadcast_message('imu_data', imu_data)
    
    def get_client_count(self) -> int:
        return len(self.clients)
    
    def is_running(self) -> bool:
        return self.server is not None