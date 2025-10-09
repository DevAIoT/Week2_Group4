import asyncio
import struct
from typing import Optional, Callable, Any
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from loguru import logger
from models import IMUData, SystemStatus

class BLEClient:
    def __init__(self, config: dict):
        self.config = config
        self.client: Optional[BleakClient] = None
        self.device_address: Optional[str] = None
        self.is_connected = False
        self.data_callback: Optional[Callable[[IMUData], None]] = None
        self.status_callback: Optional[Callable[[SystemStatus], None]] = None
        
    async def scan_for_device(self) -> Optional[str]:
        logger.info("Scanning for Golf Swing Analyzer device...")
        
        devices = await BleakScanner.discover(timeout=self.config['ble']['scan_timeout'])
        
        for device in devices:
            if device.name == self.config['ble']['device_name']:
                logger.info(f"Found device: {device.name} ({device.address})")
                return device.address
                
        logger.warning("Device not found during scan")
        return None
    
    async def connect(self) -> bool:
        if not self.device_address:
            self.device_address = await self.scan_for_device()
            if not self.device_address:
                return False
        
        try:
            self.client = BleakClient(self.device_address)
            await self.client.connect(timeout=self.config['ble']['connection_timeout'])
            
            if self.client.is_connected:
                logger.info(f"Connected to {self.device_address}")
                self.is_connected = True
                
                # Subscribe to notifications
                await self.client.start_notify(
                    self.config['ble']['characteristic_uuid'],
                    self._data_notification_handler
                )
                
                # Update status
                if self.status_callback:
                    status = SystemStatus(
                        ble_connected=True,
                        imu_active=True,
                        websocket_clients=0,
                        swing_phase=SwingPhase.IDLE,
                        last_data_time=None,
                        error_messages=[]
                    )
                    self.status_callback(status)
                
                return True
            else:
                logger.error("Failed to establish BLE connection")
                return False
                
        except Exception as e:
            logger.error(f"BLE connection error: {e}")
            return False
    
    async def disconnect(self):
        if self.client and self.is_connected:
            try:
                await self.client.stop_notify(self.config['ble']['characteristic_uuid'])
                await self.client.disconnect()
                self.is_connected = False
                logger.info("Disconnected from BLE device")
                
                # Update status
                if self.status_callback:
                    status = SystemStatus(
                        ble_connected=False,
                        imu_active=False,
                        websocket_clients=0,
                        swing_phase=SwingPhase.IDLE,
                        last_data_time=None,
                        error_messages=["BLE disconnected"]
                    )
                    self.status_callback(status)
                    
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    def _data_notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        try:
            # Unpack IMU data structure (6 floats + uint32 + uint8)
            if len(data) == 28:  # Expected packet size
                unpacked = struct.unpack('<6f1I1B', data)
                
                imu_data = IMUData(
                    accel_x=unpacked[0],
                    accel_y=unpacked[1],
                    accel_z=unpacked[2],
                    gyro_x=unpacked[3],
                    gyro_y=unpacked[4],
                    gyro_z=unpacked[5],
                    timestamp=unpacked[6],
                    checksum=unpacked[7]
                )
                
                # Verify checksum
                if self._verify_checksum(imu_data):
                    if self.data_callback:
                        self.data_callback(imu_data)
                else:
                    logger.warning("Checksum mismatch in received data")
            else:
                logger.warning(f"Unexpected data length: {len(data)}")
                
        except Exception as e:
            logger.error(f"Error processing BLE data: {e}")
    
    def _verify_checksum(self, data: IMUData) -> bool:
        # Simple XOR checksum verification
        values = [data.accel_x, data.accel_y, data.accel_z, 
                 data.gyro_x, data.gyro_y, data.gyro_z]
        
        checksum = 0
        for value in values:
            bytes_val = struct.pack('<f', value)
            for byte in bytes_val:
                checksum ^= byte
        
        # Include timestamp
        timestamp_bytes = struct.pack('<I', data.timestamp)
        for byte in timestamp_bytes:
            checksum ^= byte
            
        return (checksum & 0xFF) == data.checksum
    
    def set_data_callback(self, callback: Callable[[IMUData], None]):
        self.data_callback = callback
    
    def set_status_callback(self, callback: Callable[[SystemStatus], None]):
        self.status_callback = callback
    
    async def reconnect_loop(self):
        retry_count = 0
        max_retries = self.config['ble']['retry_attempts']
        
        while retry_count < max_retries:
            if await self.connect():
                return True
            
            retry_count += 1
            wait_time = min(2 ** retry_count, 30)  # Exponential backoff
            logger.info(f"Retrying connection in {wait_time} seconds... ({retry_count}/{max_retries})")
            await asyncio.sleep(wait_time)
        
        logger.error("Max retry attempts reached. Could not reconnect.")
        return False