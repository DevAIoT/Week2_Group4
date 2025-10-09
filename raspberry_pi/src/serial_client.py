import serial
import serial.tools.list_ports
import re
import time
import threading
from typing import Optional, Callable, List
from loguru import logger
from models import IMUData

class SerialClient:
    def __init__(self, config: dict):
        self.config = config
        self.port: Optional[str] = None
        self.baudrate = 115200  # Match your ESP32 code
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_reading = False
        self.read_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.data_callback: Optional[Callable[[IMUData], None]] = None
        self.status_callback: Optional[Callable, None] = None
        
        # Data parsing
        self.last_timestamp = time.time()
        
    def find_esp32_port(self) -> Optional[str]:
        """Automatically find ESP32 USB port"""
        logger.info("Scanning for ESP32 USB connection...")
        
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            logger.info(f"Found port: {port.device} - {port.description}")
            
            # Common ESP32 identifiers
            esp32_identifiers = [
                'CP210x',  # Common ESP32 USB chip
                'CH340',   # Another common USB chip
                'ESP32',
                'Silicon Labs',
                'USB-SERIAL CH340',
                'USB2.0-Serial'
            ]
            
            for identifier in esp32_identifiers:
                if identifier.lower() in port.description.lower():
                    logger.info(f"ESP32 device found: {port.device}")
                    return port.device
        
        # If no automatic detection, try common ports
        common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
        for port_name in common_ports:
            try:
                test_serial = serial.Serial(port_name, self.baudrate, timeout=1)
                test_serial.close()
                logger.info(f"Available port found: {port_name}")
                return port_name
            except:
                continue
        
        return None
    
    async def connect(self) -> bool:
        """Connect to ESP32 via USB serial"""
        try:
            # Find ESP32 port
            self.port = self.find_esp32_port()
            
            if not self.port:
                logger.error("No ESP32 USB device found")
                logger.info("Available ports:")
                for port in serial.tools.list_ports.comports():
                    logger.info(f"  {port.device}: {port.description}")
                return False
            
            # Connect to serial port
            logger.info(f"Connecting to ESP32 on {self.port}...")
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0,
                write_timeout=2.0
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            # Test communication
            if self._test_communication():
                self.is_connected = True
                logger.info("ESP32 connection established successfully")
                
                # Start reading thread
                self._start_reading_thread()
                
                return True
            else:
                logger.error("ESP32 communication test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to ESP32: {e}")
            return False
    
    def _test_communication(self) -> bool:
        """Test if ESP32 is sending expected data format"""
        try:
            # Clear buffer
            self.serial_connection.flushInput()
            
            logger.info("Testing ESP32 communication - reading sample data...")
            
            # Read more lines and show what we're getting
            sample_lines = []
            for i in range(10):
                line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                if line:  # Only log non-empty lines
                    sample_lines.append(line)
                    logger.info(f"ESP32 sample {i+1}: '{line}'")
                    
                    if self._parse_imu_line(line):
                        logger.info("✅ ESP32 data format confirmed!")
                        return True
            
            # If we get here, no lines parsed successfully
            logger.error("❌ ESP32 data format not recognized!")
            logger.error("Sample lines received:")
            for i, line in enumerate(sample_lines):
                logger.error(f"  Line {i+1}: '{line}'")
            
            logger.error("Expected formats:")
            logger.error("  1. 'Accel X: 123 | Y: 456 | Z: 789 | Gyro X: 123 | Y: 456 | Z: 789'")
            logger.error("  2. '1.23,4.56,7.89,1.23,4.56,7.89' (CSV)")
            logger.error("  3. '1.23 4.56 7.89 1.23 4.56 7.89' (space-separated)")
            logger.error("  4. '{\"ax\":1.23,\"ay\":4.56,\"az\":7.89,\"gx\":1.23,\"gy\":4.56,\"gz\":7.89}' (JSON)")
            
            return False
            
        except Exception as e:
            logger.error(f"Communication test error: {e}")
            return False
    
    def _parse_imu_line(self, line: str) -> Optional[IMUData]:
        """Parse ESP32 output line into IMUData - supports multiple formats"""
        try:
            # Debug: print what we're receiving
            logger.debug(f"Parsing line: '{line}'")
            
            # Try multiple common ESP32 formats
            imu_data = None
            
            # Format 1: "Accel X: -1234 | Y: 5678 | Z: 9012 | Gyro X: -345 | Y: 678 | Z: -901"
            pattern1 = r"Accel X:\s*(-?\d+)\s*\|\s*Y:\s*(-?\d+)\s*\|\s*Z:\s*(-?\d+)\s*\|\s*Gyro X:\s*(-?\d+)\s*\|\s*Y:\s*(-?\d+)\s*\|\s*Z:\s*(-?\d+)"
            match1 = re.search(pattern1, line)
            if match1:
                ax, ay, az, gx, gy, gz = map(int, match1.groups())
                imu_data = self._create_imu_data(ax, ay, az, gx, gy, gz, scale_raw=True)
            
            # Format 2: CSV format "ax,ay,az,gx,gy,gz"
            if not imu_data and ',' in line:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        ax, ay, az, gx, gy, gz = map(float, parts[:6])
                        imu_data = self._create_imu_data(ax, ay, az, gx, gy, gz, scale_raw=False)
                    except ValueError:
                        pass
            
            # Format 3: Space-separated "ax ay az gx gy gz"
            if not imu_data and ' ' in line:
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        ax, ay, az, gx, gy, gz = map(float, parts[:6])
                        imu_data = self._create_imu_data(ax, ay, az, gx, gy, gz, scale_raw=False)
                    except ValueError:
                        pass
            
            # Format 4: JSON-like format
            if not imu_data and '{' in line:
                import json
                try:
                    data = json.loads(line)
                    if all(key in data for key in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
                        imu_data = self._create_imu_data(
                            data['ax'], data['ay'], data['az'],
                            data['gx'], data['gy'], data['gz'],
                            scale_raw=False
                        )
                except (json.JSONDecodeError, KeyError):
                    pass
            
            if not imu_data:
                logger.debug(f"No matching format for line: '{line}'")
            
            return imu_data
            
        except Exception as e:
            logger.debug(f"Error parsing line '{line}': {e}")
            return None
    
    def _create_imu_data(self, ax, ay, az, gx, gy, gz, scale_raw=True):
        """Create IMUData object with optional scaling"""
        if scale_raw:
            # Convert from raw int16 to physical units
            # MPU6050 default scales: ±2g = ±16384, ±250°/s = ±131
            accel_scale = 2.0 * 9.81 / 16384.0  # Convert to m/s²
            gyro_scale = 250.0 * (3.14159 / 180.0) / 131.0  # Convert to rad/s
            
            ax *= accel_scale
            ay *= accel_scale
            az *= accel_scale
            gx *= gyro_scale
            gy *= gyro_scale
            gz *= gyro_scale
        
        return IMUData(
            accel_x=float(ax),
            accel_y=float(ay),
            accel_z=float(az),
            gyro_x=float(gx),
            gyro_y=float(gy),
            gyro_z=float(gz),
            timestamp=int(time.time() * 1000),
            checksum=0  # No checksum in serial mode
        )
    
    def _start_reading_thread(self):
        """Start background thread for reading serial data"""
        self.is_reading = True
        self.read_thread = threading.Thread(target=self._reading_loop, daemon=True)
        self.read_thread.start()
        logger.info("Serial reading thread started")
    
    def _reading_loop(self):
        """Main reading loop (runs in background thread)"""
        logger.info("Starting serial data reading loop")
        
        while self.is_reading and self.is_connected:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:  # Skip empty lines
                        imu_data = self._parse_imu_line(line)
                        
                        if imu_data and self.data_callback:
                            self.data_callback(imu_data)
                
                # Small delay to prevent CPU spinning
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Error in reading loop: {e}")
                time.sleep(0.1)
        
        logger.info("Serial reading loop ended")
    
    async def disconnect(self):
        """Disconnect from ESP32"""
        self.is_reading = False
        self.is_connected = False
        
        # Wait for reading thread to finish
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        
        # Close serial connection
        if self.serial_connection:
            try:
                self.serial_connection.close()
                logger.info("Serial connection closed")
            except Exception as e:
                logger.error(f"Error closing serial connection: {e}")
        
        self.serial_connection = None
    
    def set_data_callback(self, callback: Callable[[IMUData], None]):
        """Set callback for received IMU data"""
        self.data_callback = callback
    
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates"""
        self.status_callback = callback
    
    # For compatibility with BLE client interface
    async def reconnect_loop(self) -> bool:
        """Attempt to reconnect (for compatibility)"""
        return await self.connect()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to ESP32"""
        return self._is_connected
    
    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value
    
    def get_connection_info(self) -> dict:
        """Get connection information"""
        return {
            'type': 'USB Serial',
            'port': self.port,
            'baudrate': self.baudrate,
            'connected': self.is_connected
        }