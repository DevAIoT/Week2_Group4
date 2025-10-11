# Golf Swing Analyzer - Setup Guide

This guide will help you set up and run the complete Golf Swing Analyzer IoT system.

## System Requirements

### Hardware
- ESP32 development board (ESP32-WROOM-32 recommended)
- MPU6050 IMU sensor module
- Breadboard and jumper wires
- Raspberry Pi 4 (2GB+ RAM recommended)
- Computer for development and frontend

### Software
- Python 3.9 or higher
- Node.js 16+ (for frontend)
- Arduino IDE or PlatformIO (for ESP32)
- uv package manager for Python

## Hardware Setup

### ESP32 + MPU6050 Wiring

Connect the MPU6050 to ESP32 as follows:
```
MPU6050    ESP32
VCC     -> 3.3V
GND     -> GND
SCL     -> GPIO 22
SDA     -> GPIO 21
```

### Power Supply
- ESP32 can be powered via USB during development
- For production, consider using a battery pack (3.7V LiPo recommended)

## Software Installation

### 1. Clone and Setup Project

```bash
git clone <repository-url>
cd golf-swing-analyzer

# Install uv if not already installed
pip install uv

# Install Python dependencies
uv sync
```

### 2. ESP32 Firmware Setup

#### Option A: Using PlatformIO (Recommended)
```bash
# Install PlatformIO
pip install platformio

# Navigate to firmware directory
cd esp32_firmware

# Build and upload
pio run --target upload

# Monitor serial output
pio device monitor
```

#### Option B: Using Arduino IDE
1. Install Arduino IDE
2. Add ESP32 board support:
   - File → Preferences → Additional Board Manager URLs
   - Add: `https://dl.espressif.com/dl/package_esp32_index.json`
3. Install required libraries:
   - Adafruit MPU6050
   - Adafruit Unified Sensor
   - NimBLE-Arduino
4. Open `esp32_firmware/src/main.cpp`
5. Select Board: "ESP32 Dev Module"
6. Upload the code

### 3. Raspberry Pi Setup

```bash
# Navigate to Raspberry Pi directory
cd raspberry_pi

# Make run script executable
chmod +x scripts/run.sh

# Run the system
./scripts/run.sh
```

### 4. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Configuration

### Raspberry Pi Configuration

Edit `raspberry_pi/config/config.yaml` to adjust:
- BLE connection parameters
- Data processing settings
- Physics simulation parameters
- WebSocket server settings

### ESP32 Configuration

Edit `esp32_firmware/include/config.h` to modify:
- BLE device name and UUIDs
- IMU sampling rate
- Calibration settings

## Running the System

### Step 1: Start ESP32
1. Power on the ESP32
2. The device will start advertising as "GolfSwingAnalyzer"
3. Monitor serial output for calibration and connection status

### Step 2: Start Raspberry Pi System
```bash
cd raspberry_pi
./scripts/run.sh
```

### Step 3: Start Frontend
```bash
cd frontend
npm run dev
```
Open http://localhost:3000 in your browser

### Step 4: Connect and Calibrate
1. The Raspberry Pi will automatically scan for and connect to the ESP32
2. Perform IMU calibration by keeping the sensor stationary for 10 seconds
3. The frontend should show "System Online" when everything is connected

## Usage

### Basic Operation
1. Attach the ESP32+IMU to a golf club grip or wear it on your hand
2. Take practice swings
3. Watch real-time trajectory visualization in the browser
4. View swing metrics and ball flight analysis

### Swing Detection
The system automatically detects swing phases:
- **Idle**: No motion detected
- **Address**: Stationary before swing
- **Backswing**: Initial acceleration
- **Downswing**: Peak acceleration toward impact
- **Impact**: Maximum acceleration
- **Follow-through**: Deceleration after impact

### Calibration
- Auto-calibration occurs when the sensor is stationary
- Manual calibration can be triggered via the ESP32 serial interface
- Calibration data is stored and reused between sessions

## Troubleshooting

### ESP32 Issues
```bash
# Check serial output
pio device monitor

# Common issues:
# - IMU not found: Check wiring
# - BLE not advertising: Reset ESP32
# - Power issues: Use external power supply
```

### Raspberry Pi Issues
```bash
# Check logs
tail -f raspberry_pi/logs/golf_analyzer.log

# Bluetooth permission issues
sudo usermod -a -G bluetooth $USER

# BLE connection issues
sudo systemctl restart bluetooth
```

### Frontend Issues
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# WebSocket connection issues
# Check if Raspberry Pi system is running
# Verify port 8765 is open
```

### Common Problems

#### "Device not found during scan"
- Ensure ESP32 is powered and programmed
- Check ESP32 is advertising (blue LED should blink)
- Restart Bluetooth: `sudo systemctl restart bluetooth`

#### "WebSocket connection failed"
- Verify Raspberry Pi system is running
- Check firewall settings
- Ensure correct IP address/hostname

#### "Poor trajectory accuracy"
- Perform IMU calibration
- Check sensor mounting (should be firmly attached)
- Verify swing detection thresholds in config

## Advanced Configuration

### Physics Parameters
Adjust ball flight physics in `config.yaml`:
```yaml
physics:
  ball_mass: 0.04593        # kg
  air_density: 1.225        # kg/m³
  drag_coefficient: 0.47    # golf ball drag
  magnus_coefficient: 0.5   # spin effect
```

### Data Processing
Tune swing detection in `config.yaml`:
```yaml
data_processing:
  swing_detection_threshold: 2.0  # m/s²
  filter_alpha: 0.98              # complementary filter
  sample_rate: 100                # Hz
```

### Performance Optimization
- Increase ESP32 sampling rate for better resolution
- Adjust WebSocket broadcast frequency to reduce network load
- Enable/disable debug logging for performance

## Development

### Running Tests
```bash
# Python tests
cd raspberry_pi
uv run pytest ../tests/

# Frontend tests (if implemented)
cd frontend
npm test
```

### Code Structure
```
golf-swing-analyzer/
├── esp32_firmware/          # ESP32 C++ code
├── raspberry_pi/            # Python processing system
├── frontend/               # Web-based 3D visualization
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

### Contributing
1. Follow existing code style and conventions
2. Add tests for new features
3. Update documentation
4. Use meaningful commit messages

## Safety and Usage Notes

- Ensure secure mounting of sensors during swings
- Start with slow practice swings before full swings
- Keep firmware and software updated
- Monitor battery levels during extended use
- Store calibration data for consistent results

## Support

For issues and questions:
1. Check this documentation
2. Review log files for error messages
3. Test individual components separately
4. Check hardware connections
5. Verify software dependencies