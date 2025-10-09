# Golf Swing Analyzer IoT System

A comprehensive ESP32, Raspberry Pi, and IMU sensor IoT system for golf swing motion analysis and ball trajectory prediction.

## System Architecture

- **ESP32 Device**: IMU sensor (MPU6050) for capturing swing motion data via BLE
- **Raspberry Pi**: Central processing unit for swing analysis and ball physics calculations  
- **Frontend Interface**: Web-based 3D mini golf simulator with real-time trajectory visualization

## Quick Start

### Prerequisites
- ESP32 development board with MPU6050 IMU sensor
- Raspberry Pi 4 with Bluetooth capability
- Python 3.9+ and uv package manager

### Installation

1. Clone the repository and install dependencies:
```bash
pip install uv
uv sync
```

2. Flash ESP32 firmware:
```bash
cd esp32_firmware
# Follow ESP32 setup instructions
```

3. Run Raspberry Pi data processor:
```bash
cd raspberry_pi
uv run python src/main.py
```

4. Start frontend simulator:
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
golf-swing-analyzer/
├── esp32_firmware/          # ESP32 IMU BLE firmware
├── raspberry_pi/            # Central processing and analysis
├── frontend/               # 3D web simulator
├── tests/                  # System tests
└── docs/                   # Documentation
```

## Features

- Real-time IMU motion capture at 100Hz
- BLE communication with low latency (<50ms)
- Advanced swing phase detection and analysis
- Physics-based ball trajectory prediction
- 3D visualization with environmental factors
- Modular, clean architecture

## Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black .
uv run isort .
```