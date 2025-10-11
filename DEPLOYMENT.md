# Golf Swing Detection System - Space-Efficient Deployment Guide

This guide explains how to deploy the Halloween Mini Golf frontend with advanced physics-based swing detection to your Raspberry Pi with minimal space usage.

## 🚀 System Overview

- **Frontend**: Halloween-themed mini golf game with WebSocket integration
- **Backend**: Lightweight Python server with **physics-based classifier** (no heavy ML models)
- **Hardware**: ESP32 with IMU sensor connected via USB
- **Classifier**: Advanced multi-dimensional physics analysis (61.5% cross-validation accuracy)

## 📁 Files Structure

```
Week2_Group4/
├── frontend/
│   └── halloween_minigolf.html           # Complete game with WebSocket integration
├── raspberry_pi/
│   ├── src/
│   │   ├── golf_swing_server.py          # Simplified WebSocket server
│   │   ├── rule_based_swing_classifier.py # Advanced physics-based classifier
│   │   ├── serial_client.py              # ESP32 USB communication
│   │   └── models.py                     # Data models
│   ├── requirements.txt                  # Minimal Python dependencies
│   └── start_server.py                   # Startup script
```

## 🛠 Space-Efficient Deployment Steps

### 1. SSH into Raspberry Pi and System Package Setup

**Save space by installing large packages system-wide, smaller ones in virtual environment:**

```bash
# SSH into your Raspberry Pi
ssh pi@<raspberry_pi_ip>

# Update package manager
sudo apt update

# Install LARGE system packages (saves space vs pip install)
sudo apt install -y python3-pandas python3-numpy python3-pip

# Install other system dependencies
sudo apt install -y python3-venv
```

### 2. Create Virtual Environment for Small Packages

```bash
# Create virtual environment with access to system packages
python3 -m venv ~/golf_env --system-site-packages

# Activate environment
source ~/golf_env/bin/activate

# Install only the lightweight packages in virtual env
pip install websockets>=11.0 loguru>=0.7.0 pyserial>=3.5
```

### 3. Deploy Application Files

```bash
# Create application directory
mkdir -p ~/golf_swing_system

# Exit virtual environment temporarily
deactivate
```

**From your local machine, copy files:**
```bash
# Copy backend files
scp -r raspberry_pi/ pi@<raspberry_pi_ip>:~/golf_swing_system/

# Copy frontend file
scp frontend/halloween_minigolf.html pi@<raspberry_pi_ip>:~/golf_swing_system/frontend/
```

### 4. ESP32 Hardware Setup

```bash
# SSH back into Pi
ssh pi@<raspberry_pi_ip>

# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER

# Reboot to apply group changes
sudo reboot
```

**After reboot, reconnect and verify ESP32:**
```bash
ssh pi@<raspberry_pi_ip>

# Check if ESP32 is detected
lsusb | grep -i esp

# Check serial ports
ls -la /dev/ttyUSB* /dev/ttyACM*
```

### 5. Frontend Configuration

**Update WebSocket IP in frontend:**
```bash
# Edit the frontend file
nano ~/golf_swing_system/frontend/halloween_minigolf.html

# Find line ~1517 and change:
# From: ws://192.168.1.100:8765
# To: ws://<your_raspberry_pi_ip>:8765
```

### 6. Start the System

```bash
# Activate virtual environment
source ~/golf_env/bin/activate

# Navigate to application directory
cd ~/golf_swing_system/raspberry_pi

# Start the physics-based classifier server
python3 start_server.py
```

**Serve the frontend (in another terminal):**
```bash
# Option 1: Simple HTTP server
cd ~/golf_swing_system/frontend
python3 -m http.server 8000

# Option 2: Copy to web server (if nginx/apache installed)
sudo cp halloween_minigolf.html /var/www/html/
```

## 🎮 Usage Instructions

### Playing the Game

1. **Check Connection Status:**
   - Open browser: `http://<raspberry_pi_ip>:8000/halloween_minigolf.html`
   - Look at status indicators in bottom-left corner
   - Ensure all systems show green (connected)

2. **Record a Swing:**
   - Click the "🏌️ Swing!" button (top-right area)
   - Perform your golf swing motion with the ESP32 sensor
   - Recording lasts 2 seconds automatically

3. **View Results:**
   - Physics-based classifier analyzes your swing pattern
   - Prediction appears below the swing button
   - Golf ball automatically moves based on predicted swing type

### 🏌️ Swing Classifications

The advanced physics-based classifier recognizes:
- **fast** → Explosive acceleration patterns (FAST trajectory)
- **medium** → Moderate intensity (MEDIUM trajectory)  
- **slow** → Controlled, smooth movement (SLOW trajectory)
- **left** → Clear leftward rotational intent (CURVE_LEFT trajectory)
- **right** → Clear rightward rotational intent (CURVE_RIGHT trajectory)
- **idle** → Minimal movement/stationary (defaults to SLOW)

## 🔧 Troubleshooting

### Connection Issues

1. **WebSocket Connection Failed:**
   ```bash
   # Check if server is running
   ps aux | grep golf_swing_server
   
   # Check port availability
   sudo netstat -tlnp | grep 8765
   
   # Restart server
   source ~/golf_env/bin/activate
   cd ~/golf_swing_system/raspberry_pi
   python3 start_server.py
   ```

2. **ESP32 Not Detected:**
   ```bash
   # Check USB connection
   lsusb
   
   # Check serial permissions
   groups $USER  # Should include 'dialout'
   
   # Manual permissions if needed
   sudo chmod 666 /dev/ttyUSB0  # Adjust device name
   ```

3. **Import Errors:**
   ```bash
   # Ensure virtual environment is activated
   source ~/golf_env/bin/activate
   
   # Check system packages
   python3 -c "import pandas, numpy; print('System packages OK')"
   
   # Check virtual env packages
   pip list | grep -E "(websockets|loguru|pyserial)"
   ```

### Space Management

1. **Check disk usage:**
   ```bash
   df -h /
   du -sh ~/golf_env
   du -sh ~/golf_swing_system
   ```

2. **Clean up if needed:**
   ```bash
   # Remove pip cache
   pip cache purge
   
   # Clean apt cache
   sudo apt autoremove
   sudo apt autoclean
   ```

## ⚙️ System Architecture

```
[ESP32 + IMU] --USB--> [Raspberry Pi] --WebSocket--> [Browser Frontend]
      |                      |                           |
   IMU Data         Physics Classification           Golf Animation
```

1. **ESP32** sends IMU data via USB serial (accelerometer + gyroscope)
2. **Raspberry Pi** uses physics-based rules to classify swing patterns
3. **Frontend** receives predictions and triggers corresponding golf ball animations

## 🎯 Advanced Physics-Based Classification

The system uses **multi-dimensional physics analysis**:

- **Explosive Fast Detection**: `acc_std > 4000 AND acc_max > 25000`
- **Controlled Directional**: Strong gyroscope bias with moderate acceleration
- **Speed Hierarchy**: Acceleration variance patterns (idle < slow < medium < fast)
- **Direction Patterns**: Y-axis gyroscope thresholds for left/right intent

**Key Innovation**: Uses acceleration intensity (explosive vs controlled) to distinguish fast center swings from directional swings.

## 🔬 Technical Specifications

- **Classification Method**: Rule-based physics analysis (no ML training required)
- **Accuracy**: 61.5% ± 8.4% (cross-validation tested)
- **Features**: Multi-dimensional acceleration and gyroscope pattern analysis
- **Data Rate**: ~50Hz IMU sampling, 2-second recording windows
- **Memory Usage**: Minimal (no large ML models, uses system numpy/pandas)
- **Real-time**: Low latency classification suitable for interactive gaming

## 🚀 Performance Benefits

✅ **Space Efficient**: No heavy ML models (saves ~100MB+ vs traditional ML approach)
✅ **Fast Startup**: No model loading time
✅ **Low Memory**: Uses system packages, minimal virtual environment
✅ **Real-time**: Physics calculations are instant
✅ **Robust**: Cross-validation tested, not overfitted to training data

For technical support, check the console logs in both the browser developer tools and the Raspberry Pi terminal.