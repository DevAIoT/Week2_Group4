# Golf Swing ML System Deployment Guide

This guide explains how to deploy the Halloween Mini Golf frontend with ML-powered swing detection to your Raspberry Pi.

## System Overview

- **Frontend**: Halloween-themed mini golf game with WebSocket integration
- **Backend**: Simplified Python server with ML model integration
- **Hardware**: ESP32 with IMU sensor connected via USB
- **ML Model**: Pre-trained RandomForest classifier for swing type detection

## Files Structure

```
Week2_Group4/
├── frontend/
│   └── halloween_minigolf.html      # Complete game with WebSocket integration
├── raspberry_pi/
│   ├── src/
│   │   ├── golf_swing_server.py     # Simplified WebSocket server
│   │   ├── ml_predictor.py          # ML model integration
│   │   ├── serial_client.py         # ESP32 USB communication
│   │   └── models.py                # Data models
│   ├── models/
│   │   ├── swing_model.pkl          # Trained ML model
│   │   └── scaler.pkl               # Data scaler
│   ├── requirements.txt             # Python dependencies
│   └── start_server.py              # Startup script
```

## Deployment Steps

### 1. Raspberry Pi Setup

1. **Copy files to Raspberry Pi:**
   ```bash
   scp -r raspberry_pi/ pi@<raspberry_pi_ip>:~/golf_swing_system/
   scp frontend/halloween_minigolf.html pi@<raspberry_pi_ip>:~/golf_swing_system/frontend/
   ```

2. **Install Python dependencies:**
   ```bash
   ssh pi@<raspberry_pi_ip>
   cd ~/golf_swing_system/raspberry_pi
   pip install -r requirements.txt
   ```

3. **Connect ESP32 via USB:**
   - Connect your ESP32 with IMU sensor to Raspberry Pi via USB
   - The system will auto-detect the ESP32 port

### 2. Frontend Setup

1. **Update WebSocket IP in frontend:**
   - Edit `halloween_minigolf.html` line ~1517
   - Change `ws://192.168.1.100:8765` to your Raspberry Pi's IP address (e.g., `ws://10.78.111.133:8765`)

2. **Serve frontend files:**
   ```bash
   # Option 1: Simple HTTP server
   cd ~/golf_swing_system/frontend
   python3 -m http.server 8000
   
   # Option 2: Copy to web server directory
   sudo cp halloween_minigolf.html /var/www/html/
   ```

### 3. Start the System

1. **Start the ML server on Raspberry Pi:**
   ```bash
   cd ~/golf_swing_system/raspberry_pi
   python3 start_server.py
   ```

2. **Access the frontend:**
   - Open browser to `http://<raspberry_pi_ip>:8000/halloween_minigolf.html`
   - Or `http://<raspberry_pi_ip>/halloween_minigolf.html` if using web server

## Usage Instructions

### Playing the Game

1. **Check Connection Status:**
   - Look at status indicators in bottom-left corner
   - Ensure all systems show green (connected)

2. **Record a Swing:**
   - Click the "🏌️ Swing!" button (top-right area)
   - Perform your golf swing motion with the ESP32 sensor
   - Recording lasts 5 seconds automatically

3. **View Results:**
   - ML model analyzes your swing pattern
   - Prediction appears below the swing button
   - Golf ball automatically moves based on predicted swing type

### Swing Classifications

The ML model recognizes these swing types:
- **slow** → Gentle shot (SLOW trajectory)
- **medium** → Normal shot (MEDIUM trajectory)  
- **fast** → Power shot (FAST trajectory)
- **left** → Curve left (CURVE_LEFT trajectory)
- **right** → Curve right (CURVE_RIGHT trajectory)
- **idle** → No swing detected (defaults to SLOW)

## Troubleshooting

### Connection Issues

1. **WebSocket Connection Failed:**
   - Check Raspberry Pi IP address in frontend code
   - Ensure port 8765 is not blocked by firewall
   - Verify server is running: `ps aux | grep golf_swing_server`

2. **ESP32 Not Detected:**
   - Check USB connection
   - Run `lsusb` to see connected devices
   - Check serial permissions: `sudo usermod -a -G dialout $USER`

3. **ML Model Not Loading:**
   - Verify model files exist in `raspberry_pi/models/`
   - Check file permissions
   - Install required packages: `pip install -r requirements.txt`

### Performance Issues

1. **Slow Predictions:**
   - Reduce recording duration in `golf_swing_server.py`
   - Lower sampling rate in ESP32 code

2. **Memory Issues:**
   - Restart server periodically: `sudo systemctl restart golf_server`
   - Monitor with: `htop` or `free -h`

## System Architecture

```
[ESP32 + IMU] --USB--> [Raspberry Pi] --WebSocket--> [Browser Frontend]
      |                      |                           |
   IMU Data              ML Prediction                Golf Animation
```

1. **ESP32** sends IMU data via USB serial
2. **Raspberry Pi** collects data, runs ML model, sends predictions via WebSocket
3. **Frontend** receives predictions and triggers corresponding golf ball animations

## Advanced Configuration

### Modify Recording Duration
Edit `golf_swing_server.py` line ~26:
```python
self.recording_duration = 5.0  # Change to desired seconds
```

### Adjust ML Model Sensitivity
Edit `ml_predictor.py` line ~18:
```python
self.min_swing_samples = 50  # Minimum samples for prediction
```

### Add New Swing Types
1. Collect training data for new swing type
2. Retrain model with new data
3. Update trajectory mapping in `golf_swing_server.py` line ~177

## Technical Notes

- **WebSocket Protocol**: JSON messages for real-time communication
- **ML Features**: Statistical features (mean, std, min, max, range) from 6-axis IMU
- **Data Rate**: ~50Hz IMU sampling, 5-second recording windows
- **Model**: RandomForest classifier with 93% accuracy on test data
- **Compatibility**: Works with any modern browser supporting WebSockets

For technical support, check the console logs in both the browser developer tools and the Raspberry Pi terminal.