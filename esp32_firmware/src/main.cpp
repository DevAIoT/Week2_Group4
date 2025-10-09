#include <Arduino.h>
#include "imu_sensor.h"
#include "ble_server.h"
#include "config.h"

IMUSensor imuSensor;
BLEServer bleServer;
IMUData currentData;

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);
    
    Serial.println("=== Golf Swing Analyzer ESP32 ===");
    Serial.println("Version 1.0");
    Serial.println("Initializing system...");
    
    // Initialize I2C
    Wire.begin();
    
    // Initialize IMU sensor
    if (!imuSensor.initialize()) {
        Serial.println("ERROR: Failed to initialize IMU sensor");
        while (1) {
            delay(1000);
        }
    }
    
    // Calibrate IMU
    Serial.println("Press any key to start calibration or wait 5 seconds to skip...");
    unsigned long start_time = millis();
    bool calibrate = false;
    
    while (millis() - start_time < 5000) {
        if (Serial.available()) {
            calibrate = true;
            break;
        }
        delay(100);
    }
    
    if (calibrate) {
        if (!imuSensor.calibrate()) {
            Serial.println("ERROR: Failed to calibrate IMU");
        }
    } else {
        Serial.println("Skipping calibration - using default values");
    }
    
    // Initialize BLE server
    if (!bleServer.initialize()) {
        Serial.println("ERROR: Failed to initialize BLE server");
        while (1) {
            delay(1000);
        }
    }
    
    Serial.println("System initialized successfully!");
    Serial.println("Ready to capture golf swing data...");
}

void loop() {
    // Handle BLE connections
    bleServer.handleConnections();
    
    // Read IMU data when ready
    if (imuSensor.readData(currentData)) {
        // Send data via BLE if connected
        if (bleServer.isConnected()) {
            if (bleServer.sendData(currentData)) {
                if (DEBUG_ENABLED && Serial.available()) {
                    // Print debug info only when requested
                    Serial.printf("IMU Data - Accel: X=%.2f Y=%.2f Z=%.2f | Gyro: X=%.2f Y=%.2f Z=%.2f | Time=%lu\n",
                                  currentData.accel_x, currentData.accel_y, currentData.accel_z,
                                  currentData.gyro_x, currentData.gyro_y, currentData.gyro_z,
                                  currentData.timestamp);
                }
            }
        }
    }
    
    // Small delay to prevent overwhelming the system
    delay(1);
}