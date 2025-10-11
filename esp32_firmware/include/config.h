#ifndef CONFIG_H
#define CONFIG_H

// BLE Configuration
#define DEVICE_NAME "GolfSwingAnalyzer"
#define SERVICE_UUID "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"

// IMU Configuration
#define MPU6050_ADDRESS 0x68
#define SAMPLE_RATE_HZ 100
#define ACCEL_RANGE 2  // ±2g
#define GYRO_RANGE 250 // ±250°/s

// Data packet configuration
#define PACKET_SIZE 28  // 6 floats (accel + gyro) + timestamp + checksum
#define MAX_QUEUE_SIZE 10

// Calibration settings
#define CALIBRATION_SAMPLES 1000
#define CALIBRATION_DELAY_MS 2

// Debug settings
#define DEBUG_ENABLED true
#define SERIAL_BAUD 115200

#endif