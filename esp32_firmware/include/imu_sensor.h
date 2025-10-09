#ifndef IMU_SENSOR_H
#define IMU_SENSOR_H

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

struct IMUData {
    float accel_x, accel_y, accel_z;    // m/s²
    float gyro_x, gyro_y, gyro_z;       // rad/s
    uint32_t timestamp;                  // milliseconds
    uint8_t checksum;                    // data integrity
};

struct CalibrationData {
    float accel_offset_x, accel_offset_y, accel_offset_z;
    float gyro_offset_x, gyro_offset_y, gyro_offset_z;
    bool is_calibrated;
};

class IMUSensor {
private:
    Adafruit_MPU6050 mpu;
    CalibrationData calibration;
    uint32_t last_sample_time;
    
    uint8_t calculateChecksum(const IMUData& data);
    
public:
    bool initialize();
    bool calibrate();
    bool readData(IMUData& data);
    bool isDataReady();
    CalibrationData getCalibration() const;
    void setCalibration(const CalibrationData& cal);
    void printCalibrationData();
};

#endif