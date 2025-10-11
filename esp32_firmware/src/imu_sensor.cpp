#include "imu_sensor.h"
#include "config.h"

bool IMUSensor::initialize() {
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        return false;
    }
    
    // Configure accelerometer range
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    
    // Configure gyroscope range  
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    
    // Configure filter bandwidth
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    
    // Initialize calibration
    calibration.is_calibrated = false;
    last_sample_time = 0;
    
    Serial.println("MPU6050 initialized successfully");
    return true;
}

bool IMUSensor::calibrate() {
    Serial.println("Starting IMU calibration...");
    Serial.println("Keep the sensor stationary for 10 seconds");
    
    float accel_sum_x = 0, accel_sum_y = 0, accel_sum_z = 0;
    float gyro_sum_x = 0, gyro_sum_y = 0, gyro_sum_z = 0;
    
    for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
        sensors_event_t accel, gyro, temp;
        mpu.getEvent(&accel, &gyro, &temp);
        
        accel_sum_x += accel.acceleration.x;
        accel_sum_y += accel.acceleration.y;
        accel_sum_z += accel.acceleration.z;
        
        gyro_sum_x += gyro.gyro.x;
        gyro_sum_y += gyro.gyro.y;
        gyro_sum_z += gyro.gyro.z;
        
        delay(CALIBRATION_DELAY_MS);
        
        if (i % 100 == 0) {
            Serial.print("Calibration progress: ");
            Serial.print((i * 100) / CALIBRATION_SAMPLES);
            Serial.println("%");
        }
    }
    
    // Calculate offsets
    calibration.accel_offset_x = accel_sum_x / CALIBRATION_SAMPLES;
    calibration.accel_offset_y = accel_sum_y / CALIBRATION_SAMPLES;
    calibration.accel_offset_z = (accel_sum_z / CALIBRATION_SAMPLES) - 9.81f; // Remove gravity
    
    calibration.gyro_offset_x = gyro_sum_x / CALIBRATION_SAMPLES;
    calibration.gyro_offset_y = gyro_sum_y / CALIBRATION_SAMPLES;
    calibration.gyro_offset_z = gyro_sum_z / CALIBRATION_SAMPLES;
    
    calibration.is_calibrated = true;
    
    Serial.println("Calibration completed!");
    printCalibrationData();
    
    return true;
}

bool IMUSensor::readData(IMUData& data) {
    if (!isDataReady()) {
        return false;
    }
    
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    
    // Apply calibration offsets
    data.accel_x = accel.acceleration.x - calibration.accel_offset_x;
    data.accel_y = accel.acceleration.y - calibration.accel_offset_y;
    data.accel_z = accel.acceleration.z - calibration.accel_offset_z;
    
    data.gyro_x = gyro.gyro.x - calibration.gyro_offset_x;
    data.gyro_y = gyro.gyro.y - calibration.gyro_offset_y;
    data.gyro_z = gyro.gyro.z - calibration.gyro_offset_z;
    
    data.timestamp = millis();
    data.checksum = calculateChecksum(data);
    
    last_sample_time = data.timestamp;
    
    return true;
}

bool IMUSensor::isDataReady() {
    uint32_t current_time = millis();
    uint32_t sample_interval = 1000 / SAMPLE_RATE_HZ;
    
    return (current_time - last_sample_time) >= sample_interval;
}

uint8_t IMUSensor::calculateChecksum(const IMUData& data) {
    uint8_t checksum = 0;
    uint8_t* bytes = (uint8_t*)&data;
    
    for (int i = 0; i < sizeof(IMUData) - 1; i++) {
        checksum ^= bytes[i];
    }
    
    return checksum;
}

CalibrationData IMUSensor::getCalibration() const {
    return calibration;
}

void IMUSensor::setCalibration(const CalibrationData& cal) {
    calibration = cal;
}

void IMUSensor::printCalibrationData() {
    Serial.println("=== Calibration Data ===");
    Serial.printf("Accel offsets: X=%.3f, Y=%.3f, Z=%.3f\n", 
                  calibration.accel_offset_x, 
                  calibration.accel_offset_y, 
                  calibration.accel_offset_z);
    Serial.printf("Gyro offsets: X=%.3f, Y=%.3f, Z=%.3f\n", 
                  calibration.gyro_offset_x, 
                  calibration.gyro_offset_y, 
                  calibration.gyro_offset_z);
    Serial.println("========================");
}