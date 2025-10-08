#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLECharacteristic.h>
#include <MPU6050.h>

// BLE Configuration
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// IMU Configuration
MPU6050 mpu;
const int MPU_ADDR = 0x68;

BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
bool deviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      // Restart advertising after disconnection
      BLEAdvertising *pAdvertising = pServer->getAdvertising();
      pAdvertising->start();
    }
};

void setup() {
  Serial.begin(115200);
  
  // Initialize IMU
  Wire.begin();
  while (!mpu.begin(MPU_ADDR)) {
    Serial.println("Could not find MPU6050 sensor!");
    delay(1000);
  }
  mpu.calibrateGyro();
  mpu.calibrateAccel();
  mpu.setGyroOffset(0, 0, 0);
  mpu.setAccelOffset(0, 0, 0);
  
  // Initialize BLE
  BLEDevice::init("ESP32-IMU");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("Waiting for a client connection to notify...");
}

void loop() {
  if (deviceConnected) {
    // Read IMU data
    mpu.update();
    
    float ax = mpu.getAccelX();
    float ay = mpu.getAccelY();
    float az = mpu.getAccelZ();
    
    float gx = mpu.getGyroX();
    float gy = mpu.getGyroY();
    float gz = mpu.getGyroZ();
    
    float angleX = mpu.getAngleX();
    float angleY = mpu.getAngleY();
    float angleZ = mpu.getAngleZ();
    
    // Format data as CSV string
    String data = String(ax) + "," + 
                  String(ay) + "," + 
                  String(az) + "," +
                  String(gx) + "," + 
                  String(gy) + "," + 
                  String(gz) + "," +
                  String(angleX) + "," + 
                  String(angleY) + "," + 
                  String(angleZ);
    
    // Send data via BLE
    pCharacteristic->setValue(data.c_str());
    pCharacteristic->notify();
    
    // Debug output
    Serial.println("Sent: " + data);
  }
  delay(50); // Adjust sampling rate as needed
}
