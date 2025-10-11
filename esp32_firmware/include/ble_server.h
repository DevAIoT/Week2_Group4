#ifndef BLE_SERVER_H
#define BLE_SERVER_H

#include <NimBLEDevice.h>
#include <NimBLEServer.h>
#include <NimBLEUtils.h>
#include <NimBLE2904.h>
#include "imu_sensor.h"
#include "config.h"

class BLEServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer) override;
    void onDisconnect(NimBLEServer* pServer) override;
};

class BLEServer {
private:
    NimBLEServer* pServer;
    NimBLEService* pService;
    NimBLECharacteristic* pCharacteristic;
    BLEServerCallbacks* pCallbacks;
    bool device_connected;
    bool old_device_connected;
    
public:
    bool initialize();
    bool isConnected() const;
    bool sendData(const IMUData& data);
    void handleConnections();
    void startAdvertising();
    void stopAdvertising();
};

#endif