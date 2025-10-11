#include "ble_server.h"

void BLEServerCallbacks::onConnect(NimBLEServer* pServer) {
    Serial.println("Client connected");
}

void BLEServerCallbacks::onDisconnect(NimBLEServer* pServer) {
    Serial.println("Client disconnected - starting advertising");
    NimBLEDevice::startAdvertising();
}

bool BLEServer::initialize() {
    Serial.println("Initializing BLE server...");
    
    NimBLEDevice::init(DEVICE_NAME);
    NimBLEDevice::setPower(ESP_PWR_LVL_P9);
    
    pServer = NimBLEDevice::createServer();
    pCallbacks = new BLEServerCallbacks();
    pServer->setCallbacks(pCallbacks);
    
    pService = pServer->createService(SERVICE_UUID);
    
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        NIMBLE_PROPERTY::READ | 
        NIMBLE_PROPERTY::WRITE | 
        NIMBLE_PROPERTY::NOTIFY |
        NIMBLE_PROPERTY::INDICATE
    );
    
    pCharacteristic->setValue("Golf Swing Analyzer Ready");
    
    pService->start();
    
    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0);
    
    device_connected = false;
    old_device_connected = false;
    
    startAdvertising();
    
    Serial.println("BLE server initialized and advertising");
    return true;
}

bool BLEServer::isConnected() const {
    return pServer->getConnectedCount() > 0;
}

bool BLEServer::sendData(const IMUData& data) {
    if (!isConnected()) {
        return false;
    }
    
    uint8_t* dataBytes = (uint8_t*)&data;
    pCharacteristic->setValue(dataBytes, sizeof(IMUData));
    pCharacteristic->notify();
    
    return true;
}

void BLEServer::handleConnections() {
    device_connected = isConnected();
    
    if (!device_connected && old_device_connected) {
        delay(500);
        startAdvertising();
        Serial.println("Restarting advertising");
        old_device_connected = device_connected;
    }
    
    if (device_connected && !old_device_connected) {
        old_device_connected = device_connected;
    }
}

void BLEServer::startAdvertising() {
    NimBLEDevice::startAdvertising();
    Serial.println("Waiting for client connection...");
}

void BLEServer::stopAdvertising() {
    NimBLEDevice::stopAdvertising();
}