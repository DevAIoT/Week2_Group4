export class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.reconnectTimer = null;
        
        // Event handlers
        this.onConnect = null;
        this.onDisconnect = null;
        this.onSystemStatus = null;
        this.onSwingMetrics = null;
        this.onTrajectoryData = null;
        this.onError = null;
    }

    async connect() {
        try {
            console.log(`Connecting to WebSocket: ${this.url}`);
            
            this.websocket = new WebSocket(this.url);
            
            this.websocket.onopen = this.handleOpen.bind(this);
            this.websocket.onmessage = this.handleMessage.bind(this);
            this.websocket.onclose = this.handleClose.bind(this);
            this.websocket.onerror = this.handleError.bind(this);
            
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, 10000);
                
                this.websocket.onopen = (event) => {
                    clearTimeout(timeout);
                    this.handleOpen(event);
                    resolve();
                };
                
                this.websocket.onerror = (event) => {
                    clearTimeout(timeout);
                    reject(new Error('Connection failed'));
                };
            });
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            throw error;
        }
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    handleOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        
        if (this.onConnect) {
            this.onConnect();
        }
        
        // Send initial handshake
        this.send({
            type: 'handshake',
            client: 'golf-simulator-frontend',
            version: '1.0.0'
        });
    }

    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data.type);
            
            switch (data.type) {
                case 'system_status':
                    if (this.onSystemStatus) {
                        this.onSystemStatus(data.payload);
                    }
                    break;
                    
                case 'swing_metrics':
                    if (this.onSwingMetrics) {
                        this.onSwingMetrics(data.payload);
                    }
                    break;
                    
                case 'trajectory_data':
                    if (this.onTrajectoryData) {
                        this.onTrajectoryData(data.payload);
                    }
                    break;
                    
                case 'imu_data':
                    // Real-time IMU data - could be used for live visualization
                    // For now, we'll just log it to avoid spam
                    break;
                    
                case 'error':
                    console.error('Server error:', data.payload);
                    if (this.onError) {
                        this.onError(data.payload);
                    }
                    break;
                    
                case 'ping':
                    // Respond to ping with pong
                    this.send({ type: 'pong', timestamp: Date.now() });
                    break;
                    
                default:
                    console.warn('Unknown message type:', data.type);
            }
            
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    handleClose(event) {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        
        if (this.onDisconnect) {
            this.onDisconnect();
        }
        
        // Attempt to reconnect unless it was a clean close
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }

    handleError(event) {
        console.error('WebSocket error:', event);
        
        if (this.onError) {
            this.onError('WebSocket connection error');
        }
    }

    scheduleReconnect() {
        if (this.reconnectTimer) {
            return; // Already scheduled
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
        
        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
        
        this.reconnectTimer = setTimeout(async () => {
            this.reconnectTimer = null;
            
            try {
                await this.connect();
            } catch (error) {
                console.error('Reconnect failed:', error);
                
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.scheduleReconnect();
                } else {
                    console.error('Max reconnect attempts reached');
                    if (this.onError) {
                        this.onError('Connection lost - max retry attempts reached');
                    }
                }
            }
        }, delay);
    }

    send(data) {
        if (!this.isConnected || !this.websocket) {
            console.warn('WebSocket not connected, cannot send data');
            return false;
        }
        
        try {
            const message = JSON.stringify(data);
            this.websocket.send(message);
            return true;
        } catch (error) {
            console.error('Error sending WebSocket message:', error);
            return false;
        }
    }

    // Public methods for sending specific types of messages
    requestSystemStatus() {
        return this.send({
            type: 'request_system_status',
            timestamp: Date.now()
        });
    }

    requestCalibration() {
        return this.send({
            type: 'request_calibration',
            timestamp: Date.now()
        });
    }

    clearTrajectory() {
        return this.send({
            type: 'clear_trajectory',
            timestamp: Date.now()
        });
    }

    // Utility methods
    getConnectionState() {
        if (!this.websocket) {
            return 'CLOSED';
        }
        
        switch (this.websocket.readyState) {
            case WebSocket.CONNECTING: return 'CONNECTING';
            case WebSocket.OPEN: return 'OPEN';
            case WebSocket.CLOSING: return 'CLOSING';
            case WebSocket.CLOSED: return 'CLOSED';
            default: return 'UNKNOWN';
        }
    }

    isReady() {
        return this.isConnected && this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
}