export class UIManager {
    constructor() {
        this.elements = {
            // Status elements
            bleStatus: document.getElementById('ble-status'),
            imuStatus: document.getElementById('imu-status'),
            websocketStatus: document.getElementById('websocket-status'),
            lastData: document.getElementById('last-data'),
            swingPhase: document.getElementById('swing-phase'),
            connectionStatus: document.getElementById('connection-status'),
            
            // Swing metrics elements
            maxSpeed: document.getElementById('max-speed'),
            impactSpeed: document.getElementById('impact-speed'),
            swingAngle: document.getElementById('swing-angle'),
            attackAngle: document.getElementById('attack-angle'),
            tempo: document.getElementById('tempo'),
            swingPath: document.getElementById('swing-path'),
            
            // Trajectory elements
            ballSpeed: document.getElementById('ball-speed'),
            launchAngle: document.getElementById('launch-angle'),
            totalDistance: document.getElementById('total-distance'),
            maxHeight: document.getElementById('max-height'),
            flightTime: document.getElementById('flight-time'),
            spinRate: document.getElementById('spin-rate')
        };
        
        this.lastDataTime = null;
        this.updateLastDataTimer = null;
        
        this.startLastDataTimer();
    }

    updateConnectionStatus(isConnected) {
        const status = this.elements.connectionStatus;
        const wsStatus = this.elements.websocketStatus;
        
        if (isConnected) {
            status.textContent = 'System Online';
            status.className = 'connection-connected';
            wsStatus.textContent = 'Connected';
            wsStatus.className = 'status-value status-connected';
        } else {
            status.textContent = 'System Offline';
            status.className = 'connection-disconnected';
            wsStatus.textContent = 'Disconnected';
            wsStatus.className = 'status-value status-disconnected';
        }
    }

    updateSystemStatus(status) {
        // Update BLE status
        if (status.ble_connected) {
            this.elements.bleStatus.textContent = 'Connected';
            this.elements.bleStatus.className = 'status-value status-connected';
        } else {
            this.elements.bleStatus.textContent = 'Disconnected';
            this.elements.bleStatus.className = 'status-value status-disconnected';
        }
        
        // Update IMU status
        if (status.imu_active) {
            this.elements.imuStatus.textContent = 'Active';
            this.elements.imuStatus.className = 'status-value status-connected';
        } else {
            this.elements.imuStatus.textContent = 'Inactive';
            this.elements.imuStatus.className = 'status-value status-disconnected';
        }
        
        // Update swing phase
        this.updateSwingPhase(status.swing_phase);
        
        // Update last data time
        if (status.last_data_time) {
            this.lastDataTime = new Date(status.last_data_time * 1000);
        }
    }

    updateSwingPhase(phase) {
        const phaseElement = this.elements.swingPhase;
        
        // Remove all phase classes
        phaseElement.className = 'swing-phase';
        
        // Add appropriate phase class and text
        switch (phase.toLowerCase()) {
            case 'idle':
                phaseElement.classList.add('phase-idle');
                phaseElement.textContent = 'Idle';
                break;
            case 'address':
                phaseElement.classList.add('phase-address');
                phaseElement.textContent = 'Address';
                break;
            case 'backswing':
                phaseElement.classList.add('phase-backswing');
                phaseElement.textContent = 'Backswing';
                break;
            case 'downswing':
                phaseElement.classList.add('phase-downswing');
                phaseElement.textContent = 'Downswing';
                break;
            case 'impact':
                phaseElement.classList.add('phase-impact');
                phaseElement.textContent = 'Impact';
                break;
            case 'follow_through':
            case 'follow-through':
                phaseElement.classList.add('phase-follow-through');
                phaseElement.textContent = 'Follow Through';
                break;
            default:
                phaseElement.classList.add('phase-idle');
                phaseElement.textContent = 'Unknown';
        }
    }

    updateSwingMetrics(metrics) {
        if (!metrics) return;
        
        this.elements.maxSpeed.textContent = `${metrics.max_speed?.toFixed(1) || '0'} m/s`;
        this.elements.impactSpeed.textContent = `${metrics.impact_speed?.toFixed(1) || '0'} m/s`;
        this.elements.swingAngle.textContent = `${metrics.swing_angle?.toFixed(1) || '0'}°`;
        this.elements.attackAngle.textContent = `${metrics.attack_angle?.toFixed(1) || '0'}°`;
        this.elements.tempo.textContent = `${metrics.tempo?.toFixed(2) || '0'}s`;
        this.elements.swingPath.textContent = metrics.swing_path || 'Unknown';
        
        // Add visual feedback for good/bad metrics
        this.updateMetricColor(this.elements.maxSpeed, metrics.max_speed, 15, 30); // Good speed range
        this.updateMetricColor(this.elements.tempo, metrics.tempo, 1.0, 2.0); // Good tempo range
    }

    updateTrajectoryData(trajectory) {
        if (!trajectory) return;
        
        // Update launch conditions if available
        if (trajectory.launch_conditions) {
            const lc = trajectory.launch_conditions;
            this.elements.ballSpeed.textContent = `${lc.ball_speed?.toFixed(1) || '0'} m/s`;
            this.elements.launchAngle.textContent = `${lc.launch_angle?.toFixed(1) || '0'}°`;
            this.elements.spinRate.textContent = `${Math.round(lc.spin_rate) || '0'} rpm`;
        }
        
        // Update trajectory results
        this.elements.totalDistance.textContent = `${trajectory.total_distance?.toFixed(1) || '0'} m`;
        this.elements.maxHeight.textContent = `${trajectory.max_height?.toFixed(1) || '0'} m`;
        this.elements.flightTime.textContent = `${trajectory.flight_time?.toFixed(1) || '0'}s`;
        
        // Add visual feedback for distance
        this.updateMetricColor(this.elements.totalDistance, trajectory.total_distance, 100, 200);
    }

    updateMetricColor(element, value, minGood, maxGood) {
        if (!value || !element) return;
        
        element.style.color = '';
        
        if (value >= minGood && value <= maxGood) {
            element.style.color = '#4CAF50'; // Green for good
        } else if (value < minGood * 0.5 || value > maxGood * 1.5) {
            element.style.color = '#f44336'; // Red for poor
        } else {
            element.style.color = '#ff9800'; // Orange for fair
        }
    }

    clearTrajectoryData() {
        this.elements.ballSpeed.textContent = '0 m/s';
        this.elements.launchAngle.textContent = '0°';
        this.elements.totalDistance.textContent = '0 m';
        this.elements.maxHeight.textContent = '0 m';
        this.elements.flightTime.textContent = '0s';
        this.elements.spinRate.textContent = '0 rpm';
        
        // Reset colors
        [this.elements.ballSpeed, this.elements.launchAngle, this.elements.totalDistance,
         this.elements.maxHeight, this.elements.flightTime, this.elements.spinRate].forEach(el => {
            el.style.color = '';
        });
    }

    clearSwingMetrics() {
        this.elements.maxSpeed.textContent = '0 m/s';
        this.elements.impactSpeed.textContent = '0 m/s';
        this.elements.swingAngle.textContent = '0°';
        this.elements.attackAngle.textContent = '0°';
        this.elements.tempo.textContent = '0s';
        this.elements.swingPath.textContent = 'Unknown';
        
        // Reset colors
        [this.elements.maxSpeed, this.elements.impactSpeed, this.elements.swingAngle,
         this.elements.attackAngle, this.elements.tempo].forEach(el => {
            el.style.color = '';
        });
    }

    startLastDataTimer() {
        this.updateLastDataTimer = setInterval(() => {
            this.updateLastDataDisplay();
        }, 1000);
    }

    updateLastDataDisplay() {
        const lastDataElement = this.elements.lastData;
        
        if (!this.lastDataTime) {
            lastDataElement.textContent = 'Never';
            return;
        }
        
        const now = new Date();
        const diffMs = now - this.lastDataTime;
        const diffSeconds = Math.floor(diffMs / 1000);
        
        if (diffSeconds < 60) {
            lastDataElement.textContent = `${diffSeconds}s ago`;
        } else if (diffSeconds < 3600) {
            const minutes = Math.floor(diffSeconds / 60);
            lastDataElement.textContent = `${minutes}m ago`;
        } else {
            const hours = Math.floor(diffSeconds / 3600);
            lastDataElement.textContent = `${hours}h ago`;
        }
        
        // Change color based on how old the data is
        if (diffSeconds < 5) {
            lastDataElement.style.color = '#4CAF50'; // Green - recent
        } else if (diffSeconds < 30) {
            lastDataElement.style.color = '#ff9800'; // Orange - somewhat old
        } else {
            lastDataElement.style.color = '#f44336'; // Red - old
        }
    }

    showNotification(message, type = 'info', duration = 3000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#f44336' : type === 'success' ? '#4CAF50' : '#2196F3'};
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            z-index: 1000;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = message;
        
        // Add animation CSS if not already present
        if (!document.querySelector('#notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(notification);
        
        // Remove notification after duration
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }

    // Cleanup method
    destroy() {
        if (this.updateLastDataTimer) {
            clearInterval(this.updateLastDataTimer);
        }
    }
}