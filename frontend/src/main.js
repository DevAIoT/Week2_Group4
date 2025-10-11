import * as THREE from 'three';
import { GolfSimulator } from './GolfSimulator.js';
import { WebSocketClient } from './WebSocketClient.js';
import { UIManager } from './UIManager.js';

class App {
    constructor() {
        this.simulator = null;
        this.websocketClient = null;
        this.uiManager = null;
        this.isInitialized = false;
    }

    async init() {
        try {
            console.log('Initializing Golf Swing Analyzer...');
            
            // Initialize UI Manager
            this.uiManager = new UIManager();
            
            // Initialize 3D Simulator
            const canvasContainer = document.getElementById('canvas-container');
            this.simulator = new GolfSimulator(canvasContainer);
            await this.simulator.init();
            
            // Initialize WebSocket connection
            this.websocketClient = new WebSocketClient('ws://10.78.111.133:8765');
            this.setupWebSocketHandlers();
            
            // Setup UI event handlers
            this.setupUIHandlers();
            
            // Hide loading screen and show UI
            document.getElementById('loading').style.display = 'none';
            document.getElementById('status-panel').style.display = 'block';
            document.getElementById('metrics-panel').style.display = 'block';
            document.getElementById('trajectory-panel').style.display = 'block';
            
            // Start WebSocket connection
            await this.websocketClient.connect();
            
            this.isInitialized = true;
            console.log('Golf Swing Analyzer initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application: ' + error.message);
        }
    }

    setupWebSocketHandlers() {
        this.websocketClient.onConnect = () => {
            console.log('Connected to server');
            this.uiManager.updateConnectionStatus(true);
        };

        this.websocketClient.onDisconnect = () => {
            console.log('Disconnected from server');
            this.uiManager.updateConnectionStatus(false);
        };

        this.websocketClient.onSystemStatus = (status) => {
            this.uiManager.updateSystemStatus(status);
        };

        this.websocketClient.onSwingMetrics = (metrics) => {
            this.uiManager.updateSwingMetrics(metrics);
        };

        this.websocketClient.onTrajectoryData = (trajectory) => {
            this.uiManager.updateTrajectoryData(trajectory);
            this.simulator.displayTrajectory(trajectory);
        };

        this.websocketClient.onError = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error: ' + error);
        };
    }

    setupUIHandlers() {
        // Clear trajectory button
        document.getElementById('clear-trajectory').addEventListener('click', () => {
            this.simulator.clearTrajectory();
            this.uiManager.clearTrajectoryData();
        });

        // Reset camera button
        document.getElementById('reset-camera').addEventListener('click', () => {
            this.simulator.resetCamera();
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.simulator) {
                this.simulator.handleResize();
            }
        });

        // Handle visibility change (pause/resume when tab is hidden)
        document.addEventListener('visibilitychange', () => {
            if (this.simulator) {
                if (document.hidden) {
                    this.simulator.pause();
                } else {
                    this.simulator.resume();
                }
            }
        });
    }

    showError(message) {
        const loading = document.getElementById('loading');
        loading.innerHTML = `
            <div style="color: #f44336;">
                <h3>Error</h3>
                <p>${message}</p>
                <button onclick="location.reload()">Reload</button>
            </div>
        `;
        loading.style.display = 'block';
    }

    // Public methods for debugging
    getSimulator() {
        return this.simulator;
    }

    getWebSocketClient() {
        return this.websocketClient;
    }

    getUIManager() {
        return this.uiManager;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const app = new App();
    await app.init();
    
    // Make app globally available for debugging
    window.golfApp = app;
});

// Handle any unhandled errors
window.addEventListener('error', (event) => {
    console.error('Unhandled error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});