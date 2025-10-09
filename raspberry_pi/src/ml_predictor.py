"""
Machine Learning Predictor for Golf Swing Analysis
Integrates the trained ML model to predict swing types from IMU data
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from models import IMUData

class MLPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.swing_classes = ['fast', 'idle', 'left', 'medium', 'right', 'slow']
        
        # Data collection for swing analysis
        self.swing_data_buffer = []
        self.max_buffer_size = 200  # Store last 200 IMU readings (~4 seconds at 50Hz)
        self.min_swing_samples = 50  # Minimum samples needed for prediction
        
        self.load_models()
    
    def load_models(self):
        """Load the trained ML model and scaler"""
        try:
            models_path = Path(__file__).parent.parent / "models"
            
            model_path = models_path / "swing_model.pkl"
            scaler_path = models_path / "scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                logger.error(f"Model files not found in {models_path}")
                logger.error(f"Expected: {model_path} and {scaler_path}")
                return
            
            logger.info(f"Loading ML model from {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            logger.info("ML model and scaler loaded successfully")
            logger.info(f"Model classes: {self.model.classes_}")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.model = None
            self.scaler = None
    
    def add_imu_data(self, imu_data: IMUData):
        """Add IMU data to the buffer for swing analysis"""
        if not self.is_ready():
            return
        
        # Convert IMU data to array format expected by model
        data_point = [
            imu_data.accel_x,
            imu_data.accel_y,
            imu_data.accel_z,
            imu_data.gyro_x,
            imu_data.gyro_y,
            imu_data.gyro_z
        ]
        
        self.swing_data_buffer.append(data_point)
        
        # Keep buffer size manageable
        if len(self.swing_data_buffer) > self.max_buffer_size:
            self.swing_data_buffer.pop(0)
    
    def extract_features_from_buffer(self, data: List[List[float]]) -> np.ndarray:
        """Extract mean features from IMU data buffer (same as training)"""
        if len(data) < self.min_swing_samples:
            return None
        
        # Convert to numpy array for easier manipulation
        data_array = np.array(data)  # Shape: (n_samples, 6) - [ax, ay, az, gx, gy, gz]
        
        # Extract only mean features as used in original training
        # This matches Test_Model2.py: df.mean().to_numpy().reshape(1, -1)
        features = np.mean(data_array, axis=0)  # Mean of each column: [ax_mean, ay_mean, az_mean, gx_mean, gy_mean, gz_mean]
        
        return features.reshape(1, -1)
    
    def predict_swing_simple(self, imu_data: IMUData) -> Optional[Dict]:
        """Simple real-time prediction using single IMU reading"""
        if not self.is_ready():
            return None
        
        try:
            # Prepare single data point
            X = np.array([[
                imu_data.accel_x,
                imu_data.accel_y,
                imu_data.accel_z,
                imu_data.gyro_x,
                imu_data.gyro_y,
                imu_data.gyro_z
            ]])
            
            # Scale the data
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get confidence score
            confidence = max(probabilities) * 100
            
            # Create probability dictionary
            prob_dict = {
                cls: float(prob) for cls, prob in zip(self.model.classes_, probabilities)
            }
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'probabilities': prob_dict,
                'timestamp': imu_data.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error in swing prediction: {e}")
            return None
    
    def predict_swing_buffered(self, use_last_n_samples: Optional[int] = None) -> Optional[Dict]:
        """Predict swing using buffered data for better accuracy"""
        if not self.is_ready():
            return None
        
        if len(self.swing_data_buffer) < self.min_swing_samples:
            logger.debug(f"Not enough data for prediction: {len(self.swing_data_buffer)} < {self.min_swing_samples}")
            return None
        
        try:
            # Use specified number of samples or all available
            data_to_analyze = self.swing_data_buffer
            if use_last_n_samples:
                data_to_analyze = self.swing_data_buffer[-use_last_n_samples:]
            
            # Extract features from buffer
            features = self.extract_features_from_buffer(data_to_analyze)
            if features is None:
                return None
            
            # Scale the features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get confidence score
            confidence = max(probabilities) * 100
            
            # Create probability dictionary
            prob_dict = {
                cls: float(prob) for cls, prob in zip(self.model.classes_, probabilities)
            }
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'probabilities': prob_dict,
                'samples_used': len(data_to_analyze),
                'buffer_size': len(self.swing_data_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error in buffered swing prediction: {e}")
            return None
    
    def clear_buffer(self):
        """Clear the swing data buffer"""
        self.swing_data_buffer.clear()
        logger.debug("Swing data buffer cleared")
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the current buffer"""
        if not self.swing_data_buffer:
            return {'size': 0, 'ready_for_prediction': False}
        
        return {
            'size': len(self.swing_data_buffer),
            'ready_for_prediction': len(self.swing_data_buffer) >= self.min_swing_samples,
            'max_size': self.max_buffer_size,
            'min_samples_needed': self.min_swing_samples
        }
    
    def is_ready(self) -> bool:
        """Check if the ML predictor is ready to make predictions"""
        return self.model is not None and self.scaler is not None
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_ready():
            return {'loaded': False}
        
        return {
            'loaded': True,
            'classes': list(self.model.classes_),
            'n_features': self.scaler.n_features_in_,
            'n_samples_seen': int(self.scaler.n_samples_seen_),
            'buffer_stats': self.get_buffer_stats()
        }