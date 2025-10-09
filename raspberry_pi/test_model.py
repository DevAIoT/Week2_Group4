#!/usr/bin/env python3
"""
Test the ML model with different input values to see if it's working correctly
"""

import sys
import os
sys.path.append('src')

import joblib
import numpy as np
from pathlib import Path

def test_model():
    """Test the model with various input patterns"""
    
    # Load model and scaler
    models_path = Path("models")
    model = joblib.load(models_path / "swing_model.pkl")
    scaler = joblib.load(models_path / "scaler.pkl")
    
    print("Model classes:", model.classes_)
    print("\nTesting different input patterns:\n")
    
    # Test 1: All zeros (idle)
    test_idle = np.array([[0, 0, 0, 0, 0, 0]])
    test_idle_scaled = scaler.transform(test_idle)
    pred = model.predict(test_idle_scaled)[0]
    probs = model.predict_proba(test_idle_scaled)[0]
    prob_dict = {cls: prob for cls, prob in zip(model.classes_, probs)}
    print(f"Test 1 - All zeros (expecting idle): {pred}")
    print(f"  Probabilities: {prob_dict}")
    print()
    
    # Test 2: Small values (slow)
    test_slow = np.array([[100, 100, 100, 10, 10, 10]])
    test_slow_scaled = scaler.transform(test_slow)
    pred = model.predict(test_slow_scaled)[0]
    probs = model.predict_proba(test_slow_scaled)[0]
    prob_dict = {cls: prob for cls, prob in zip(model.classes_, probs)}
    print(f"Test 2 - Small values (expecting slow): {pred}")
    print(f"  Probabilities: {prob_dict}")
    print()
    
    # Test 3: Large values (fast)
    test_fast = np.array([[5000, 5000, 5000, 500, 500, 500]])
    test_fast_scaled = scaler.transform(test_fast)
    pred = model.predict(test_fast_scaled)[0]
    probs = model.predict_proba(test_fast_scaled)[0]
    prob_dict = {cls: prob for cls, prob in zip(model.classes_, probs)}
    print(f"Test 3 - Large values (expecting fast): {pred}")
    print(f"  Probabilities: {prob_dict}")
    print()
    
    # Test 4: Your actual ESP32 data pattern
    # Based on the log: '-1376,13384,9560,-307,136,-35'
    test_esp32 = np.array([[-1376, 13384, 9560, -307, 136, -35]])
    test_esp32_scaled = scaler.transform(test_esp32)
    pred = model.predict(test_esp32_scaled)[0]
    probs = model.predict_proba(test_esp32_scaled)[0]
    prob_dict = {cls: prob for cls, prob in zip(model.classes_, probs)}
    print(f"Test 4 - Your ESP32 data: {pred}")
    print(f"  Probabilities: {prob_dict}")
    print()
    
    # Test 5: Asymmetric values (left/right)
    test_left = np.array([[-2000, 1000, 1000, -200, 50, 50]])
    test_left_scaled = scaler.transform(test_left)
    pred = model.predict(test_left_scaled)[0]
    probs = model.predict_proba(test_left_scaled)[0]
    prob_dict = {cls: prob for cls, prob in zip(model.classes_, probs)}
    print(f"Test 5 - Asymmetric left: {pred}")
    print(f"  Probabilities: {prob_dict}")
    print()

if __name__ == "__main__":
    test_model()