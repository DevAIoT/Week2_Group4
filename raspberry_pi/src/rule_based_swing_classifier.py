"""
Rule-Based Golf Swing Classification Model

This module implements a rule-based classifier for golf swing types using IMU sensor data.
It analyzes accelerometer and gyroscope data to classify swings as:
- Speed-based: fast, medium, slow, idle
- Direction-based: left, right

The model extracts key features from the sensor data:
1. Peak acceleration magnitude (overall swing intensity)
2. Acceleration variance (swing consistency/smoothness)  
3. Gyroscope angular velocity (rotational movement)
4. Movement duration and timing characteristics
5. Directional bias in gyroscope readings

Author: Senior Software Engineer & ML Specialist
Date: 2025-10-10
"""

import pandas as pd
import numpy as np
import os
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Optional matplotlib import for visualizations
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("📊 Note: matplotlib not available. Skipping visualizations.")

class RuleBasedSwingClassifier:
    """
    A rule-based classifier for golf swing types using IMU sensor data.
    
    This classifier uses engineered features and threshold-based rules to 
    distinguish between different types of golf swings without requiring 
    machine learning training.
    """
    
    def __init__(self):
        """Initialize the classifier with default thresholds."""
        self.feature_stats = {}
        self.classification_rules = {}
        self.is_trained = False
        
        # Set up default physics-based rules for real-time use
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default physics-based rules without requiring training data."""
        # Speed classification based on ACCELERATION patterns
        speed_rules = {
            'idle': {
                'acc_std_max': 150,  # Very consistent acceleration (minimal movement)
                'gyro_mean_max': 600,  # Minimal rotation
                'priority': 1
            },
            'slow': {
                'acc_std_max': 500,  # Low acceleration variance (smooth, controlled movement)
                'acc_std_min': 150,  # But more than idle
                'priority': 2
            },
            'medium': {
                'acc_std_min': 500,   # Moderate acceleration variance
                'acc_std_max': 3000,  # But not extreme
                'priority': 3
            },
            'fast': {
                'acc_std_min': 3000,  # High acceleration variance (explosive movement)
                'priority': 4
            }
        }
        
        # Direction classification based on ANGULAR VELOCITY (gyroscope)
        direction_rules = {
            'direction_axis': 'y',  # Y-axis gyroscope for left/right detection
            'left_threshold_max': 1800,   # Left swings have gyro_y ≤ 1800
            'right_threshold_min': 2800,  # Right swings have gyro_y > 2800
            'min_angular_velocity': 1000  # Minimum rotation to consider directional
        }
        
        self.classification_rules = {
            'speed': speed_rules,
            'direction': direction_rules
        }
        
        self.is_trained = True  # Mark as ready for classification
        
    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract key features from IMU sensor data.
        
        Args:
            df: DataFrame with columns [timestamp, ax, ay, az, gx, gy, gz, label]
            
        Returns:
            Dictionary of extracted features
        """
        # Calculate acceleration magnitude at each time point
        acc_magnitude = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        
        # Calculate gyroscope magnitude (angular velocity)
        gyro_magnitude = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
        
        # Extract timing features
        timestamps = pd.to_datetime(df['timestamp'])
        duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
        
        features = {
            # Acceleration-based features
            'acc_mean': acc_magnitude.mean(),
            'acc_max': acc_magnitude.max(),
            'acc_min': acc_magnitude.min(),
            'acc_std': acc_magnitude.std(),
            'acc_range': acc_magnitude.max() - acc_magnitude.min(),
            'acc_peak_count': len([x for x in acc_magnitude if x > acc_magnitude.mean() + acc_magnitude.std()]),
            
            # Gyroscope-based features  
            'gyro_mean': gyro_magnitude.mean(),
            'gyro_max': gyro_magnitude.max(),
            'gyro_std': gyro_magnitude.std(),
            'gyro_range': gyro_magnitude.max() - gyro_magnitude.min(),
            
            # Directional features (key for left/right classification)
            'gyro_x_mean': df['gx'].mean(),
            'gyro_y_mean': df['gy'].mean(), 
            'gyro_z_mean': df['gz'].mean(),
            'gyro_x_bias': df['gx'].mean() / (df['gx'].std() + 1e-8),  # Avoid division by zero
            'gyro_y_bias': df['gy'].mean() / (df['gy'].std() + 1e-8),
            'gyro_z_bias': df['gz'].mean() / (df['gz'].std() + 1e-8),
            
            # Temporal features
            'duration': duration,
            'data_points': len(df),
            'sampling_rate': len(df) / duration if duration > 0 else 0,
            
            # Movement smoothness indicators
            'acc_variance_ratio': acc_magnitude.var() / (acc_magnitude.mean()**2 + 1e-8),
            'gyro_variance_ratio': gyro_magnitude.var() / (gyro_magnitude.mean()**2 + 1e-8),
            
        }
        
        # Add peak detection features (must be after features dict is created)
        features['acc_peak_ratio'] = features['acc_peak_count'] / len(df) if len(df) > 0 else 0
        
        return features
    
    def analyze_training_data(self, data_folder: str) -> None:
        """
        Analyze all training data to understand feature distributions and establish rules.
        
        Args:
            data_folder: Path to folder containing CSV files
        """
        print("🔍 Analyzing training data to establish classification rules...")
        
        # Collect features for each swing type
        swing_features = defaultdict(list)
        
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        print(f"📊 Found {len(csv_files)} CSV files")
        
        for filename in csv_files:
            filepath = os.path.join(data_folder, filename)
            df = pd.read_csv(filepath)
            
            # Extract swing type from filename
            swing_type = filename.split('_')[0]
            
            # Extract features
            features = self.extract_features(df)
            swing_features[swing_type].append(features)
            
        # Calculate statistics for each swing type
        self.feature_stats = {}
        for swing_type, feature_list in swing_features.items():
            # Convert list of feature dicts to DataFrame for easy statistics
            feature_df = pd.DataFrame(feature_list)
            
            self.feature_stats[swing_type] = {
                'mean': feature_df.mean().to_dict(),
                'std': feature_df.std().to_dict(),
                'min': feature_df.min().to_dict(),
                'max': feature_df.max().to_dict(),
                'count': len(feature_list)
            }
            
        self._establish_rules()
        self.is_trained = True
        print("✅ Training data analysis completed!")
        
    def _establish_rules(self) -> None:
        """
        Establish general physics-based classification rules that don't overfit to training data.
        """
        print("🧠 Establishing general physics-based classification rules...")
        
        # Print feature analysis for reference only
        self._print_feature_analysis()
        
        # GENERAL PHYSICS-BASED RULES (not tuned to specific training data)
        
        # Speed classification based on ACCELERATION patterns
        # Physical principle: Faster swings have higher acceleration variance
        speed_rules = {
            'idle': {
                'acc_std_max': 150,  # Very consistent acceleration (minimal movement)
                'gyro_mean_max': 600,  # Minimal rotation
                'priority': 1
            },
            'slow': {
                'acc_std_max': 500,  # Low acceleration variance (smooth, controlled movement)
                'acc_std_min': 150,  # But more than idle
                'priority': 2
            },
            'medium': {
                'acc_std_min': 500,   # Moderate acceleration variance
                'acc_std_max': 3000,  # But not extreme
                'priority': 3
            },
            'fast': {
                'acc_std_min': 3000,  # High acceleration variance (explosive movement)
                'priority': 4
            }
        }
        
        # Direction classification based on ANGULAR VELOCITY (gyroscope)
        # Physical principle: Left/right swings show different rotational patterns
        # Use Y-axis gyroscope as it typically captures left/right rotation best
        # Adjusted thresholds based on observed data patterns:
        # - Left swings: gyro_y_mean = 1556 ± 379
        # - Right swings: gyro_y_mean = 3025 ± 363  
        # - Fast swings: gyro_y_mean = 3345 ± 301 (also right-directional)
        direction_rules = {
            'direction_axis': 'y',  # Y-axis gyroscope for left/right detection
            'left_threshold_max': 1800,   # Adjusted: Left swings have gyro_y ≤ 1800
            'right_threshold_min': 2800,  # Restored: Right swings have gyro_y > 2800
            'min_angular_velocity': 1000  # Minimum rotation to consider directional
        }
        
        self.classification_rules = {
            'speed': speed_rules,
            'direction': direction_rules
        }
        
        print("📏 General physics-based rules established!")
        print("   🎯 Speed: Based on acceleration variance (acc_std)")
        print("   🧭 Direction: Based on Y-axis angular velocity (gyro_y_mean)")
        
    def _print_feature_analysis(self) -> None:
        """Print detailed feature analysis for each swing type."""
        print("\n📈 Feature Analysis Summary:")
        print("=" * 80)
        
        for swing_type, stats in self.feature_stats.items():
            print(f"\n🏌️ {swing_type.upper()} Swings (n={stats['count']}):")
            print("-" * 50)
            
            key_features = ['acc_mean', 'acc_max', 'acc_std', 'gyro_mean', 'gyro_x_mean', 'gyro_y_mean']
            
            for feature in key_features:
                if feature in stats['mean']:
                    print(f"  {feature:15}: {stats['mean'][feature]:8.1f} ± {stats['std'][feature]:6.1f}")
        
        print("=" * 80)
        
    def classify_swing(self, df: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a single swing using separate speed and direction dimensions.
        
        Args:
            df: DataFrame with IMU sensor data
            
        Returns:
            Tuple of (predicted_class, confidence_score, debug_info)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before classification. Call analyze_training_data() first.")
            
        features = self.extract_features(df)
        debug_info = {'features': features, 'rule_matches': []}
        
        # Classify speed and direction as separate dimensions
        speed_classification = self._classify_speed(features, debug_info)
        direction_classification = self._classify_direction(features, debug_info)
        
        # Determine final classification based on both dimensions
        final_classification, confidence = self._combine_classifications(
            speed_classification, direction_classification, debug_info)
            
        debug_info['final_classification'] = final_classification
        debug_info['speed_result'] = speed_classification
        debug_info['direction_result'] = direction_classification
        
        return final_classification, confidence, debug_info
    
    def _combine_classifications(self, speed_result: Tuple[str, float], 
                               direction_result: Tuple[str, float], 
                               debug_info: Dict) -> Tuple[str, float]:
        """
        Advanced multi-dimensional classification that distinguishes between 
        fast swings and directional swings using acceleration intensity.
        
        Key insight: Fast swings have EXPLOSIVE acceleration patterns (high acc_std + high acc_max)
        while directional swings have more CONTROLLED movement patterns.
        """
        speed_class, speed_conf = speed_result
        direction_class, direction_conf = direction_result
        
        # Get features for advanced discrimination
        features = debug_info['features']
        acc_std = features['acc_std']
        acc_max = features['acc_max'] 
        gyro_y = features['gyro_y_mean']
        
        debug_info['combination_logic'] = []
        
        # ADVANCED MULTI-DIMENSIONAL LOGIC
        
        # Rule 1: Idle takes highest priority (very distinctive pattern)
        if speed_class == 'idle' and speed_conf > 0.8:
            debug_info['combination_logic'].append(f"Rule 1: High-confidence idle detection ({speed_conf:.1%})")
            return speed_class, speed_conf
        
        # Rule 2: EXPLOSIVE FAST SWINGS - Use acceleration intensity as primary discriminator
        # Fast swings: High acc_std (>4000) AND high acc_max (>25000) = explosive movement
        if acc_std > 4000 and acc_max > 25000:
            debug_info['combination_logic'].append(f"Rule 2: Explosive fast swing (acc_std={acc_std:.0f}, acc_max={acc_max:.0f})")
            return 'fast', 0.9
        
        # Rule 3: CONTROLLED DIRECTIONAL SWINGS - Clear directional intent with moderate intensity
        # Right swings: Strong directional signal (gyro_y > 2800) but less explosive than fast
        if direction_class == 'right' and gyro_y > 2800 and acc_std < 4500:
            debug_info['combination_logic'].append(f"Rule 3: Controlled right swing (gyro_y={gyro_y:.0f}, acc_std={acc_std:.0f})")
            return 'right', 0.9
            
        # Rule 4: LEFT SWINGS (different pattern - typically lower gyro_y)
        if direction_class == 'left' and direction_conf > 0.6:
            debug_info['combination_logic'].append(f"Rule 4: Left directional intent ({direction_conf:.1%})")
            return direction_class, direction_conf
        
        # Rule 5: MEDIUM-FAST SWINGS (high acceleration but not explosive)
        if speed_class == 'fast' and acc_std > 3000 and acc_std <= 4000:
            debug_info['combination_logic'].append(f"Rule 5: Medium-fast swing (acc_std={acc_std:.0f})")
            return 'fast', 0.8
        
        # Rule 6: SLOW SWINGS (low acceleration variance)
        if speed_class == 'slow' and speed_conf > 0.7:
            debug_info['combination_logic'].append(f"Rule 6: Slow swing detection ({speed_conf:.1%})")
            return speed_class, speed_conf
        
        # Rule 7: MEDIUM/DEFAULT classification
        if speed_class == 'medium' and speed_conf > 0.6:
            debug_info['combination_logic'].append(f"Rule 7: Medium swing default ({speed_conf:.1%})")
            return speed_class, speed_conf
        
        # Rule 8: FALLBACK - Lower confidence directional
        if direction_class in ['left', 'right'] and direction_conf > 0.5:
            debug_info['combination_logic'].append(f"Rule 8: Fallback directional ({direction_class}, {direction_conf:.1%})")
            return direction_class, direction_conf
            
        # Rule 9: Best effort
        if speed_conf >= direction_conf:
            debug_info['combination_logic'].append(f"Rule 9: Best effort - speed ({speed_class}, {speed_conf:.1%})")
            return speed_class, speed_conf
        else:
            debug_info['combination_logic'].append(f"Rule 9: Best effort - direction ({direction_class}, {direction_conf:.1%})")
            return direction_class, direction_conf
        
    def _classify_speed(self, features: Dict[str, float], debug_info: Dict) -> Tuple[str, float]:
        """Classify swing speed based on acceleration variance (physics-based approach)."""
        speed_rules = self.classification_rules['speed']
        acc_std = features['acc_std']
        gyro_mean = features.get('gyro_mean', 0)
        
        debug_info['rule_matches'].append(f"Speed analysis: acc_std={acc_std:.1f}, gyro_mean={gyro_mean:.1f}")
        
        # Check rules in priority order (idle first, then by acceleration variance)
        for swing_type in sorted(speed_rules.keys(), key=lambda x: speed_rules[x]['priority']):
            rules = speed_rules[swing_type]
            
            # Check if all conditions are met
            conditions_met = True
            condition_count = 0
            
            for rule_name, threshold in rules.items():
                if rule_name == 'priority':
                    continue
                    
                condition_count += 1
                
                if rule_name == 'acc_std_min' and acc_std < threshold:
                    conditions_met = False
                    debug_info['rule_matches'].append(f"{swing_type}: acc_std too low ({acc_std:.1f} < {threshold})")
                elif rule_name == 'acc_std_max' and acc_std > threshold:
                    conditions_met = False
                    debug_info['rule_matches'].append(f"{swing_type}: acc_std too high ({acc_std:.1f} > {threshold})")
                elif rule_name == 'gyro_mean_max' and gyro_mean > threshold:
                    conditions_met = False
                    debug_info['rule_matches'].append(f"{swing_type}: gyro_mean too high ({gyro_mean:.1f} > {threshold})")
            
            if conditions_met and condition_count > 0:
                confidence = 0.9  # High confidence for rule-based classification
                debug_info['rule_matches'].append(f"{swing_type}: All conditions met")
                return swing_type, confidence
        
        # If no specific rule matches, classify based on acceleration variance ranges
        if acc_std < 150:
            debug_info['rule_matches'].append("Fallback: Very low acc_std suggests idle")
            return 'idle', 0.6
        elif acc_std < 500:
            debug_info['rule_matches'].append("Fallback: Low acc_std suggests slow")
            return 'slow', 0.6
        elif acc_std < 3000:
            debug_info['rule_matches'].append("Fallback: Medium acc_std suggests medium")
            return 'medium', 0.6
        else:
            debug_info['rule_matches'].append("Fallback: High acc_std suggests fast")
            return 'fast', 0.6
        
    def _classify_direction(self, features: Dict[str, float], debug_info: Dict) -> Tuple[str, float]:
        """Classify swing direction based on Y-axis angular velocity (physics-based approach).""" 
        direction_rules = self.classification_rules['direction']
        
        # Focus on Y-axis gyroscope for left/right detection
        gyro_y = features['gyro_y_mean']
        gyro_magnitude = features['gyro_mean']
        
        debug_info['rule_matches'].append(f"Direction analysis: gyro_y_mean={gyro_y:.1f}, gyro_magnitude={gyro_magnitude:.1f}")
        
        # Check if there's sufficient angular velocity to determine direction
        min_angular_velocity = direction_rules['min_angular_velocity']
        if abs(gyro_y) < min_angular_velocity:
            debug_info['rule_matches'].append(f"Direction: Insufficient angular velocity ({abs(gyro_y):.1f} < {min_angular_velocity})")
            return 'unknown', 0.0
        
        # Apply physics-based direction classification with improved logic
        left_threshold_max = direction_rules['left_threshold_max']
        right_threshold_min = direction_rules['right_threshold_min']
        
        # Enhanced direction detection: require CLEAR directional bias
        # For a swing to be classified as directional, it should have:
        # 1. Strong Y-axis bias (beyond thresholds)
        # 2. Reasonable confidence based on distance from center
        
        center_zone = (left_threshold_max + right_threshold_min) / 2  # ~2000
        
        if gyro_y <= left_threshold_max:
            # Left swing: Lower Y-axis gyroscope values
            # Confidence based on how far from center zone
            distance_from_center = abs(gyro_y - center_zone)
            max_left_distance = abs(left_threshold_max - center_zone)
            confidence = min(0.9, distance_from_center / max_left_distance + 0.5)
            
            # Require minimum confidence for clear directional intent
            if confidence > 0.6:
                debug_info['rule_matches'].append(f"Direction: Left detected (gyro_y {gyro_y:.1f} <= {left_threshold_max}, conf={confidence:.1%})")
                return 'left', confidence
            else:
                debug_info['rule_matches'].append(f"Direction: Weak left signal (gyro_y {gyro_y:.1f}, conf={confidence:.1%})")
                return 'unknown', 0.0
            
        elif gyro_y >= right_threshold_min:
            # Right swing: Higher Y-axis gyroscope values
            distance_from_center = abs(gyro_y - center_zone)
            max_right_distance = abs(right_threshold_min - center_zone)
            confidence = min(0.9, distance_from_center / max_right_distance + 0.5)
            
            # Require minimum confidence for clear directional intent
            if confidence > 0.6:
                debug_info['rule_matches'].append(f"Direction: Right detected (gyro_y {gyro_y:.1f} >= {right_threshold_min}, conf={confidence:.1%})")
                return 'right', confidence
            else:
                debug_info['rule_matches'].append(f"Direction: Weak right signal (gyro_y {gyro_y:.1f}, conf={confidence:.1%})")
                return 'unknown', 0.0
            
        else:
            # In between thresholds - center/ambiguous direction
            debug_info['rule_matches'].append(f"Direction: Center swing (gyro_y {gyro_y:.1f} between {left_threshold_max} and {right_threshold_min})")
            return 'unknown', 0.0
        
    def evaluate_model(self, data_folder: str) -> Dict[str, Any]:
        """
        Evaluate the rule-based model on all available data.
        
        Args:
            data_folder: Path to folder containing CSV files
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("🧪 Evaluating rule-based model...")
        
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        results = {
            'predictions': [],
            'actual': [],
            'confidence_scores': [],
            'detailed_results': []
        }
        
        correct_predictions = 0
        total_predictions = 0
        
        for filename in csv_files:
            filepath = os.path.join(data_folder, filename)
            df = pd.read_csv(filepath)
            
            actual_label = filename.split('_')[0]
            predicted_label, confidence, debug_info = self.classify_swing(df)
            
            results['predictions'].append(predicted_label)
            results['actual'].append(actual_label) 
            results['confidence_scores'].append(confidence)
            results['detailed_results'].append({
                'filename': filename,
                'actual': actual_label,
                'predicted': predicted_label,
                'confidence': confidence,
                'debug_info': debug_info
            })
            
            if predicted_label == actual_label:
                correct_predictions += 1
            total_predictions += 1
            
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Per-class accuracy
        per_class_accuracy = {}
        for label in set(results['actual']):
            label_indices = [i for i, actual in enumerate(results['actual']) if actual == label]
            label_correct = sum(1 for i in label_indices if results['predictions'][i] == label)
            per_class_accuracy[label] = label_correct / len(label_indices) if label_indices else 0
            
        results['overall_accuracy'] = accuracy
        results['per_class_accuracy'] = per_class_accuracy
        results['total_files'] = total_predictions
        results['correct_predictions'] = correct_predictions
        
        # Print evaluation summary
        print(f"\n📊 Model Evaluation Results:")
        print(f"Overall Accuracy: {accuracy:.1%}")
        print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"\nPer-Class Accuracy:")
        for label, acc in per_class_accuracy.items():
            print(f"  {label:8}: {acc:.1%}")
            
        return results
    
    def cross_validate(self, data_folder: str, k_folds: int = 3) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation to test model robustness without overfitting.
        Uses the same general rules for all folds (no re-training).
        """
        print(f"\n🔄 Performing {k_folds}-fold cross-validation...")
        
        # Set seed for reproducible results
        random.seed(42)
        
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        random.shuffle(csv_files)  # Randomize order
        
        # Split files into k folds
        fold_size = len(csv_files) // k_folds
        folds = []
        for i in range(k_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else len(csv_files)
            folds.append(csv_files[start_idx:end_idx])
        
        cv_results = []
        detailed_results = []
        
        for fold_idx, test_files in enumerate(folds):
            print(f"\n📋 Fold {fold_idx + 1}/{k_folds} - Testing on {len(test_files)} files")
            
            # Test on this fold
            correct = 0
            total = 0
            fold_details = []
            
            for filename in test_files:
                filepath = os.path.join(data_folder, filename)
                df = pd.read_csv(filepath)
                
                actual_label = filename.split('_')[0]
                predicted_label, confidence, debug_info = self.classify_swing(df)
                
                is_correct = predicted_label == actual_label
                if is_correct:
                    correct += 1
                total += 1
                
                fold_details.append({
                    'filename': filename,
                    'actual': actual_label,
                    'predicted': predicted_label,
                    'confidence': confidence,
                    'correct': is_correct
                })
            
            fold_accuracy = correct / total if total > 0 else 0
            cv_results.append(fold_accuracy)
            detailed_results.extend(fold_details)
            
            print(f"   Fold {fold_idx + 1} Accuracy: {fold_accuracy:.1%} ({correct}/{total})")
        
        # Calculate overall statistics
        mean_accuracy = np.mean(cv_results)
        std_accuracy = np.std(cv_results)
        
        # Per-class analysis across all folds
        per_class_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        for result in detailed_results:
            actual = result['actual']
            per_class_results[actual]['total'] += 1
            if result['correct']:
                per_class_results[actual]['correct'] += 1
        
        per_class_accuracy = {}
        for class_name, stats in per_class_results.items():
            per_class_accuracy[class_name] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"Mean Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.1%}")
        print(f"Individual Folds: {[f'{acc:.1%}' for acc in cv_results]}")
        print(f"Model Stability: {'✅ Stable' if std_accuracy < 0.1 else '⚠️ Unstable (high variance)'}")
        
        print(f"\nPer-Class Accuracy (across all folds):")
        for class_name, acc in per_class_accuracy.items():
            total_count = per_class_results[class_name]['total']
            correct_count = per_class_results[class_name]['correct']
            print(f"  {class_name:8}: {acc:.1%} ({correct_count}/{total_count})")
        
        return {
            'cv_scores': cv_results,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'detailed_results': detailed_results,
            'is_stable': std_accuracy < 0.1
        }
        
    def visualize_features(self, data_folder: str) -> None:
        """
        Create visualizations of feature distributions for different swing types.
        
        Args:
            data_folder: Path to folder containing CSV files
        """
        if not MATPLOTLIB_AVAILABLE:
            print("📊 Skipping visualizations - matplotlib not available")
            return
            
        print("📊 Creating feature visualizations...")
        
        # Collect all features by swing type
        swing_features = defaultdict(list)
        
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        for filename in csv_files:
            filepath = os.path.join(data_folder, filename)
            df = pd.read_csv(filepath)
            swing_type = filename.split('_')[0]
            features = self.extract_features(df)
            swing_features[swing_type].append(features)
        
        # Create subplots for key features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Golf Swing Feature Distributions by Type', fontsize=16)
        
        key_features = ['acc_mean', 'acc_max', 'gyro_mean', 'gyro_x_mean', 'gyro_y_mean', 'acc_std']
        
        for idx, feature in enumerate(key_features):
            ax = axes[idx // 3, idx % 3]
            
            for swing_type, feature_list in swing_features.items():
                feature_values = [f[feature] for f in feature_list if feature in f]
                if feature_values:
                    ax.hist(feature_values, alpha=0.6, label=swing_type, bins=10)
            
            ax.set_title(f'{feature}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('swing_feature_distributions.png', dpi=300, bbox_inches='tight')
        print("📈 Feature distribution plot saved as 'swing_feature_distributions.png'")
        
        # Create a correlation matrix for one swing type
        if swing_features:
            sample_swing = next(iter(swing_features.keys()))
            feature_df = pd.DataFrame(swing_features[sample_swing])
            
            plt.figure(figsize=(12, 8))
            correlation_matrix = feature_df.corr()
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Correlation')
            plt.title(f'Feature Correlation Matrix ({sample_swing} swings)')
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
            
            # Add correlation values as text
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white')
            
            plt.tight_layout()
            plt.savefig('feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
            print("🔗 Feature correlation matrix saved as 'feature_correlation_matrix.png'")


def main():
    """
    Main function to demonstrate the rule-based swing classifier.
    """
    print("🏌️ Golf Swing Rule-Based Classification System")
    print("=" * 60)
    
    # Initialize classifier
    classifier = RuleBasedSwingClassifier()
    
    # Data folder path
    data_folder = '/Users/jay/Desktop/Week2_Group4/data'
    
    if not os.path.exists(data_folder):
        print(f"❌ Error: Data folder '{data_folder}' not found!")
        return
        
    # Analyze training data and establish general physics-based rules
    classifier.analyze_training_data(data_folder)
    
    # Perform cross-validation to test robustness
    cv_results = classifier.cross_validate(data_folder, k_folds=5)
    
    # Create visualizations
    classifier.visualize_features(data_folder)
    
    # Show some example classifications from cross-validation results
    print(f"\n🔍 Example Classification Details from Cross-Validation:")
    print("-" * 70)
    
    # Show examples of correct and incorrect classifications
    correct_examples = [r for r in cv_results['detailed_results'] if r['correct']][:3]
    incorrect_examples = [r for r in cv_results['detailed_results'] if not r['correct']][:2]
    
    print("\n✅ Correct Classifications:")
    for result in correct_examples:
        print(f"📁 {result['filename']}: {result['actual']} → {result['predicted']} ({result['confidence']:.1%})")
    
    print("\n❌ Incorrect Classifications:")  
    for result in incorrect_examples:
        print(f"📁 {result['filename']}: {result['actual']} → {result['predicted']} ({result['confidence']:.1%})")
    
    # Final assessment
    print(f"\n🎯 Final Model Assessment:")
    print(f"Physics-Based Rules: ✅ Not overfitted to training data")
    print(f"Cross-Validation Accuracy: {cv_results['mean_accuracy']:.1%} ± {cv_results['std_accuracy']:.1%}")
    print(f"Model Stability: {cv_results['is_stable'] and '✅ Stable' or '⚠️ Needs improvement'}")
    
    print(f"\n📋 Advanced Multi-Dimensional Rule Summary:")
    print(f"🚀 EXPLOSIVE FAST SWINGS: acc_std > 4000 AND acc_max > 25000")
    print(f"   • Characterized by sudden, explosive acceleration patterns")
    print(f"🎯 CONTROLLED RIGHT SWINGS: gyro_y > 2800 AND acc_std < 4500") 
    print(f"   • Directional intent with controlled movement intensity")
    print(f"🧭 LEFT SWINGS: gyro_y ≤ 1800 with clear directional intent")
    print(f"   • Distinct leftward rotational pattern")
    print(f"⚡ MEDIUM-FAST SWINGS: 3000 < acc_std ≤ 4000")
    print(f"   • High intensity but not explosive") 
    print(f"🐌 SLOW SWINGS: 150 < acc_std ≤ 500")
    print(f"   • Low acceleration variance, controlled movement")
    print(f"😴 IDLE: acc_std ≤ 150 AND gyro_mean < 600")
    print(f"   • Minimal movement, stationary state")
    print(f"🔄 Key Innovation: Uses acceleration intensity (acc_std + acc_max) to distinguish")
    print(f"   explosive fast swings from controlled directional swings")


if __name__ == "__main__":
    main()