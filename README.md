# IMU Swing Classifier 🏌️‍♂️

This project trains a machine learning model to classify different types of swings (e.g., 'slow', 'fast', 'left', 'right') using sensor data from an Inertial Measurement Unit (IMU). It extracts statistical and frequency-domain features from raw accelerometer and gyroscope data to train a `RandomForestClassifier`.

The script automates the process of loading data, engineering features, training a model, evaluating its performance on the training set, and saving the final artifacts for deployment.

---

## ✨ Features

* **Data Loading**: Automatically loads and processes all `.csv` files from a specified directory.
* **Feature Engineering**: Extracts a robust set of 48 features from the raw time-series data to create a "fingerprint" for each swing.
* **Model Training**: Uses a `RandomForestClassifier`, a powerful and versatile model for this type of classification task.
* **Evaluation**: Generates a classification report and a visual confusion matrix to assess model performance.
* **Serialization**: Saves the trained model, data scaler, and label encoder using `joblib` for easy use in other applications.

---

## ⚙️ How It Works

The script follows a standard machine learning pipeline to classify motion gestures.

### 1. Data Preparation

The script expects your IMU data to be in `.csv` files. The label for each motion is extracted directly from the filename.

* **File Naming Convention**: Files must be named in the format `label_description.csv`. For example, `fast_swing_1.csv`, `slow_swing_A.csv`, or `left_turn.csv`. The script will use `fast`, `slow`, and `left` as the classification labels.
* **Data Columns**: Each CSV file must contain columns for the 6-axis IMU data: `ax`, `ay`, `az` (accelerometer) and `gx`, `gy`, `gz` (gyroscope).

### 2. Feature Extraction

To classify the entire time-series of a swing, the script condenses the data from each file into a single feature vector of 48 values. This is done by calculating a set of descriptive statistics for each of the 6 sensor axes.

#### Statistical Features (Time-Domain)
These describe the distribution and magnitude of the sensor readings.
* **Mean**: Average value.
* **Standard Deviation (std)**: Measures the amount of variation or dispersion. Key for detecting intensity.
* **Min & Max**: The peak minimum and maximum values reached.
* **Median**: The middle value, robust to outliers.
* **Root Mean Square (RMS)**: Represents the signal's energy or magnitude.

#### Frequency Features (Frequency-Domain)
Calculated using the Fast Fourier Transform (FFT) to analyze the frequency components of the motion.
* **FFT Mean**: The average magnitude of the frequency components.
* **FFT Peak**: The single most dominant frequency, indicating the primary rate of motion.



The final feature vector for one swing is created by concatenating all 8 feature types for each of the 6 sensor axes ($8 \times 6 = 48$ features).

### 3. Model Training

1.  **Scaling**: The feature matrix `X` is standardized using `StandardScaler` to ensure all features have a mean of 0 and a standard deviation of 1. This is crucial for model performance.
2.  **Encoding**: The string labels (`'fast'`, `'slow'`, etc.) are converted to integers using `LabelEncoder`.
3.  **Training**: A `RandomForestClassifier` is trained on the scaled features and encoded labels. `class_weight='balanced'` is used to handle any imbalance in the number of examples for each swing type.

---

## 🚀 Usage

### 1. Requirements

Make sure you have Python 3 installed. You can install the necessary libraries using pip:
```bash
pip install pandas numpy scikit-learn joblib matplotlib
