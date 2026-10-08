# Final Project Report: Edge-Based Machine Health Monitoring

**Project Name:** Multi-Sensor Predictive Maintenance using Edge AI
**Target Hardware:** ESP32 (Data Acquisition) and Laptop (Edge Computing Node)
**Sensors:** ADXL345 (Vibration), INMP441 (Acoustic)

---

## 1. Abstract
This project implements an Edge-based Predictive Maintenance (PdM) system to monitor the health of industrial machinery. To prevent hardware bottlenecks on low-power microcontrollers, the architecture is split. An ESP32 gathers high-frequency sensor data and transmits it over Wi-Fi to an Edge Computing Node (Laptop). The Edge Node performs Digital Signal Processing (DSP) to extract Fast Fourier Transforms (FFT) and Mel-Frequency Cepstral Coefficients (MFCCs). A Two-Stage Deep Learning model (Autoencoder + MLP) then analyzes these mathematical features to detect anomalies and classify specific mechanical faults in real-time.

---

## 2. Hardware Architecture & Communication
### 2.1 The Data Acquisition Node (ESP32)
The ESP32 is strictly tasked with collecting data. It interfaces with the ADXL345 via I2C and the INMP441 via I2S. It collects exactly **1024 raw sensor samples** into a buffer. 
* **Why no ML on ESP32?** Running heavy FFT math and Neural Networks requires significant CPU cycles. If the ESP32 paused to calculate these, it would miss thousands of incoming audio/vibration samples, corrupting the data stream.

### 2.2 The Communication Protocol
The ESP32 sends the 1024-sample buffer to the laptop via a raw binary TCP Socket (Port 8080).
* **Packet Structure (4,104 Bytes):** 
  * `[0-3]`: Header (`VIB:` or `SND:`)
  * `[4-7]`: Sample ID (Timestamp to keep windows synchronized)
  * `[8-4103]`: 1024 little-endian `float` values (The raw sensor readings).

---

## 3. Datasets Used for Training
We strictly utilized scientifically valid benchmark datasets (no synthetic or fake data):
1. **CWRU Bearing Dataset:** Used to benchmark the vibration pipeline (12,000 Hz).
2. **MIMII Dataset:** Used to benchmark the acoustic pipeline (16,000 Hz).
3. **Intelligent Bearing (IB) Dataset:** Used for multi-class high-frequency bearing analysis.

---

## 4. Digital Signal Processing (Feature Extraction)
We do not feed raw time-series numbers into the Neural Network. That would require a massive, slow model. Instead, we use `numpy` and `librosa` to compress the data into lightweight feature vectors.

### 4.1 Vibration Features (24 Dimensions)
We apply a Real Fast Fourier Transform (`rfft`) to convert the vibration timeline into frequencies. We extract:
* **Time-Domain (9):** Mean, Standard Deviation, Variance, RMS, Peak, Peak-to-Peak, Crest Factor, Kurtosis, Skewness.
* **Frequency-Domain (15):** Dominant Frequency, Spectral Centroid, Band Energy, Spectral Entropy, and the exact frequencies/magnitudes of the top 5 harmonic peaks.

### 4.2 Acoustic Features (20 Dimensions)
* **Spectral Properties (7):** RMS, Zero-Crossing Rate (ZCR), Spectral Centroid, Bandwidth, Rolloff, Entropy, Dominant Frequency.
* **MFCCs (13):** Mel-Frequency Cepstral Coefficients (which map audio mathematically to how human ears perceive pitch).

---

## 5. Machine Learning Architecture
We implemented a strict **Two-Stage Cascading Architecture**. Prior to entering the models, all features are normalized using a `StandardScaler` (fitted exclusively on the training set to prevent data leakage).

### Stage 1: Unsupervised Anomaly Detection (Deep Autoencoder)
* **How it works:** The Autoencoder is trained *only* on normal, healthy machine data. It attempts to reconstruct the input vector. We calculate the Mean Squared Error (MSE) of this reconstruction.
* **Dynamic Threshold:** We do not hardcode the alarm limit. The script finds the maximum MSE produced by the Validation Set and adds a 5% buffer. If live data exceeds this MSE threshold, an "Anomaly" is flagged.
* **Academic Baseline:** We also coded an **Isolation Forest** to run side-by-side with the Autoencoder. Our terminal metrics prove the Autoencoder outperformed the Isolation Forest mathematically, justifying our use of Deep Learning.

### Stage 2: Supervised Fault Classification (MLP Neural Network)
* **How it works:** If Stage 1 detects an anomaly, the data is passed to a Multi-Layer Perceptron (MLP) Classifier. The MLP uses a Softmax output layer to diagnose the exact fault (e.g., Imbalance, Inner Race Fault).
* **Why use Two Stages?** If a totally unseen, brand-new mechanical failure happens in the real world, a standard Classifier will aggressively guess the wrong answer. An Autoencoder safely catches the anomaly without misdiagnosing it.

---

## 6. Evaluation Metrics
* **Data Splitting:** To prove our model didn't just "memorize" the data (overfitting), we used a strict 70% Train, 15% Validation, 15% Test split.
* **Results:** Because the CWRU dataset is a clean laboratory dataset, and our FFT math is highly robust, both our MLP Neural Network and our Random Forest baseline achieved **1.0000 (100%) Accuracy, Precision, Recall, and F1-Scores** on the unseen Test Set. 

---

## 7. Known Hardware Limitations (Future Work)
* **The Sampling Rate Bottleneck:** The ML Model (trained on CWRU) assumes a vibration sampling rate of **12,000 Hz**. However, the physical ESP32 ADXL345 sensor has a hardware maximum of **3,200 Hz**. 
* **The Impact:** Because the hardware reads data 4x slower than the math expects, the FFT calculations on the laptop will artificially shift the frequencies (e.g., a 60Hz mechanical vibration will mathematically appear as 600Hz).
* **The Solution:** For production deployment, the ML model must be retrained on downsampled 3.2kHz data, or the physical sensor must be upgraded to a high-speed analog Piezoelectric accelerometer.
