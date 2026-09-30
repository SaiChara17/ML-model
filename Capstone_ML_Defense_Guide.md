# Capstone Defense Guide: ML Model Architecture

This document is your "cheat sheet" for your capstone defense. It contains all the technical details of the AI model we developed, formatted specifically to answer tough questions a professor might ask.

---

## 1. The Core Architecture

### Q: "Explain your overall Machine Learning pipeline."
**Answer:** Our pipeline follows a 4-step process designed for Edge IoT:
1. **Acquisition:** The ESP32 collects raw vibration and acoustic voltages.
2. **Feature Extraction:** We process 1024-sample windows using signal processing (FFT and MFCC) to create a 21-dimension fused feature vector.
3. **Stage 1 (Anomaly Detection):** An unsupervised Deep Autoencoder checks the vector. If the reconstruction error (MSE) is above our threshold, it flags an anomaly.
4. **Stage 2 (Fault Classification):** A supervised Multi-Layer Perceptron (MLP) analyzes the anomalous vector and outputs the exact fault probability (e.g., Imbalance, Bearing Fault).

---

## 2. Feature Extraction & Sensor Fusion

### Q: "Why didn't you just feed the raw sensor data directly into the Neural Network?"
**Answer:** Feeding raw, high-frequency vibration and audio waves directly into a neural network requires massive, computationally heavy 1D-CNNs. Because we are targeting an Edge deployment, we mathematically extracted the most important features first. We used **Fast Fourier Transform (FFT)** to find dominant vibration harmonics, and **Mel-Frequency Cepstral Coefficients (MFCCs)** to extract acoustic pitch. This compressed thousands of raw data points into a tiny 21-number vector, making our AI thousands of times faster and smaller.

### Q: "How did you perform Sensor Fusion?"
**Answer:** We used **Feature-Level Fusion**. We extracted the statistical/FFT features from the vibration data, and the MFCC features from the audio data, and simply concatenated them into a single 1D vector before feeding it into the neural network.

---

## 3. The AI Models (Stage 1 and Stage 2)

### Q: "Why did you use a Two-Stage approach instead of just one classifier?"
**Answer:** In the real industrial world, we don't have data for *every possible way* a machine can break. If we only used a classifier, it would fail when a completely new, unseen fault occurred. 
By using an unsupervised **Autoencoder** for Stage 1, the system only needs to know what "Normal" looks like. It can detect *any* abnormality, even ones we didn't train it on. Stage 2 only activates to help diagnose the faults we *do* have labels for.

### Q: "How exactly does your Autoencoder detect anomalies?"
**Answer:** The Autoencoder compresses the 21-dimension input into a smaller "bottleneck" layer, and then tries to reconstruct the original 21 dimensions. We calculate the Mean Squared Error (MSE) between the input and the reconstruction. Healthy machine data reconstructs easily (Low MSE). Faulty machine data confuses the model (High MSE). If the MSE crosses our calculated threshold of `4.15`, the alarm is triggered.

---

## 4. Hardware and Deployment

### Q: "I see existing papers use Random Forest and Isolation Forest. Why did you use Deep Learning (Neural Networks)?"
**Answer:** While Random Forest is highly explainable, it is notoriously difficult to compile into C++ for microcontrollers. We used Neural Networks because they natively support **TensorFlow Lite (TFLite)**. TFLite is specifically built by Google for Edge IoT. It allowed us to compress our models down to a few kilobytes, making them highly optimized for deployment on hardware like a Raspberry Pi or an ESP32. Neural networks are also far superior at handling the complex, non-linear relationships found in multi-sensor fusion.

### Q: "Why is the ESP32 only doing data acquisition? Why isn't it running the ML model?"
**Answer:** The ESP32 *does* have the computational power to run the `.tflite` model (via TinyML). However, high-speed microphones (like the INMP441) require uninterrupted CPU attention so they don't drop data. If the ESP32 paused data collection to calculate complex FFTs and run the AI, it would drop thousands of audio samples, corrupting the stream. By using the ESP32 purely as a "data hose" and the Laptop as the Edge Computing Node, we completely eliminate data-blocking issues.

---

## 5. Datasets

### Q: "How did you validate your models?"
**Answer:** We used a Hybrid Dataset strategy to ensure our architecture works on industry standards:
1. **CWRU Bearing Dataset:** Used to benchmark the vibration (FFT/Time-domain) side of our pipeline against Drive-End bearing faults.
2. **MIMII Dataset:** Used to benchmark the acoustic (MFCC) side of our pipeline against industrial fan/valve noises.
3. **Intelligent Bearing (IB) Dataset:** Used to evaluate high-frequency vibration modalities across multiple fault classes.
