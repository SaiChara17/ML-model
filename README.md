# Edge-Based Machine Health Monitoring Using Multi-Sensor Data

This repository contains the Machine Learning pipeline for an Edge-Based Predictive Maintenance (PdM) capstone project. 
It strictly adheres to rigorous academic standards for feature extraction, scaling, and evaluation.

## Architecture
The system uses a two-stage learning strategy designed for edge deployment:
1. **Stage 1 (Anomaly Detection):** An Autoencoder trained solely on normal machine behavior. It dynamically calculates an anomaly threshold using Mean Squared Error (MSE) reconstruction loss from a validation set.
2. **Stage 2 (Fault Classification):** A Multi-Layer Perceptron (MLP) to classify specific known faults (e.g., Imbalance, Bearing Faults). A Random Forest baseline is included for academic comparison.

## Hardware Flow
`ESP32 -> Laptop -> ML/TFLite Inference`
* **ESP32:** Performs sensor acquisition from ADXL345 (vibration), INMP441 (acoustic), INA219 (electrical), and DS18B20 (temperature). Streams raw data via Serial/Wi-Fi.
* **Laptop:** Performs mathematical preprocessing, feature extraction, `StandardScaler` normalization, and AI inference using `.tflite` models.

## Datasets Supported
We strictly use scientifically valid benchmark datasets without artificial padding or cross-dataset fusion:
- **CWRU Bearing Dataset**: Used for vibration (FFT/Time-domain) modality benchmarking.
- **MIMII Dataset**: Used for machine-acoustic (MFCC) anomaly analysis.
- **Intelligent Bearing (IB) Dataset**: Used for multi-class high-frequency bearing analysis.

## Usage
Install dependencies:
```bash
pip install -r requirements.txt
```
Train the models (Outputs optimized `.tflite` models, `scaler.pkl`, and evaluation metrics):
```bash
cd ml_pipeline
python train.py --dataset cwru
```
Test the hardware Serial handoff:
```bash
cd ml_pipeline
python inference_demo.py
```
