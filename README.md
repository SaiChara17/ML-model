# Edge-Based Machine Health Monitoring Using Multi-Sensor Data

This repository contains the Machine Learning pipeline for an Edge-Based Predictive Maintenance (PdM) capstone project. 

## Architecture
The system uses a two-stage learning strategy designed for edge deployment (e.g., Raspberry Pi, ESP32):
1. **Stage 1 (Anomaly Detection):** An Autoencoder trained on normal machine behavior to identify deviations.
2. **Stage 2 (Fault Classification):** A 1D-CNN / MLP to classify specific known faults (e.g., Imbalance, Bearing Faults).

## Sensor Fusion
The pipeline extracts Time-Domain, Frequency-Domain (FFT), and Mel-Spectrogram (MFCC) features from:
- Accelerometers (Vibration)
- INMP441 MEMS Microphones (Acoustic)
- Current & Temperature sensors

## Datasets Supported
- **CWRU Bearing Dataset**: Automated ingestion of `.mat` files for vibration benchmarking.
- **MIMII Dataset**: Acoustic benchmarking for industrial machinery.
- **Synthetic Data**: Integrated generator for rapid pipeline testing.

## Usage
Install dependencies:
```bash
pip install -r requirements.txt
```
Train the models (Outputs optimized `.tflite` models for the edge):
```bash
cd ml_pipeline
python train.py --dataset cwru
```
