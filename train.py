import numpy as np
import os
import argparse
import tensorflow as tf
from data_gen import generate_synthetic_data
from data_cwru import load_cwru_data
from data_mimii import load_mimii_data
from preprocessing.features import fuse_features
from models.anomaly_detector import build_autoencoder
from models.fault_classifier import build_classifier

def main(dataset='synthetic'):
    print(f"\n========================================================")
    print(f"--- Running Pipeline with {dataset.upper()} Dataset ---")
    print(f"========================================================\n")
    
    if dataset == 'synthetic':
        print("Generating synthetic data...")
        normal_aud, normal_vib, normal_labels = generate_synthetic_data(num_samples=200, state='normal')
        imb_aud, imb_vib, imb_labels = generate_synthetic_data(num_samples=100, state='imbalance')
        brg_aud, brg_vib, brg_labels = generate_synthetic_data(num_samples=100, state='bearing_fault')
        
    elif dataset == 'cwru':
        print("Loading CWRU Bearing Dataset (Downloading if needed)...")
        cwru_aud, cwru_vib, cwru_labels = load_cwru_data()
        normal_idx = cwru_labels == 0
        normal_aud, normal_vib, normal_labels = cwru_aud[normal_idx], cwru_vib[normal_idx], cwru_labels[normal_idx]
        
        fault_idx = cwru_labels != 0
        imb_aud, imb_vib, imb_labels = cwru_aud[fault_idx], cwru_vib[fault_idx], cwru_labels[fault_idx]
        brg_aud, brg_vib, brg_labels = [], [], [] # All faults combined above
        
    elif dataset == 'mimii':
        print("Loading MIMII Acoustic Dataset...")
        mimii_aud, mimii_vib, mimii_labels = load_mimii_data()
        normal_idx = mimii_labels == 0
        normal_aud, normal_vib, normal_labels = mimii_aud[normal_idx], mimii_vib[normal_idx], mimii_labels[normal_idx]
        
        fault_idx = mimii_labels != 0
        imb_aud, imb_vib, imb_labels = mimii_aud[fault_idx], mimii_vib[fault_idx], mimii_labels[fault_idx]
        brg_aud, brg_vib, brg_labels = [], [], []
        
    else:
        print("Invalid dataset choice.")
        return

    print("Extracting features...")
    # Process Normal Data
    normal_features = np.array([fuse_features(a, v) for a, v in zip(normal_aud, normal_vib)])
    
    # Process Fault Data
    all_fault_aud = np.concatenate([imb_aud, brg_aud]) if len(brg_aud) > 0 else imb_aud
    all_fault_vib = np.concatenate([imb_vib, brg_vib]) if len(brg_vib) > 0 else imb_vib
    all_fault_labels = np.concatenate([imb_labels, brg_labels]) if len(brg_labels) > 0 else imb_labels
    
    fault_features = np.array([fuse_features(a, v) for a, v in zip(all_fault_aud, all_fault_vib)])
    
    input_dim = normal_features.shape[1]
    
    print(f"Feature Vector Dimension: {input_dim}")
    
    # --- STAGE 1: Train Anomaly Detector (Autoencoder) ---
    print("\n--- Training Stage 1: Autoencoder (Anomaly Detection) ---")
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(normal_features, normal_features, 
                    epochs=20, batch_size=16, 
                    validation_split=0.2, verbose=1)
    
    reconstructions = autoencoder.predict(normal_features)
    mse = np.mean(np.power(normal_features - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    print(f"Anomaly Threshold calculated at: {threshold:.4f}")
    
    # --- STAGE 2: Train Fault Classifier ---
    print("\n--- Training Stage 2: Classifier (Fault Diagnosis) ---")
    classifier = build_classifier(input_dim, num_classes=3)
    
    all_features = np.concatenate([normal_features, fault_features])
    all_labels = np.concatenate([normal_labels, all_fault_labels])
    
    classifier.fit(all_features, all_labels, 
                   epochs=20, batch_size=16, 
                   validation_split=0.2, verbose=1)
    
    # Save models
    print("\nSaving models...")
    os.makedirs('saved_models', exist_ok=True)
    autoencoder.save('saved_models/autoencoder.keras')
    classifier.save('saved_models/classifier.keras')
    
    print("\nConverting models to TensorFlow Lite (.tflite)...")
    converter_ae = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    tflite_ae = converter_ae.convert()
    with open('saved_models/autoencoder.tflite', 'wb') as f:
        f.write(tflite_ae)
        
    converter_clf = tf.lite.TFLiteConverter.from_keras_model(classifier)
    tflite_clf = converter_clf.convert()
    with open('saved_models/classifier.tflite', 'wb') as f:
        f.write(tflite_clf)
        
    print(f"\nPipeline execution for {dataset.upper()} complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Pipeline")
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'cwru', 'mimii'], help="Dataset to use")
    args = parser.parse_args()
    main(dataset=args.dataset)
