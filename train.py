import numpy as np
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf

from data_cwru import load_cwru_data
from data_mimii import load_mimii_data
from data_ib import load_ib_data
from preprocessing.features import extract_vibration_features, extract_acoustic_features
from models.anomaly_detector import build_autoencoder
from models.fault_classifier import build_classifier

def main(dataset='cwru'):
    print(f"\n========================================================")
    print(f"--- Running V2 Academic Pipeline: {dataset.upper()} Dataset ---")
    print(f"========================================================\n")
    
    # 1. Dataset Loader & Feature Extraction
    if dataset == 'cwru':
        raw_data, labels = load_cwru_data()
        extract_fn = extract_vibration_features
    elif dataset == 'mimii':
        raw_data, labels = load_mimii_data()
        extract_fn = extract_acoustic_features
    elif dataset == 'ib':
        raw_data, labels = load_ib_data()
        extract_fn = extract_vibration_features
    else:
        print("Invalid dataset choice.")
        return

    if len(raw_data) == 0:
        print(f"Error: No data found for {dataset.upper()}.")
        return

    print("Extracting advanced mathematical features...")
    features = np.array([extract_fn(chunk) for chunk in raw_data])
    input_dim = features.shape[1]
    
    # 2. Strict Train/Val/Test Split (70/15/15)
    print("Splitting data (70% Train, 15% Validation, 15% Test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42) # 0.1764 of 0.85 = 0.15
    
    # 3. Scaling (StandardScaler)
    print("Applying StandardScaler (Fit on Train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # --- STAGE 1: Autoencoder (Anomaly Detection) ---
    print("\n--- Training Stage 1: Autoencoder (Anomaly Detection) ---")
    normal_train = X_train_scaled[y_train == 0]
    normal_val = X_val_scaled[y_val == 0]
    
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(normal_train, normal_train, epochs=20, batch_size=16, validation_data=(normal_val, normal_val), verbose=0)
    
    # Dynamic Thresholding (using Validation Max Error)
    reconstructions_val = autoencoder.predict(normal_val, verbose=0)
    mse_val = np.mean(np.power(normal_val - reconstructions_val, 2), axis=1)
    dynamic_threshold = np.max(mse_val) * 1.05 # Add 5% buffer
    print(f"Dynamic Anomaly Threshold calculated at: {dynamic_threshold:.4f}")
    
    # Train Isolation Forest (Academic Baseline for Stage 1)
    print("Training Isolation Forest (Baseline Anomaly Detector)...")
    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(normal_train)
    
    # Evaluate Anomaly Detectors on Test Set
    true_anomalies = (y_test > 0).astype(int)
    
    # Autoencoder predictions
    reconstructions_test = autoencoder.predict(X_test_scaled, verbose=0)
    mse_test = np.mean(np.power(X_test_scaled - reconstructions_test, 2), axis=1)
    ae_anomaly_preds = (mse_test > dynamic_threshold).astype(int)
    
    # Isolation Forest predictions (outputs 1 for normal, -1 for anomaly)
    iso_preds_raw = iso_forest.predict(X_test_scaled)
    iso_anomaly_preds = (iso_preds_raw == -1).astype(int)
    
    print("\n========================================================")
    print(f"--- STAGE 1 (ANOMALY DETECTION) EVALUATION ---")
    print(f"========================================================\n")
    print("== Deep Autoencoder Metrics ==")
    print(f"Accuracy:  {accuracy_score(true_anomalies, ae_anomaly_preds):.4f}")
    print(f"F1-Score:  {f1_score(true_anomalies, ae_anomaly_preds):.4f}")
    
    print("\n== Isolation Forest (Baseline) Metrics ==")
    print(f"Accuracy:  {accuracy_score(true_anomalies, iso_anomaly_preds):.4f}")
    print(f"F1-Score:  {f1_score(true_anomalies, iso_anomaly_preds):.4f}")
    
    # --- STAGE 2: Fault Classifier Baselines ---
    print("\n--- Training Stage 2: Classifier Baselines ---")
    
    # Model A: Multi-Layer Perceptron (Our Neural Network)
    print("Training MLP (Neural Network)...")
    num_classes = len(np.unique(labels))
    mlp_classifier = build_classifier(input_dim, num_classes=num_classes)
    mlp_classifier.fit(X_train_scaled, y_train, epochs=20, batch_size=16, validation_data=(X_val_scaled, y_val), verbose=0)
    
    # Model B: Random Forest (Academic Baseline)
    print("Training Random Forest (Baseline)...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # --- STAGE 2 EVALUATION ---
    print("\n========================================================")
    print(f"--- STAGE 2 (FAULT CLASSIFICATION) EVALUATION ---")
    print(f"========================================================\n")
    
    # MLP Evaluation
    mlp_preds = np.argmax(mlp_classifier.predict(X_test_scaled, verbose=0), axis=1)
    print("== Neural Network (MLP) Metrics ==")
    print(f"Accuracy:  {accuracy_score(y_test, mlp_preds):.4f}")
    print(f"Precision: {precision_score(y_test, mlp_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, mlp_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, mlp_preds, average='weighted', zero_division=0):.4f}")
    
    # Random Forest Evaluation
    rf_preds = rf_classifier.predict(X_test_scaled)
    print("\n== Random Forest (Baseline) Metrics ==")
    print(f"Accuracy:  {accuracy_score(y_test, rf_preds):.4f}")
    print(f"Precision: {precision_score(y_test, rf_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, rf_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, rf_preds, average='weighted', zero_division=0):.4f}")
    
    # --- TFLITE CONVERSION ---
    print("\nConverting Neural Networks to TensorFlow Lite (.tflite)...")
    autoencoder.save('saved_models/autoencoder.keras')
    mlp_classifier.save('saved_models/classifier.keras')
    
    converter_ae = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    with open('saved_models/autoencoder.tflite', 'wb') as f:
        f.write(converter_ae.convert())
        
    converter_clf = tf.lite.TFLiteConverter.from_keras_model(mlp_classifier)
    with open('saved_models/classifier.tflite', 'wb') as f:
        f.write(converter_clf.convert())
        
    print("\nPipeline execution complete! TFLite models and Scaler saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cwru', choices=['cwru', 'mimii', 'ib'])
    args = parser.parse_args()
    main(dataset=args.dataset)
