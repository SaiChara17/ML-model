# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
import pickle
from preprocessing.features import extract_vibration_features, extract_acoustic_features

# ============================================================================
# PHASE 4: ESP32 HARDWARE INTEGRATION HAND-OFF (V2 Pipeline)
# ============================================================================

# 1. Load the TFLite Models and Scaler
ae_interpreter = tf.lite.Interpreter(model_path="saved_models/autoencoder.tflite")
ae_interpreter.allocate_tensors()
ae_input = ae_interpreter.get_input_details()[0]['index']
ae_output = ae_interpreter.get_output_details()[0]['index']

clf_interpreter = tf.lite.Interpreter(model_path="saved_models/classifier.tflite")
clf_interpreter.allocate_tensors()
clf_input = clf_interpreter.get_input_details()[0]['index']
clf_output = clf_interpreter.get_output_details()[0]['index']

try:
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: scaler.pkl not found. Please run train.py first!")
    exit(1)

# UPDATE THIS TO MATCH THE OUTPUT FROM train.py!
DYNAMIC_ANOMALY_THRESHOLD = 0.2426 

def predict_machine_health(raw_array, sensor_type='vibration'):
    """
    Hardware Team: Call this function every time you receive 
    1024 samples from the ESP32!
    
    sensor_type must be either 'vibration' or 'acoustic'.
    Cross-dataset fake fusion is strictly prohibited.
    """
    # 2. Extract Rigorous Academic Features
    if sensor_type == 'vibration':
        features = extract_vibration_features(raw_array)
    else:
        features = extract_acoustic_features(raw_array)
        
    features = np.expand_dims(features, axis=0).astype(np.float32)
    
    # 3. Apply StandardScaler (CRITICAL for accuracy)
    features_scaled = scaler.transform(features).astype(np.float32)
    
    # 4. Stage 1: Anomaly Detection
    ae_interpreter.set_tensor(ae_input, features_scaled)
    ae_interpreter.invoke()
    reconstruction = ae_interpreter.get_tensor(ae_output)
    
    mse = np.mean(np.power(features_scaled - reconstruction, 2))
    
    if mse <= DYNAMIC_ANOMALY_THRESHOLD:
        return f"🟢 MACHINE NORMAL (MSE: {mse:.4f})"
        
    # 5. Stage 2: Fault Classification (Only runs if anomaly is detected!)
    clf_interpreter.set_tensor(clf_input, features_scaled)
    clf_interpreter.invoke()
    predictions = clf_interpreter.get_tensor(clf_output)[0]
    
    fault_class = np.argmax(predictions)
    
    if fault_class == 1:
        return f"🔴 ANOMALY DETECTED: Imbalance / Inner Race (Confidence: {predictions[1]*100:.2f}%)"
    elif fault_class == 2:
        return f"🔴 ANOMALY DETECTED: Bearing Fault / Outer Race (Confidence: {predictions[2]*100:.2f}%)"
    else:
        return "🟡 ANOMALY DETECTED: Unknown Fault"

if __name__ == "__main__":
    # --- TEST IT ---
    print("Testing V2 pipeline with dummy ESP32 vibration data...")
    dummy_vib = np.random.normal(0, 0.1, 1024)
    print(predict_machine_health(dummy_vib, sensor_type='vibration'))
