import numpy as np
import tensorflow as tf
from preprocessing.features import fuse_features

# 1. Load the TFLite Models
ae_interpreter = tf.lite.Interpreter(model_path="saved_models/autoencoder.tflite")
ae_interpreter.allocate_tensors()
ae_input = ae_interpreter.get_input_details()[0]['index']
ae_output = ae_interpreter.get_output_details()[0]['index']

clf_interpreter = tf.lite.Interpreter(model_path="saved_models/classifier.tflite")
clf_interpreter.allocate_tensors()
clf_input = clf_interpreter.get_input_details()[0]['index']
clf_output = clf_interpreter.get_output_details()[0]['index']

# Hardcoded threshold from our training phase
ANOMALY_THRESHOLD = 4.15 

def predict_machine_health(raw_audio_array, raw_vibration_array):
    """
    Hardware Team: Call this function every time you receive 
    1024 samples from the ESP32!
    """
    # 2. Extract Features
    features = fuse_features(raw_audio_array, raw_vibration_array)
    features = np.expand_dims(features, axis=0).astype(np.float32) # Format for TFLite
    
    # 3. Stage 1: Anomaly Detection
    ae_interpreter.set_tensor(ae_input, features)
    ae_interpreter.invoke()
    reconstruction = ae_interpreter.get_tensor(ae_output)
    
    mse = np.mean(np.power(features - reconstruction, 2))
    
    if mse <= ANOMALY_THRESHOLD:
        return "🟢 MACHINE NORMAL"
        
    # 4. Stage 2: Fault Classification (Only runs if anomaly is detected!)
    clf_interpreter.set_tensor(clf_input, features)
    clf_interpreter.invoke()
    predictions = clf_interpreter.get_tensor(clf_output)[0]
    
    fault_class = np.argmax(predictions)
    
    if fault_class == 1:
        return f"🔴 ANOMALY DETECTED: Imbalance (Confidence: {predictions[1]*100:.2f}%)"
    elif fault_class == 2:
        return f"🔴 ANOMALY DETECTED: Bearing Fault (Confidence: {predictions[2]*100:.2f}%)"
    else:
        return "🟡 ANOMALY DETECTED: Unknown Fault"

if __name__ == "__main__":
    # --- TEST IT ---
    print("Testing pipeline with dummy ESP32 data...")
    dummy_audio = np.random.normal(0, 0.1, 1024)
    dummy_vib = np.random.normal(0, 0.1, 1024)
    print(predict_machine_health(dummy_audio, dummy_vib))
