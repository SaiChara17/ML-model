import os
import librosa
import numpy as np

def load_mimii_data(data_dir='data/mimii', window_size=1024, sr=16000):
    """
    Loads real MIMII dataset audio files (.wav).
    MIMII is strictly an acoustic dataset. No fake vibration data is generated.
    
    Users must download the MIMII dataset from Zenodo and place .wav files in:
    - data/mimii/normal/
    - data/mimii/abnormal/
    """
    audio_data = []
    labels = []
    
    label_map = {'normal': 0, 'abnormal': 1}
    
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found. Please download MIMII dataset.")
        return np.array([]), np.array([])
        
    for state, label_val in label_map.items():
        state_dir = os.path.join(data_dir, state)
        if not os.path.exists(state_dir):
            continue
            
        for filename in os.listdir(state_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(state_dir, filename)
                # Load audio
                y, _ = librosa.load(filepath, sr=sr)
                
                # Segment into windows
                num_windows = len(y) // window_size
                for i in range(num_windows):
                    start = i * window_size
                    end = start + window_size
                    audio_data.append(y[start:end])
                    labels.append(label_val)
                    
    return np.array(audio_data), np.array(labels)
