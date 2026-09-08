import numpy as np

def generate_synthetic_data(num_samples=100, length=1024, state='normal'):
    """
    Generates synthetic audio and vibration data for testing.
    states: 'normal', 'imbalance', 'bearing_fault'
    """
    audio_data = []
    vibration_data = []
    labels = []
    
    t = np.linspace(0, 1, length, endpoint=False)
    
    for _ in range(num_samples):
        if state == 'normal':
            # Clean low-frequency sine waves
            aud = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.1, length)
            vib = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.05, length)
            label = 0
        elif state == 'imbalance':
            # Higher amplitude at base frequency
            aud = 2.0 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.1, length)
            vib = 2.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.05, length)
            label = 1
        elif state == 'bearing_fault':
            # High-frequency noise added
            aud = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 500 * t) + np.random.normal(0, 0.3, length)
            vib = np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 120 * t) + np.random.normal(0, 0.2, length)
            label = 2
            
        audio_data.append(aud)
        vibration_data.append(vib)
        labels.append(label)
        
    return np.array(audio_data), np.array(vibration_data), np.array(labels)
