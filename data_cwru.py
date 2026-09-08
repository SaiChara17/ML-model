import os
import requests
import scipy.io as sio
import numpy as np

# CWRU 12k Drive End Bearing Fault Data URLs (1HP Load)
CWRU_URLS = {
    'normal': 'https://engineering.case.edu/sites/default/files/97.mat', # Normal Baseline 
    'imbalance': 'https://engineering.case.edu/sites/default/files/105.mat', # Inner Race fault
    'bearing_fault': 'https://engineering.case.edu/sites/default/files/130.mat' # Outer Race fault
}

def download_cwru():
    os.makedirs('data/cwru', exist_ok=True)
    import urllib3
    urllib3.disable_warnings() # Case Western site has a self-signed cert
    
    for label, url in CWRU_URLS.items():
        filename = url.split('/')[-1]
        filepath = os.path.join('data/cwru', filename)
        if not os.path.exists(filepath):
            print(f"Downloading CWRU {label} data ({filename})...")
            try:
                # SSL Verification disabled due to Case Western server config issues
                response = requests.get(url, verify=False)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def load_cwru_data(window_size=1024, max_windows_per_file=200):
    """
    Loads downloaded CWRU .mat files and chunks them into windows.
    Since CWRU is only vibration, we return dummy arrays for audio.
    """
    download_cwru()
    
    audio_data = []
    vibration_data = []
    labels = []
    
    label_map = {'normal': 0, 'imbalance': 1, 'bearing_fault': 2}
    mat_keys = {'normal': 'X097_DE_time', 'imbalance': 'X105_DE_time', 'bearing_fault': 'X130_DE_time'}
    
    for state, url in CWRU_URLS.items():
        filename = url.split('/')[-1]
        filepath = os.path.join('data/cwru', filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found.")
            continue
            
        mat_data = sio.loadmat(filepath)
        key = mat_keys[state]
        
        # Fallback dynamic key finder if hardcoded keys fail
        if key not in mat_data:
            for k in mat_data.keys():
                if 'DE_time' in k:
                    key = k
                    break
                    
        raw_vibration = mat_data[key].flatten()
        
        # Chunk into overlapping or non-overlapping windows
        num_windows = min(len(raw_vibration) // window_size, max_windows_per_file)
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            vib_chunk = raw_vibration[start:end]
            
            # CWRU benchmark trick: Feed zeros to the audio channel
            audio_chunk = np.zeros(window_size)
            
            audio_data.append(audio_chunk)
            vibration_data.append(vib_chunk)
            labels.append(label_map[state])
            
    return np.array(audio_data), np.array(vibration_data), np.array(labels)
