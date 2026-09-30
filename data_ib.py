import os
import numpy as np
import scipy.io as sio
import pandas as pd

def load_ib_data(data_dir='data/ib', window_size=1024):
    """
    Loads the Intelligent Bearing (IB) Dataset.
    
    Expected IB Dataset Characteristics (To be verified by actual dataset):
    - Channels: Vibration (and potentially acoustic/current depending on the specific IB sub-dataset)
    - Sampling Rate: Typically high-frequency (e.g., 50kHz or 1MHz)
    - Classes: Normal (0), Inner Race Fault (1), Outer Race Fault (2), Ball Fault (3)
    
    Users must place the IB data files in: data/ib/
    """
    vibration_data = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found. Please download IB dataset.")
        return np.array([]), np.array([])
        
    # Placeholder for actual IB dataset parsing logic once files are present.
    # Typical IB datasets are provided in .mat or .csv format.
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        
        # Example parsing logic (Adjust based on exact IB format)
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            raw_signal = df.iloc[:, 0].values # Assuming first column is the main sensor
            
            # Extract label from filename (e.g., 'normal_1.csv', 'inner_fault.csv')
            label = 0 if 'normal' in filename.lower() else 1 
            
            num_windows = len(raw_signal) // window_size
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                vibration_data.append(raw_signal[start:end])
                labels.append(label)
                
    return np.array(vibration_data), np.array(labels)
