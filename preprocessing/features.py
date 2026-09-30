import numpy as np
import scipy.stats as stats
import librosa

# ============================================================================
# PHASE 2 UPGRADE: ACADEMIC FEATURE EXTRACTION (No fake fusion)
# ============================================================================

def _spectral_entropy(pxx):
    """Calculate spectral entropy from power spectrum."""
    pxx_norm = pxx / (np.sum(pxx) + 1e-10)
    return -np.sum(pxx_norm * np.log2(pxx_norm + 1e-10))

def extract_vibration_features(vibration_segment, sr=12000, num_fft_peaks=5):
    """
    Extracts rigorous vibration features including advanced FFT metrics.
    Meets Groupmate's PDF Requirement #4.
    """
    features = []
    v = np.array(vibration_segment)
    
    # 1. Time-Domain Statistical Features
    mean_val = np.mean(v)
    std_val = np.std(v)
    var_val = np.var(v)
    rms = np.sqrt(np.mean(v**2))
    peak = np.max(np.abs(v))
    peak_to_peak = np.max(v) - np.min(v)
    crest_factor = peak / (rms + 1e-10)
    kurtosis = stats.kurtosis(v)
    skewness = stats.skew(v)
    
    features.extend([mean_val, std_val, var_val, rms, peak, peak_to_peak, crest_factor, kurtosis, skewness])
    
    # 2. Frequency-Domain (FFT) Features
    freqs = np.fft.rfftfreq(len(v), d=1/sr)
    fft_vals = np.abs(np.fft.rfft(v))
    
    # Dominant Frequency
    dom_idx = np.argmax(fft_vals)
    dom_freq = freqs[dom_idx]
    dom_mag = fft_vals[dom_idx]
    
    # Spectral Centroid (Vibration)
    spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)
    
    # Band Energy
    band_energy = np.sum(fft_vals**2)
    
    # Spectral Entropy
    spec_entropy = _spectral_entropy(fft_vals)
    
    features.extend([dom_freq, dom_mag, spectral_centroid, band_energy, spec_entropy])
    
    # 3. Multiple FFT Peaks (Top N Peaks)
    peak_indices = np.argsort(fft_vals)[-num_fft_peaks:][::-1]
    for idx in peak_indices:
        features.extend([freqs[idx], fft_vals[idx]])
        
    # Pad if less than num_fft_peaks found
    if len(peak_indices) < num_fft_peaks:
        for _ in range(num_fft_peaks - len(peak_indices)):
            features.extend([0.0, 0.0])
            
    return np.array(features)

def extract_acoustic_features(audio_segment, sr=16000):
    """
    Extracts rigorous acoustic features including MFCCs and Spectral properties.
    Meets Groupmate's PDF Requirement #5.
    """
    y = np.array(audio_segment).astype(np.float32)
    features = []
    
    # 1. Base Audio Features
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # 2. Spectral Features
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Spectral Entropy (Using FFT)
    fft_vals = np.abs(np.fft.rfft(y))
    spec_entropy = _spectral_entropy(fft_vals)
    
    # Dominant Frequency
    freqs = np.fft.rfftfreq(len(y), d=1/sr)
    dom_freq = freqs[np.argmax(fft_vals)]
    
    features.extend([rms, zcr, centroid, bandwidth, rolloff, spec_entropy, dom_freq])
    
    # 3. MFCCs (Mean across time)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    features.extend(mfccs_mean)
    
    return np.array(features)

# REMOVED: fuse_features(audio, vib). 
# Fake cross-dataset fusion is now strictly prohibited as per Requirement #6.
