import numpy as np
import scipy.stats as stats
import librosa

def extract_audio_features(audio_segment, sr=16000):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio segment.
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    # Average the MFCCs over time to get a single vector per segment
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def extract_vibration_features(vibration_segment):
    """
    Extracts time-domain and frequency-domain features from vibration data.
    """
    # Time domain
    rms = np.sqrt(np.mean(vibration_segment**2))
    kurtosis = stats.kurtosis(vibration_segment)
    skewness = stats.skew(vibration_segment)
    peak_to_peak = np.ptp(vibration_segment)
    crest_factor = np.max(np.abs(vibration_segment)) / rms if rms > 0 else 0
    
    # Frequency domain (FFT)
    fft_vals = np.abs(np.fft.rfft(vibration_segment))
    # Get top 3 dominant frequencies magnitudes
    top_freqs_mags = np.sort(fft_vals)[-3:][::-1]
    
    features = np.array([rms, kurtosis, skewness, peak_to_peak, crest_factor])
    features = np.concatenate([features, top_freqs_mags])
    return features

def fuse_features(audio_seg, vib_seg, sr=16000):
    """
    Extracts and concatenates both audio and vibration features.
    """
    audio_feats = extract_audio_features(audio_seg, sr=sr)
    vib_feats = extract_vibration_features(vib_seg)
    return np.concatenate([audio_feats, vib_feats])
