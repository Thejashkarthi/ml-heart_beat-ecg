# scripts/r_peak_detector.py
import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Detect R-peaks using only SciPy (no NeuroKit2).
    Works on any numeric ECG signal.
    """
    # Remove baseline wander
    filtered = signal - np.convolve(signal, np.ones(int(fs*0.2))/ (fs*0.2), mode='same')
    
    # Use absolute value for QRS detection
    abs_sig = np.abs(filtered)
    
    # Adaptive prominence
    prominence = max(0.15 * (abs_sig.max() - abs_sig.min()), 0.05)
    
    peaks, _ = find_peaks(
        abs_sig,
        distance=fs // 3,        # min 200 ms between beats
        prominence=prominence,
        height=0.2 * abs_sig.max()
    )
    
    if len(peaks) < 3:
        raise ValueError(f"Only {len(peaks)} R-peaks found. Signal too short/noisy.")
    
    return np.array(peaks, dtype=int)