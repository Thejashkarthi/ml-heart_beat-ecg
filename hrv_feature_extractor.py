# scripts/hrv_feature_extractor.py
import numpy as np
import scipy.stats as stats

FEATURE_COLUMNS = [
    'mean_rr', 'median_rr', 'std_rr', 'min_rr', 'max_rr', 'cv_rr',
    'mad_rr', 'iqr_rr', 'rr_range', 'heart_rate_mean', 'heart_rate_min',
    'heart_rate_max', 'heart_rate_std', 'num_beats', 'sdnn', 'rmssd',
    'sdsd', 'pnn50', 'pnn20', 'mean_abs_diff', 'median_abs_diff',
    'max_abs_diff', 'sd1', 'sd2', 'sd_ratio', 'shannon_entropy',
    'triangular_index', 'turning_points_ratio'
]

def extract_hrv_features(r_peaks: np.ndarray, fs: int = 250) -> dict:
    """
    Compute HRV features from R-peak indices and sampling rate.
    
    Args:
        r_peaks: Array of R-peak sample indices
        fs: Sampling frequency in Hz (default 250)
    
    Returns:
        Dictionary of 28 HRV features
    """
    if len(r_peaks) < 2:
        raise ValueError("Need at least 2 R-peaks to compute RR intervals.")

    # Convert R-peak indices to RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs  # in seconds

    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    std_rr = np.std(rr_intervals)
    min_rr = np.min(rr_intervals)
    max_rr = np.max(rr_intervals)
    cv_rr = std_rr / mean_rr if mean_rr != 0 else 0
    mad_rr = np.mean(np.abs(rr_intervals - mean_rr))
    iqr_rr = stats.iqr(rr_intervals)
    rr_range = max_rr - min_rr

    heart_rates = 60.0 / rr_intervals
    heart_rate_mean = np.mean(heart_rates)
    heart_rate_min = np.min(heart_rates)
    heart_rate_max = np.max(heart_rates)
    heart_rate_std = np.std(heart_rates)

    num_beats = len(rr_intervals)
    sdnn = std_rr * 1000
    diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diffs**2)) * 1000
    sdsd = np.std(diffs) * 1000

    nn50 = np.sum(np.abs(diffs) > 0.05)
    pnn50 = nn50 / len(diffs) * 100 if len(diffs) > 0 else 0
    nn20 = np.sum(np.abs(diffs) > 0.02)
    pnn20 = nn20 / len(diffs) * 100 if len(diffs) > 0 else 0

    mean_abs_diff = np.mean(np.abs(diffs))
    median_abs_diff = np.median(np.abs(diffs))
    max_abs_diff = np.max(np.abs(diffs))

    sd1 = rmssd / np.sqrt(2)
    sd2 = np.sqrt(2 * (sdnn**2) - sd1**2) if sdnn else 0
    sd_ratio = sd2 / sd1 if sd1 != 0 else 0

    bins = np.arange(0, rr_intervals.max() + 0.05, 0.05)
    hist, _ = np.histogram(rr_intervals, bins=bins, density=True)
    hist = hist[hist > 0]
    shannon_entropy = -np.sum(hist * np.log2(hist)) if len(hist) else 0

    triangular_index = num_beats / (np.max(np.histogram(rr_intervals, bins=30)[0]) + 1e-9)

    turning = sum(1 for i in range(1, len(rr_intervals)-1)
                  if (rr_intervals[i-1] < rr_intervals[i] > rr_intervals[i+1]) or
                     (rr_intervals[i-1] > rr_intervals[i] < rr_intervals[i+1]))
    turning_points_ratio = turning / (len(rr_intervals) - 2) if len(rr_intervals) > 2 else 0

    # Create the features dictionary
    features = dict(zip(FEATURE_COLUMNS, [
        mean_rr, median_rr, std_rr, min_rr, max_rr, cv_rr, mad_rr, iqr_rr, rr_range,
        heart_rate_mean, heart_rate_min, heart_rate_max, heart_rate_std, num_beats,
        sdnn, rmssd, sdsd, pnn50, pnn20, mean_abs_diff, median_abs_diff,
        max_abs_diff, sd1, sd2, sd_ratio, shannon_entropy, triangular_index,
        turning_points_ratio
    ]))
    #print("Features:", features)  # Moved after features is defined
    return features

# Standalone test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = np.arange(0, 10, 1/250)
    signal = np.sin(2*np.pi*1.2*t) + 0.2*np.random.randn(len(t))
    from r_peak_detector import detect_r_peaks
    peaks = detect_r_peaks(signal, fs=250)
    features = extract_hrv_features(peaks, fs=250)
    print("Features computed:", len(features))
    print("Sample:", {k: round(v, 3) for k, v in list(features.items())[:5]})