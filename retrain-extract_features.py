import numpy as np
import pandas as pd
import neurokit2 as nk
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

sampling_rate = 300

def process_single_record(row, data_dir, sampling_rate):
    """Process a single ECG record and extract HRV features"""
    record = row['record']
    label = row['label']
    filepath = os.path.join(data_dir, f'{record}.npy')
    
    if not os.path.exists(filepath):
        return None
    
    try:
        signal = np.load(filepath)
        
        # Detect R-peaks using faster method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Try pantompkins first (fastest and most reliable)
                _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method="pantompkins")
            except:
                # Fallback to neurokit method
                _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method="neurokit")
        
        r_peaks = rpeaks['ECG_R_Peaks']

        if len(r_peaks) <= 2:
            return None

        # Calculate RR intervals (in seconds)
        rr_intervals = np.diff(r_peaks) / sampling_rate
        
        if len(rr_intervals) < 2:
            return None
        
        # Basic RR statistics
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals, ddof=1)
        median_rr = np.median(rr_intervals)
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)
        
        # RR differences
        rr_diff = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0
        sdsd = np.std(rr_diff, ddof=1) if len(rr_diff) > 1 else 0
        
        # pNN50 - percentage of successive RR intervals that differ by more than 50ms
        nn50 = np.sum(np.abs(rr_diff) > 0.05) if len(rr_diff) > 0 else 0
        pnn50 = (nn50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
        
        # pNN20 - more sensitive for AFib
        nn20 = np.sum(np.abs(rr_diff) > 0.02) if len(rr_diff) > 0 else 0
        pnn20 = (nn20 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
        
        # Heart rate statistics
        heart_rate_mean = 60 / mean_rr if mean_rr > 0 else 0
        heart_rate_min = 60 / max_rr if max_rr > 0 else 0
        heart_rate_max = 60 / min_rr if min_rr > 0 else 0
        heart_rate_std = np.std(60 / rr_intervals) if len(rr_intervals) > 0 else 0
        
        # Coefficient of variation (normalized measure of variability)
        cv_rr = (std_rr / mean_rr) * 100 if mean_rr > 0 else 0
        
        # Median absolute deviation (robust measure of variability)
        mad_rr = np.median(np.abs(rr_intervals - median_rr))
        
        # IQR (Interquartile range)
        q75, q25 = np.percentile(rr_intervals, [75, 25])
        iqr_rr = q75 - q25
        
        # Range of RR intervals
        rr_range = max_rr - min_rr
        
        # Successive difference measures
        mean_abs_diff = np.mean(np.abs(rr_diff)) if len(rr_diff) > 0 else 0
        median_abs_diff = np.median(np.abs(rr_diff)) if len(rr_diff) > 0 else 0
        max_abs_diff = np.max(np.abs(rr_diff)) if len(rr_diff) > 0 else 0
        
        # Poincaré plot features (SD1 and SD2)
        sd1 = np.sqrt(0.5 * np.var(rr_diff)) if len(rr_diff) > 0 else 0  # Short-term variability
        sd2_var = 2 * np.var(rr_intervals) - 0.5 * np.var(rr_diff) if len(rr_diff) > 0 else 0
        sd2 = np.sqrt(max(0, sd2_var))  # Long-term variability (ensure non-negative)
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0
        
        # Simplified entropy measure (much faster than sample entropy)
        # Using histogram-based approach
        try:
            hist, _ = np.histogram(rr_intervals, bins=min(20, len(rr_intervals) // 2))
            hist = hist[hist > 0]  # Remove zero bins
            prob = hist / hist.sum()
            shannon_entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0
        except:
            shannon_entropy = 0
        
        # Triangular index (geometric measure)
        try:
            hist_full, _ = np.histogram(rr_intervals, bins=min(128, len(rr_intervals)))
            triangular_index = len(rr_intervals) / np.max(hist_full) if np.max(hist_full) > 0 else 0
        except:
            triangular_index = 0
        
        # Turning points ratio (measure of randomness - high in AFib)
        turning_points = 0
        if len(rr_intervals) >= 3:
            for i in range(1, len(rr_intervals) - 1):
                if (rr_intervals[i] > rr_intervals[i-1] and rr_intervals[i] > rr_intervals[i+1]) or \
                   (rr_intervals[i] < rr_intervals[i-1] and rr_intervals[i] < rr_intervals[i+1]):
                    turning_points += 1
            turning_points_ratio = turning_points / (len(rr_intervals) - 2) if len(rr_intervals) > 2 else 0
        else:
            turning_points_ratio = 0

        return {
            'record': record,
            # Basic RR statistics
            'mean_rr': mean_rr,
            'median_rr': median_rr,
            'std_rr': std_rr,
            'min_rr': min_rr,
            'max_rr': max_rr,
            'cv_rr': cv_rr,
            'mad_rr': mad_rr,
            'iqr_rr': iqr_rr,
            'rr_range': rr_range,
            
            # Heart rate
            'heart_rate_mean': heart_rate_mean,
            'heart_rate_min': heart_rate_min,
            'heart_rate_max': heart_rate_max,
            'heart_rate_std': heart_rate_std,
            'num_beats': len(r_peaks),
            
            # HRV time-domain features
            'sdnn': std_rr,
            'rmssd': rmssd,
            'sdsd': sdsd,
            'pnn50': pnn50,
            'pnn20': pnn20,
            
            # Successive differences
            'mean_abs_diff': mean_abs_diff,
            'median_abs_diff': median_abs_diff,
            'max_abs_diff': max_abs_diff,
            
            # Poincaré features
            'sd1': sd1,
            'sd2': sd2,
            'sd_ratio': sd_ratio,
            
            # Entropy and complexity
            'shannon_entropy': shannon_entropy,
            'triangular_index': triangular_index,
            'turning_points_ratio': turning_points_ratio,
            
            'label': label
        }
    except Exception as e:
        print(f"Error processing {record}: {str(e)[:100]}")
        return None


def main():
    # Load labels
    labels = pd.read_csv('data/REFERENCE-v3.csv', header=None, names=['record', 'label'])
    labels = labels[labels['label'].isin(['N', 'A'])].reset_index(drop=True)
    
    data_dir = 'data/preprocessed'
    
    print(f"Processing {len(labels)} records...")
    print(f"Using {cpu_count()} CPU cores for parallel processing\n")
    
    # Parallel processing with chunk size for better progress tracking
    process_func = partial(process_single_record, data_dir=data_dir, sampling_rate=sampling_rate)
    
    # Use multiprocessing Pool with smaller chunks
    chunk_size = max(1, len(labels) // (cpu_count() * 4))
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_func, [row for _, row in labels.iterrows()], chunksize=chunk_size)
    
    # Filter out None results
    feature_list = [r for r in results if r is not None]
    
    # Create DataFrame and save
    df_features = pd.DataFrame(feature_list)
    
    os.makedirs('features', exist_ok=True)
    df_features.to_csv('features/hrv_features.csv', index=False)
    
    print(f"\n✓ Processed {len(feature_list)}/{len(labels)} records successfully")
    print(f"✓ Features saved to features/hrv_features.csv")
    
    print(f"\nDataset distribution:")
    print(df_features['label'].value_counts())
    
    print("\nFirst few rows:")
    print(df_features.head())
    
    print(f"\nSummary statistics for key features:")
    key_features = ['mean_rr', 'std_rr', 'cv_rr', 'rmssd', 'pnn50', 'heart_rate_mean']
    print(df_features.groupby('label')[key_features].mean())
    
    # Check for NaN/inf values
    nan_counts = df_features.isna().sum()
    inf_counts = np.isinf(df_features.select_dtypes(include=[np.number])).sum()
    
    if nan_counts.any() or inf_counts.any():
        print("\n⚠ Warning - Invalid values found:")
        if nan_counts.any():
            print("NaN values:", nan_counts[nan_counts > 0].to_dict())
        if inf_counts.any():
            print("Inf values:", inf_counts[inf_counts > 0].to_dict())


if __name__ == '__main__':
    main()