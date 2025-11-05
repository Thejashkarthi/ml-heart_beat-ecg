import numpy as np
import pandas as pd
import neurokit2 as nk
import os

# Load label file
labels = pd.read_csv('data/REFERENCE-v3.csv', header=None, names=['record', 'label'])
labels = labels[labels['label'].isin(['N', 'A'])].reset_index(drop=True)

data_dir = 'data/preprocessed'
sampling_rate = 300  # Change if different for your dataset

records, mean_rr, std_rr, rmssd_rr, heart_rates, num_beats, labels_list = [], [], [], [], [], [], []

for i, row in labels.iterrows():
    record = row['record']
    label = row['label']
    npy_file = os.path.join(data_dir, f'{record}.npy')
    if os.path.exists(npy_file):
        signal = np.load(npy_file)
        # Find R-peaks
        try:
            _, rpeak_data = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
            r_peaks = rpeak_data['ECG_R_Peaks']
            if len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks) / sampling_rate  # seconds
                records.append(record)
                mean_rr.append(np.mean(rr_intervals))
                std_rr.append(np.std(rr_intervals))
                rmssd_rr.append(np.sqrt(np.mean(np.square(np.diff(rr_intervals)))))
                heart_rates.append(60 / np.mean(rr_intervals))
                num_beats.append(len(r_peaks))
                labels_list.append(label)
        except Exception as e:
            print(f"Error extracting features from {record}: {e}")

feature_df = pd.DataFrame({
    'record': records,
    'mean_rr': mean_rr,
    'std_rr': std_rr,
    'rmssd_rr': rmssd_rr,
    'heart_rate': heart_rates,
    'num_beats': num_beats,
    'label': labels_list
})

print(feature_df.head())
feature_df.to_csv('features/rr_interval_features_extracted.csv', index=False)
print("Feature extraction complete and saved to 'features/rr_interval_features_extracted.csv'")
