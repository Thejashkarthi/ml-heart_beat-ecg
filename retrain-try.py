import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
# Load preprocessed data
features = pd.read_csv('features/rr_interval_features.csv')
labels = pd.read_csv('data/REFERENCE-v3.csv', header=None, names=['record', 'label'])
# Merge features and labels
df = pd.merge(features, labels, on='record')
# Filter to keep only classes 'N' (majority) and 'A' (minority)
df = df[df['label'].isin(['N', 'A'])].reset_index(drop=True)
X = df.drop(columns=['record', 'label'])
y = df['label']
# Encode labels to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Define undersampler to reduce majority class to minority class count
# sampling_strategy='auto' sets majority to minority class count
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y_encoded)
print("Original class distribution:", np.bincount(y_encoded))
print("Resampled class distribution:", np.bincount(y_resampled))
# Optionally convert labels back to original classes for convenience
y_resampled_labels = le.inverse_transform(y_resampled)
# Save or return resampled data as needed
# For example, save to CSV temporarily:
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['label'] = y_resampled_labels
resampled_df.to_csv('features/rr_interval_features_undersampled.csv', index=False)
print("Undersampled data saved to 'features/rr_interval_features_undersampled.csv'") 