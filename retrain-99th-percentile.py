import pandas as pd
import numpy as np

# Load the original data
df = pd.read_csv('features/hrv_features.csv')

# Select numerical columns, excluding 'record' and 'label'
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Winsorizing: cap to 1st and 99th percentiles
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower, upper)

# Save cleaned data
df.to_csv('features/hrv_features_outlayers_removed.csv', index=False)
