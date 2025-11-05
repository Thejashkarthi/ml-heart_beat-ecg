import pandas as pd
import numpy as np

# Load the original data
df = pd.read_csv('features/hrv_features.csv')

# Identify majority and minority classes based on label counts
class_counts = df['label'].value_counts()
majority_class = class_counts.idxmax()  # Class with the most instances
minority_class = class_counts.idxmin()  # Class with the least instances

# Separate majority and minority class data
df_majority = df[df['label'] == majority_class].copy()  # Explicit copy to avoid warnings
df_minority = df[df['label'] == minority_class].copy()  # Explicit copy to avoid warnings

# Select numerical columns, excluding 'record' and 'label'
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Winsorize minority class: cap values at 1st and 99th percentiles
for col in num_cols:
    lower = df_minority[col].quantile(0.01)
    upper = df_minority[col].quantile(0.99)
    df_minority.loc[:, col] = np.clip(df_minority[col], lower, upper)  # Use .loc to avoid warnings

# Remove outliers from majority class: drop rows outside 1st and 99th percentiles
mask = pd.Series(True, index=df_majority.index)
for col in num_cols:
    lower = df_majority[col].quantile(0.01)
    upper = df_majority[col].quantile(0.99)
    mask = mask & (df_majority[col].between(lower, upper))

# Filter majority class to keep only rows without outliers
df_majority = df_majority[mask]

# Recombine the cleaned majority class with the winsorized minority class
df_cleaned = pd.concat([df_majority, df_minority], axis=0)

# Save cleaned data
df_cleaned.to_csv('features/hrv_features_outliers+wincorzin.csv', index=False)

# Print the number of data points before and after
print(f"Original dataset size: {len(df)}")
print(f"Cleaned dataset size: {len(df_cleaned)}")
print(f"Rows removed from majority class: {class_counts[majority_class] - len(df_majority)}")
print(f"Minority class size (unchanged): {len(df_minority)}")