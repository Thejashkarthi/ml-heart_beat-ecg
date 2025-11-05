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
num_cols = df_majority.select_dtypes(include=['float64', 'int64']).columns

# Initialize a mask to keep track of rows to retain (True = keep, False = remove)
mask = pd.Series(True, index=df_majority.index)

# Identify rows with outliers in majority class (outside 1st and 99th percentiles)
for col in num_cols:
    lower = df_majority[col].quantile(0.01)
    upper = df_majority[col].quantile(0.99)
    # Update mask: keep rows where values are within bounds for this column
    mask = mask & (df_majority[col].between(lower, upper))

# Filter majority class to keep only rows without outliers
df_majority = df_majority[mask]

# Recombine the cleaned majority class with the unchanged minority class
df_cleaned = pd.concat([df_majority, df_minority], axis=0)

# Save cleaned data
df_cleaned.to_csv('features/hrv_features_outliers_removed_majority.csv', index=False)

# Print the number of data points before and after
print(f"Original dataset size: {len(df)}")
print(f"Cleaned dataset size: {len(df_cleaned)}")
print(f"Rows removed from majority class: {class_counts[majority_class] - len(df_majority)}")