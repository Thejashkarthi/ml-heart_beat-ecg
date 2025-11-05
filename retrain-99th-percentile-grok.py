import pandas as pd
import numpy as np

# Load the original data
df = pd.read_csv('features/hrv_features.csv')

# Assuming 'label' column contains class labels (e.g., 0 for minority, 1 for majority)
# Identify majority and minority classes based on label counts
class_counts = df['label'].value_counts()
majority_class = class_counts.idxmax()  # Class with the most instances
minority_class = class_counts.idxmin()  # Class with the least instances

# Separate majority and minority class data
df_majority = df[df['label'] == majority_class].copy()  # Create an explicit copy
df_minority = df[df['label'] == minority_class].copy()  # Create an explicit copy

# Select numerical columns, excluding 'record' and 'label'
num_cols = df_majority.select_dtypes(include=['float64', 'int64']).columns

# Winsorizing: cap to 1st and 99th percentiles for majority class only
for col in num_cols:
    lower = df_majority[col].quantile(0.01)
    upper = df_majority[col].quantile(0.99)
    df_majority.loc[:, col] = np.clip(df_majority[col], lower, upper)  # Use .loc for assignment

# Recombine the data
df_cleaned = pd.concat([df_majority, df_minority], axis=0)

# Save cleaned data
df_cleaned.to_csv('features/hrv_features_outliers_removed_majority.csv', index=False)