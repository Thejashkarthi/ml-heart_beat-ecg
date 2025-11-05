import pandas as pd

# Load the dataset
df = pd.read_csv('features/hrv_features_outlayers_removed.csv')

# Select numerical columns (excluding 'record' and 'label')
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Dictionary to store outliers info
outliers_info = {}

# Detect outliers in each numerical column
for col in numerical_cols:
    outliers = detect_outliers_iqr(df[col])
    outliers_info[col] = outliers.sum()
    print(f'Outliers detected in {col}: {outliers.sum()}')

# Optional: Summary of features with outliers
print("\nSummary of outliers in each feature:")
for feature, count in outliers_info.items():
    print(f"{feature}: {count}")
