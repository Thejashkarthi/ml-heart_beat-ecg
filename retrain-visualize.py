import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('features/hrv_features.csv')

# Plot boxplots for selected features by label
features_to_plot = ['mean_rr', 'std_rr', 'heart_rate_mean', 'sdnn']  # add more features as needed

plt.figure(figsize=(15, 8))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df, x=feature, hue='label', kde=True, stat="density", common_norm=False)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()
