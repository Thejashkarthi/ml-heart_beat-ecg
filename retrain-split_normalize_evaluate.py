import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from datetime import datetime

# Load the processed dataset
df_cleaned = pd.read_csv('features/hrv_features_outliers+wincorzin.csv')

# Check feature scales to decide if normalization is needed
print("Feature scales before undersampling:")
print(df_cleaned.select_dtypes(include=['float64', 'int64']).describe())

# Encode string labels ('A', 'N') to numerical (0, 1)
label_encoder = LabelEncoder()
df_cleaned['label'] = label_encoder.fit_transform(df_cleaned['label'])
print("\nLabel encoding mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Prepare features and target
X = df_cleaned.drop(columns=['label', 'record'])  # Drop non-feature columns
y = df_cleaned['label']

# Apply RandomUnderSampler to balance the dataset (1:1 ratio)
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)

# Create a balanced DataFrame
df_balanced = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name='label')], axis=1)

# Split into training and testing sets (80% train, 20% test)
test_size = 0.2
train_df, test_df = train_test_split(df_balanced, test_size=test_size, random_state=42, stratify=df_balanced['label'])

# Decide whether to normalize (set to False for XGBoost default)
normalize = False  # Change to True if feature scales vary widely

if normalize:
    # Select numerical columns for normalization (excluding label)
    num_cols = [col for col in train_df.select_dtypes(include=['float64', 'int64']).columns if col != 'label']
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    print("\nFeature scales after normalization (training set):")
    print(train_df[num_cols].describe())
else:
    scaler = None  # Define scaler as None if not normalizing

# Ensure directories exist for saving
os.makedirs('features', exist_ok=True)

# Save the datasets
train_df.to_csv('features/hrv_features_balanced_train.csv', index=False)
test_df.to_csv('features/hrv_features_balanced_test.csv', index=False)

# Prepare features and target for training
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# Define hyperparameter grid for XGBoost tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200]
}

# Initialize XGBoost classifier
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# Perform k-fold cross-validation with grid search
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=kfold,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest hyperparameters from k-fold CV:", grid_search.best_params_)
print(f"Best ROC AUC from k-fold CV: {grid_search.best_score_:.4f}")

# Train final model with best parameters
final_model = XGBClassifier(**grid_search.best_params_, random_state=42, eval_metric='logloss')
final_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Assuming 1 is the positive class

# Decode predictions for classification report (for interpretability)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

print("\nTest Set Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, target_names=label_encoder.classes_))
print(f"Test Set ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save feature importances
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values(by='importance', ascending=False)
feature_importance.to_csv('features/xgboost_feature_importance.csv', index=False)
print("\nFeature Importances:")
print(feature_importance)

# Print summary
print(f"\nOriginal dataset size: {len(df_cleaned)}")
print(f"Balanced dataset size: {len(df_balanced)}")
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Class distribution in training set:\n{train_df['label'].value_counts()}")
print(f"Class distribution in test set:\n{test_df['label'].value_counts()}")

# --- Model Saving Section ---
# Create models directory if it doesn't exist
artifacts_dir = 'models'
os.makedirs(artifacts_dir, exist_ok=True)

# Timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the model
model_path = f'{artifacts_dir}/xgboost_afib_model_{timestamp}.joblib'
joblib.dump(final_model, model_path)
print(f"\nTrained model saved to: {model_path}")

# Save the label encoder
encoder_path = f'{artifacts_dir}/label_encoder_{timestamp}.joblib'
joblib.dump(label_encoder, encoder_path)
print(f"Label encoder saved to: {encoder_path}")

# Save the scaler if normalization was used
if normalize and scaler is not None:
    scaler_path = f'{artifacts_dir}/scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

# Save feature columns for reference (to ensure consistent ordering during inference)
feature_cols_path = f'{artifacts_dir}/feature_columns_{timestamp}.joblib'
joblib.dump(X_train.columns.tolist(), feature_cols_path)
print(f"Feature columns saved to: {feature_cols_path}")