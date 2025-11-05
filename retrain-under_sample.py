import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib


features_df = pd.read_csv('data/retrained_extracted_features.csv')


features_df = features_df[features_df['label'].isin(['A', 'N'])]


label_mapping = {'N': 0, 'A': 1}
features_df['label'] = features_df['label'].map(label_mapping)


print("Class distribution before undersampling:")
print(features_df['label'].value_counts())


X = features_df.drop(columns=['record', 'label'])
y = features_df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)


print("\nClass distribution after undersampling:")
print(pd.Series(y_train_res).value_counts())

# Compute scale_pos_weight for XGBoost based on undersampled training
neg_count = sum(y_train_res == 0)
pos_count = sum(y_train_res == 1)
scale_pos_weight = neg_count / pos_count


classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
}

for name, clf in classifiers.items():
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)

    print(f"\n{name} Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    if accuracy >= 0.85 and precision >= 0.85 and recall >= 0.85 and f1 >= 0.85:
        print(f"{name} meets the 85% performance threshold on all metrics.")
    else:
        print(f"{name} does NOT meet the 85% threshold on all metrics.")

    print(classification_report(y_test, y_pred))

    # Optionally save model
    # joblib.dump(clf, f'{name.lower().replace(" ", "_")}_model.pkl')
