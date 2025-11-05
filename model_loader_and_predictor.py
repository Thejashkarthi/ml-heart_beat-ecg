# scripts/model_loader_and_predictor.py
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from hrv_feature_extractor import FEATURE_COLUMNS  # Import FEATURE_COLUMNS

def load_model_artifacts(model_dir: str = "models", timestamp: str = None):
    """Load model, encoder, feature columns, and optional scaler."""
    model_dir = Path(model_dir)

    if timestamp is None:
        model_files = list(model_dir.glob("xgboost_afib_model_*.joblib"))
        if not model_files:
            raise FileNotFoundError("No XGBoost model found in models/ folder")
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        timestamp = "_".join(latest.stem.split("_")[-2:])
    else:
        latest = model_dir / f"xgboost_afib_model_{timestamp}.joblib"

    model = joblib.load(latest)
    encoder = joblib.load(model_dir / f"label_encoder_{timestamp}.joblib")
    feat_cols = joblib.load(model_dir / f"feature_columns_{timestamp}.joblib")
    scaler = None
    if (model_dir / f"scaler_{timestamp}.joblib").exists():
        scaler = joblib.load(model_dir / f"scaler_{timestamp}.joblib")

    return model, encoder, feat_cols, scaler, timestamp


# ----------------------------------------------------------------------
# PUBLIC API – used by predict_ecg_file.py / retrain-predict_ecg_file.py
# ----------------------------------------------------------------------
def load_model_and_predict(feature_vector):
    """
    Load the **latest** model and return a prediction.

    Parameters
    ----------
    feature_vector : dict, pd.Series or 1-D np.ndarray
        Output of extract_hrv_features (28 values).

    Returns
    -------
    dict
        {
            "label": "AFib" or "Normal",
            "probability_afib": float,
            "num_beats": int  ← Must be set by caller
        }
    """
    model_dir = Path(__file__).parent.parent / "models"
    model_files = list(model_dir.glob("xgboost_afib_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError("No model in models/ folder!")

    latest_path = max(model_files, key=lambda p: p.stat().st_mtime)
    timestamp = "_".join(latest_path.stem.split("_")[-2:])
    print(f"Loading model: {latest_path.name}")

    # Load model and artifacts
    model = joblib.load(latest_path)
    encoder = joblib.load(model_dir / f"label_encoder_{timestamp}.joblib")
    feat_cols = joblib.load(model_dir / f"feature_columns_{timestamp}.joblib")
    scaler = None
    if (model_dir / f"scaler_{timestamp}.joblib").exists():
        scaler = joblib.load(model_dir / f"scaler_{timestamp}.joblib")

    # Build 2D DataFrame (1 row, 28 columns)
    if isinstance(feature_vector, (dict, pd.Series)):
        X = pd.DataFrame([feature_vector]).reindex(columns=feat_cols, fill_value=0)
    else:  # NumPy array case
        # Ensure feature_vector is 2D and use FEATURE_COLUMNS for initial labeling
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        X = pd.DataFrame(feature_vector, columns=FEATURE_COLUMNS).reindex(columns=feat_cols, fill_value=0)
    print("Input to model X:", X)  # Debug print

    if scaler:
        X = scaler.transform(X)

    # CORRECT: Use probability directly
    proba = model.predict_proba(X)[0, 1]  # P(AFIB) since classes=['A','N']
    proba=1-proba
    label = "AFib" if proba >= 0.5 else "Normal"

    return {
        "label": label,
        "probability_afib": float(proba),
        "num_beats": None  # ← Must be filled by caller
    }


# ----------------------------------------------------------------------
# Stand-alone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy = {col: 0.5 for col in [
        'mean_rr', 'median_rr', 'std_rr', 'min_rr', 'max_rr', 'cv_rr',
        'mad_rr', 'iqr_rr', 'rr_range', 'heart_rate_mean', 'heart_rate_min',
        'heart_rate_max', 'heart_rate_std', 'num_beats', 'sdnn', 'rmssd',
        'sdsd', 'pnn50', 'pnn20', 'mean_abs_diff', 'median_abs_diff',
        'max_abs_diff', 'sd1', 'sd2', 'sd_ratio', 'shannon_entropy',
        'triangular_index', 'turning_points_ratio'
    ]}

    result = load_model_and_predict(dummy)
    result["num_beats"] = 22  # Simulate

    print("\n=== STAND-ALONE TEST ===")
    print(f"Label               : {result['label']}")
    print(f"AFib probability    : {result['probability_afib']:.4f}")
    print(f"Number of beats     : {result['num_beats']}")