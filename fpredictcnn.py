# fpredictcnn.py - CNN-based Future Risk Prediction
import pandas as pd
import numpy as np
import os
import io
import contextlib
import tensorflow as tf
from tensorflow import keras

# --- 1. Define Constants ---
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'models', 'cnn_future_model.keras')
SIGNAL_LENGTH = 30000                 # MUST match the training script

def load_signal_from_file(file_path):
    """
    Loads a signal from .csv, .npy, or .mat file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the file: {file_path}")
        
    if file_path.endswith('.csv'):
        new_signal_full = pd.read_csv(file_path, skiprows=1, header=0, names=['signal']).values
        return new_signal_full.flatten()
    elif file_path.endswith('.npy'):
        new_signal_full = np.load(file_path)
        return new_signal_full.flatten()
    elif file_path.endswith('.mat'):
        import scipy.io
        mat = scipy.io.loadmat(file_path)
        # Find the signal key (usually 'val' or something similar)
        signal_key = [k for k in mat.keys() if not k.startswith('__')][0]
        return mat[signal_key].flatten()
    else:
        raise ValueError(f"Unknown file type: {file_path}. Please use .csv, .npy, or .mat")

def prepare_signal(signal, target_length):
    if len(signal) == target_length:
        return signal
    elif len(signal) > target_length:
        return signal[:target_length]
    else:
        return np.pad(signal, (0, target_length - len(signal)), 'constant', constant_values=0)

def predict_future_abnormality(file_path):
    """
    Loads a new ECG, prepares it, normalizes it, and predicts its future.
    Returns a dictionary of results.
    """
    logs = io.StringIO()
    result = {"prediction": None, "probability": 0.0, "label": "Error", "raw_logs": ""}

    with contextlib.redirect_stdout(logs):
        try:
            # --- 1. Load Model ---
            print("Loading CNN model...")
            loaded_model = keras.models.load_model(MODEL_FILE)

            # --- 2. Load and Prepare Signal ---
            new_signal_raw = load_signal_from_file(file_path)
            
            # Prepare the signal to 30,000 points
            new_signal_prepared = prepare_signal(new_signal_raw, SIGNAL_LENGTH)

            # --- 3. 🔧 Per-Sample Normalization ---
            if np.std(new_signal_prepared) > 0:
                signal_normalized = (new_signal_prepared - np.mean(new_signal_prepared)) / np.std(new_signal_prepared)
            else:
                signal_normalized = new_signal_prepared - np.mean(new_signal_prepared)
            
            # --- 4. Reshape for CNN input ---
            cnn_input = signal_normalized.reshape(1, SIGNAL_LENGTH, 1)

            # --- 5. Make Prediction ---
            prediction_proba = loaded_model.predict(cnn_input, verbose=0)
            proba_abnormal = float(prediction_proba[0][0])
            
            # --- 6. Set Results ---
            result["probability"] = proba_abnormal
            if proba_abnormal > 0.5:
                result["prediction"] = 1
                result["label"] = "Abnormal Risk"
            else:
                result["prediction"] = 0
                result["label"] = "Normal Risk"

        except Exception as e:
            print(f"ERROR: {e}")
            result["label"] = "Prediction Error"

    result["raw_logs"] = logs.getvalue()
    return result

if __name__ == "__main__":
    import sys
    # If user passed a file path as argument
    if len(sys.argv) > 1:
        res = predict_future_abnormality(sys.argv[1])
    else:
        default_file = os.path.join(os.path.dirname(__file__), "data", "E00002.npy")
        res = predict_future_abnormality(default_file)
    
    print(res["raw_logs"])
    print(f"Prediction: {res['label']} ({res['probability']*100:.2f}%)")
