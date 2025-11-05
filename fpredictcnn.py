import pandas as pd
import numpy as np
import joblib 
import os
import tensorflow as tf
from tensorflow import keras
from colorama import Fore, Style, init

# Initialize colorama for colored text
init(autoreset=True)

# --- 1. Define Constants ---
MODEL_FILE = r'C:\Users\thejash\OneDrive\ドキュメント\ml\scripts\future_cnn\cnn_future_model.keras' # The new model file
SIGNAL_LENGTH = 30000                 # MUST match the training script

def load_signal_from_file(file_path):
    """
    Loads a signal from either a .csv or .npy file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the file: {file_path}")
        
    if file_path.endswith('.csv'):
        new_signal_full = pd.read_csv(file_path, skiprows=1, header=0, names=['signal']).values
        return new_signal_full.flatten()
    elif file_path.endswith('.npy'):
        new_signal_full = np.load(file_path)
        return new_signal_full.flatten()
    else:
        raise ValueError(f"Unknown file type: {file_path}. Please use .csv or .npy")

def prepare_signal(signal, target_length):
    """
    Pads or truncates a signal to the target_length.
    """
    if len(signal) == target_length:
        return signal
    elif len(signal) > target_length:
        # Truncate from the beginning
        return signal[:target_length]
    else:
        # Pad with zeros at the end
        return np.pad(signal, (0, target_length - len(signal)), 'constant', constant_values=0)

def predict_future_abnormality(file_path):
    """
    Loads a new ECG, prepares it, normalizes it, and predicts its future.
    """
    print(Fore.MAGENTA + f"\n--- PREDICTING FOR: {file_path} (CNN Model) ---")

    try:
        # --- 1. Load Model ---
        print(Fore.CYAN + "Loading CNN model...")
        loaded_model = keras.models.load_model(MODEL_FILE)

        # --- 2. Load and Prepare Signal ---
        new_signal_raw = load_signal_from_file(file_path)
        print(f"Original signal length: {len(new_signal_raw)}")
        
        # Prepare the signal to 30,000 points
        new_signal_prepared = prepare_signal(new_signal_raw, SIGNAL_LENGTH)
        print(f"Signal prepared to length: {len(new_signal_prepared)}")

        # --- 3. 🔧 Per-Sample Normalization ---
        # We MUST apply the same normalization as the training script
        if np.std(new_signal_prepared) > 0:
            signal_normalized = (new_signal_prepared - np.mean(new_signal_prepared)) / np.std(new_signal_prepared)
        else:
            signal_normalized = new_signal_prepared - np.mean(new_signal_prepared)
        
        # --- 4. Reshape for CNN input ---
        # The CNN needs (1, 30000, 1)
        cnn_input = signal_normalized.reshape(1, SIGNAL_LENGTH, 1)
        print("Signal successfully loaded, prepared, and normalized.")

        # --- 5. Make Prediction ---
        prediction_proba = loaded_model.predict(cnn_input, verbose=0)
        proba_abnormal = prediction_proba[0][0]
        
        # --- 6. Show the Result ---
        print(Fore.MAGENTA + "\n--- FINAL PREDICTION ---")
        if proba_abnormal > 0.5:
            prediction = 1
            print(Fore.YELLOW + Style.BRIGHT + f"Prediction: 1 (This ECG suggests a 'Future Abnormal' outcome)")
            print(f"Confidence (Probability of 'Abnormal'): {proba_abnormal*100:.2f}%")
        else:
            prediction = 0
            print(Fore.GREEN + Style.BRIGHT + f"Prediction: 0 (This ECG suggests a 'Future Normal' outcome)")
            print(f"Confidence (Probability of 'Normal'): {(1-proba_abnormal)*100:.2f}%")

    except FileNotFoundError as e:
        print(Fore.RED + f"--- ERROR: Cannot find a required file! ---")
        print(f"Missing: {e.filename}")
        print(Fore.YELLOW + "Please run your main 'cnn_train.py' script first.")
    except Exception as e:
        print(Fore.RED + f"--- ERROR during prediction: {e} ---")

# --- Run the prediction function for all your files ---
if __name__ == "__main__":
    import sys

    # If user passed a file path as argument → use it
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        predict_future_abnormality(file_path)
    else:
        # fallback: your default file
        default_file = r"C:\Users\thejash\OneDrive\ドキュメント\ml\scripts\future_cnn\E00002.npy"
        print(f"No file path provided. Using default file:\n{default_file}")
        predict_future_abnormality(default_file)
