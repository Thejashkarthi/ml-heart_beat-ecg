# scripts/predict_ecg_file.py
import argparse
import numpy as np
import pandas as pd
from r_peak_detector import detect_r_peaks
from hrv_feature_extractor import extract_hrv_features
from model_loader_and_predictor import load_model_and_predict


def predict_from_file(file_path: str, fs: int = 250, out: str = None):
    # 1. Load signal
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        signal = df.iloc[:, 0].values  # First column
    elif file_path.endswith('.npy'):
        signal = np.load(file_path)
    elif file_path.endswith('.txt'):
        signal = np.loadtxt(file_path)
    else:
        raise ValueError("Supported: .csv, .npy, .txt")

    print(f"Signal loaded: {len(signal)} samples, {len(signal)/fs:.1f} seconds")

    # 2. Detect R-peaks
    r_peaks = detect_r_peaks(signal, fs=fs)
    print(f"R-peaks detected: {len(r_peaks)}")

    if len(r_peaks) < 3:
        raise ValueError("Too few R-peaks. Check signal quality.")

    # 3. Extract features
    features = extract_hrv_features(r_peaks, fs=fs)
    feature_vec = np.array(list(features.values())).reshape(1, -1)

    # 4. Predict
    print("Feature vector shape:", feature_vec.shape, "Content:", feature_vec)
    result = load_model_and_predict(feature_vec)
    result["num_beats"] = len(r_peaks) - 1  # Correct number of RR intervals
    #print("Feature vector:", feature_vec)

    # 5. Print result
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Label: {result['label']}")  # Already "AFib" or "Normal"
    print(f"AFib Probability: {result['probability_afib']:.3f}")
    print(f"Number of beats: {result['num_beats']}")
    print("="*50)

    # 6. Save to JSON
    if out:
        import json
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {out}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AFib from ECG file")
    parser.add_argument("file_path", help="Path to ECG file (.csv, .txt, .npy)")
    parser.add_argument("--fs", type=int, default=250, help="Sampling rate in Hz (default: 250)")
    parser.add_argument("--out", help="Output JSON file (optional)")
    args = parser.parse_args()

    predict_from_file(args.file_path, fs=args.fs, out=args.out)