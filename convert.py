# convert_mat_to_npy.py
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

def convert_mat(mat_path, output_dir="converted"):
    mat_path = Path(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load .mat file
    mat = scipy.io.loadmat(mat_path)
    print(f"Keys in .mat: {list(mat.keys())}")

    # Find ECG signal (common names)
    signal = None
    for key in ['val', 'ECG', 'ecg', 'signal', 'data']:
        if key in mat:
            signal = mat[key]
            print(f"Found signal in key: '{key}'")
            break

    if signal is None:
        print("No ECG signal found!")
        return

    # Flatten to 1D
    signal = signal.flatten()

    # Save as .npy
    npy_path = output_dir / f"{mat_path.stem}.npy"
    np.save(npy_path, signal)
    print(f"Saved .npy: {npy_path}")

    # Save as .csv
    csv_path = output_dir / f"{mat_path.stem}.csv"
    pd.DataFrame(signal).to_csv(csv_path, index=False, header=False)
    print(f"Saved .csv: {csv_path}")

    print(f"Signal length: {len(signal)} samples")

# === RUN IT ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_mat_to_npy.py your_file.mat")
    else:
        convert_mat(sys.argv[1])