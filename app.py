import streamlit as st
import subprocess
import tempfile
import os
import re
import sys  # ✅ ensures we use the same Python interpreter as Streamlit

st.set_page_config(page_title="Dual ECG Model Prediction", layout="wide")

st.title("🫀 ECG AFib Detection — Dual Model Comparison")

# Upload ECG file
uploaded_file = st.file_uploader("Upload ECG file (.csv, .txt, or .npy)", type=["csv", "txt", "npy"])
fs = st.number_input("Sampling Rate (Hz)", value=250, step=50)

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ ECG file uploaded successfully")

    # Columns for both models
    col1, col2 = st.columns(2)

    # ---------------- MODEL 1 (XGBoost / HRV) ----------------
    with col1:
        st.subheader("Model 1: Retrain + XGBoost")

        cmd1 = [
            sys.executable,  # ✅ Use same Python interpreter
            r"C:\Users\thejash\OneDrive\ドキュメント\ml\scripts\retrain-predict_ecg_file.py",
            file_path,
            "--fs",
            str(fs)
        ]

        xgb_output_clean = ""  # always define before try-block
        try:
            with st.spinner("Running Model 1..."):
                result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)

            # Combine stdout + stderr
            xgb_output = (result1.stdout or "") + "\n" + (result1.stderr or "")

            # Remove ANSI color codes
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            xgb_output_clean = ansi_escape.sub('', xgb_output)

        except Exception as e:
            xgb_output_clean = f"❌ Error running Model 1: {str(e)}"

        st.text_area("Output (Model 1)", xgb_output_clean.strip(), height=300)

    # ---------------- MODEL 2 (CNN) ----------------
    with col2:
        st.subheader("Model 2: CNN Future Model")

        cmd2 = [
            sys.executable,  # ✅ Use same Python interpreter
            r"C:\Users\thejash\OneDrive\ドキュメント\ml\scripts\future_cnn\fpredictcnn.py",
            file_path
        ]

        cnn_output_clean = ""  # always define before try-block
        try:
            with st.spinner("Running Model 2..."):
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)

            # Combine stdout + stderr
            cnn_output = (result2.stdout or "") + "\n" + (result2.stderr or "")

            # Remove ANSI color codes
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            cnn_output_clean = ansi_escape.sub('', cnn_output)

        except Exception as e:
            cnn_output_clean = f"❌ Error running Model 2: {str(e)}"

        st.text_area("Output (Model 2)", cnn_output_clean.strip(), height=300)
