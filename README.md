# ⚡ Pulse AI: Dual-Model ECG Diagnostic Instrument

Pulse AI is an advanced, production-ready machine learning system built to classify ECG (electrocardiogram) time-series data. It features a modern, real-time diagnostic dashboard that utilizes a dual-model approach to detect arrhythmias (like AFib) and predict future cardiovascular vulnerability.

---

## 🧠 Core Architecture (Dual-Model Pipeline)

This project implements the **"Operation Flower"** analytics pipeline, utilizing two distinct ML architectures to cross-verify signals:
1. **Module 01 (Clinical HRV Analysis - XGBoost):** Extracts exact physiological features (Heart Rate Variability, Shannon Entropy, Winsorization) using `neurokit2` from detected R-peaks. This tabular data is fed into an extreme gradient boosted decision tree (XGBoost) to detect real-time atrial fibrillation.
2. **Module 02 (Neural Waveform Risk - CNN):** Processes raw, temporal signal distributions up to 30,000 points. A 1D Convolutional Neural Network (TensorFlow/Keras) scans for future morphological vulnerabilities missed by standard HRV features.

---

## 📁 Project Structure

The repository has been refactored for professional production use:

```text
/
├── app.py                      # Main Streamlit Dashboard (Native Premium UI)
├── fpredictcnn.py              # CNN Inference Logic
├── hrv_feature_extractor.py    # HRV Feature Extraction & Mathematical Engine
├── model_loader_and_predictor.py# XGBoost Inference Engine
├── r_peak_detector.py          # Core Signal Processing 
├── predict_hrv.py              # End-to-end HRV processing pipeline
├── data/                       # Sample ECG streams (.npy, .mat, .csv)
├── models/                     # Compiled ML artifacts (.joblib, .keras, .pkl)
├── scripts/                    # Training and preprocessing scripts
└── assets/                     # UI diagrams and logic flow images
```

---

## 🚀 How to Run the Dashboard

The instrument runs entirely locally via a fast, native Streamlit interface.

### 1. Install Dependencies
Make sure you have Python 3.10+ installed.
```bash
pip install streamlit pandas numpy tensorflow xgboost neurokit2 scikit-learn joblib scipy
```

### 2. Launch the Application
Start the Pulse AI diagnostic instrument.
```bash
streamlit run app.py
```

### 3. Usage
- Once the browser opens (`http://localhost:8501`), drop an ECG file into the central ingestion hub.
- You can find sample inputs in the `/data/` folder (e.g., `E00001.mat`, `A00015.npy`).
- The system will automatically generate a real-time signal trace and deploy both AI modules to analyze the signal.

---

## 📊 Key Features
- **Real-Time Signal Telemetry**: Immediate visual traces of ingested time-series data.
- **Zero-Latency Module Imports**: Models bypass terminal overhead, importing directly for maximum UI stability.
- **Dynamic Portability**: No hardcoded paths; models detect execution context automatically.
- **Premium Native UI**: Dark-mode Streamlit metrics and dynamic layout engines for surgical precision.
