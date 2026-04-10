import streamlit as st
import os
import pandas as pd
import numpy as np
import tempfile
from predict_hrv import predict_from_file
from fpredictcnn import predict_future_abnormality

# --- 1. SET PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Pulse AI Instrument", layout="wide", page_icon="⚡")

# Safe CSS to hide Streamlit clutter, no layout hacking.
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. HEADER (Native Flow, No Fixed Positioning) ---
col_head1, col_head2, col_head3 = st.columns([1, 2, 1])
with col_head2:
    st.markdown("<h1 style='text-align: center; color: #ef4444;'>⚡ PULSE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #f8fafc; letter-spacing: 2px;'>DIAGNOSTIC INSTRUMENT</h3>", unsafe_allow_html=True)
st.divider()

# --- 3. CONFIGURATION (Sidebar) ---
with st.sidebar:
    st.markdown("### SIGNAL CONFIG")
    fs = st.number_input("Frequency (Hz)", value=250, step=50)
    st.divider()
    if st.button("Reset Session", use_container_width=True):
        st.rerun()

# --- 4. CENTRAL INTERACTION (Native Uploader) ---
st.markdown("### 1. Data Ingestion")
# Center the uploader without breaking layout
col_up_l, col_up_c, col_up_r = st.columns([1, 2, 1])
with col_up_c:
    uploaded_file = st.file_uploader("Upload ECG Array (.npy, .csv, .mat)", type=["csv", "txt", "npy", "mat"])

if uploaded_file:
    # Save temp file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    from fpredictcnn import load_signal_from_file
    
    st.divider()
    
    # 📊 Signal Preview using native Streamlit layout
    st.markdown("### 2. Signal Telemetry")
    try:
        raw_signal = load_signal_from_file(temp_path)
        preview_data = pd.DataFrame(raw_signal[:1250], columns=["Amplitude"])
        st.line_chart(preview_data, height=250)
    except Exception as e:
        st.error(f"Signal stream error: {e}")

    st.divider()

    # --- 5. DIAGNOSTIC MATRIX ---
    st.markdown("### 3. Diagnostic Modules")
    hrv_col, cnn_col = st.columns(2)
    
    # --- MODULE A: HRV ANALYSIS ---
    with hrv_col:
        st.subheader("Module 01: Clinical HRV Analysis")
        
        with st.spinner("Processing Heart Rate Variability..."):
            res1 = predict_from_file(temp_path, fs=fs)
        
        # Native Status styling
        if res1['label'] == "Normal":
            st.success(f"**STATUS:** {res1['label']} Detected")
        else:
            st.error(f"**STATUS:** {res1['label']} Detected")
            
        c1, c2 = st.columns(2)
        c1.metric("Confidence Score", f"{res1['probability_afib']*100:.1f}%")
        c2.metric("Sample Size", f"{res1['num_beats']} points")

    # --- MODULE B: PREDICTIVE WAVEFORM ---
    with cnn_col:
        st.subheader("Module 02: Neural Waveform Risk")
        
        with st.spinner("Analyzing Morphology..."):
            res2 = predict_future_abnormality(temp_path)
        
        # Native Status styling
        if res2['label'] == "Normal Risk":
            st.success(f"**STATUS:** {res2['label']}")
        else:
            st.error(f"**STATUS:** {res2['label']}")
            
        c1, c2 = st.columns(2)
        c1.metric("Future Alert Probability", f"{res2['probability']*100:.1f}%")
        c2.metric("Temporal State", "Stable" if res2['prediction'] == 0 else "Vulnerable")

    # Cleanup
    os.unlink(temp_path)

else:
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    st.info("System Ready. Awaiting ECG signal input to begin diagnostic analysis.")
