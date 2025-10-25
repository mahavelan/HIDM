import streamlit as st
import pandas as pd
import numpy as np
from hybrid_models import (
    load_csv_auto, preprocess_df,
    build_cnn_bilstm_attention, build_conv1d_bilstm_attention, build_transformer_encoder,
    kfold_validate, plot_learning_curve
)

st.set_page_config(page_title="Hybrid IDS Evaluation", layout="wide")
st.title("🚀 Hybrid Deep Learning Intrusion Detection System")

st.markdown("""
This web app evaluates three hybrid deep learning models on the **CIC-DDoS2019** dataset with
**overfitting diagnostics (K-Fold + Learning Curves)**.
Upload your CSV to begin.
""")

uploaded_file = st.file_uploader("📂 Upload your CIC-DDoS2019 CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Loading and preprocessing data..."):
        df = load_csv_auto(uploaded_file)
        X_df, y, label_encoder = preprocess_df(df)
        X_data = X_df.values
        n_classes = len(np.unique(y))
        st.success(f"✅ Data loaded successfully! Samples: {df.shape[0]}, Features: {df.shape[1]}, Classes: {n_classes}")

    model_choice = st.selectbox(
        "Select Model for Evaluation",
        ["CNN + BiLSTM + Attention", "Conv1D + BiLSTM + Attention", "Transformer Encoder"]
    )

    if st.button("Run K-Fold Validation & Learning Curve"):
        with st.spinner("Training and evaluating... this may take several minutes ⏳"):
            if model_choice == "CNN + BiLSTM + Attention":
                kfold_validate(build_cnn_bilstm_attention, n_classes, X_data, y, le=label_encoder, n_splits=3)
                plot_learning_curve(build_cnn_bilstm_attention, n_classes, X_data, y)
            elif model_choice == "Conv1D + BiLSTM + Attention":
                kfold_validate(build_conv1d_bilstm_attention, n_classes, X_data, y, le=label_encoder, n_splits=3)
                plot_learning_curve(build_conv1d_bilstm_attention, n_classes, X_data, y)
            else:
                kfold_validate(build_transformer_encoder, n_classes, X_data, y, le=label_encoder, n_splits=3)
                plot_learning_curve(build_transformer_encoder, n_classes, X_data, y)
        st.success("🎯 Evaluation Complete! Scroll down to view overfitting plots.")
else:
    st.info("👆 Please upload a dataset CSV file to start.")
