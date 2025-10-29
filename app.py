# app.py
# Streamlit dashboard for Smart Adaptive Intrusion Detection System (SA-IDS)

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from hybrid_models import preprocess_data, train_or_load_model, predict_with_severity

st.set_page_config(page_title="Smart Adaptive Intrusion Detection System", layout="wide")

st.title("🧠 Smart Adaptive Intrusion Detection System (SA-IDS)")
st.markdown("### Real-Time Attack Monitoring with Severity Scoring")

# ------------------------------
# Load dataset
# ------------------------------
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
else:
    st.info("Using default dataset (dataset.csv)")
    df = pd.read_csv("dataset.csv")

st.write(f"**Dataset shape:** {df.shape}")

# ------------------------------
# Preprocess and Train model (quick)
# ------------------------------
st.write("🔄 Preprocessing & Model Training...")
X, y, le, scaler = preprocess_data(df)
n_classes = len(np.unique(y))
model = train_or_load_model(X, y, n_classes, epochs=2)
st.success("✅ Model trained successfully!")

# ------------------------------
# Real-time Simulation
# ------------------------------
st.subheader("⚡ Real-Time Simulation")

simulation_speed = st.slider("Simulation Speed (seconds per batch)", 1, 10, 3)
start_button = st.button("🚀 Start Real-Time Detection")

if start_button:
    st.info("Simulation running... Please wait")
    placeholder_chart = st.empty()
    placeholder_table = st.empty()

    chunk_size = 200
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        batch_df = df.iloc[start:end]
        batch_X = scaler.transform(batch_df.select_dtypes(include=[np.number]).fillna(0.0))
        results = predict_with_severity(model, batch_X, le, batch_df)

        result_df = pd.DataFrame(results)
        benign = (result_df["Prediction"] == "BENIGN").sum()
        attacks = len(result_df) - benign

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].pie([benign, attacks], labels=["BENIGN", "ATTACK"], autopct="%1.1f%%", colors=["#8bc34a", "#f44336"])
        ax[0].set_title("Traffic Composition")

        ax[1].bar(result_df.index, result_df["Severity"], color=["#f44336" if s > 60 else "#ff9800" if s > 30 else "#8bc34a" for s in result_df["Severity"]])
        ax[1].set_title("Attack Severity per Record")
        ax[1].set_xlabel("Record Index")
        ax[1].set_ylabel("Severity (0-100)")

        placeholder_chart.pyplot(fig)
        placeholder_table.dataframe(result_df.head(10))
        time.sleep(simulation_speed)

    st.success("✅ Simulation completed!")
