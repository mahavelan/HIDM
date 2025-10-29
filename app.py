# Disable watchdog to avoid Hugging Face space errors
import os
os.environ["STREAMLIT_WATCHDOG_ENABLED"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers, optimizers, callbacks
from hybrid_models import (
    build_cnn_bilstm_attention,
    build_conv1d_bilstm_attention,
    build_transformer_encoder
)

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Hybrid IDS System", layout="wide")
st.title("🚨 Intrusion Detection Hybrid Deep Learning System")

st.markdown("""
### Models Used
- **Model 1:** CNN2D + BiLSTM + Attention  
- **Model 2:** Conv1D + BiLSTM + Attention  
- **Model 3:** Transformer Encoder  
- **Final Output:** Ensemble (Soft Voting of Models 1–3)
""")

# ---------------- Dataset Upload ----------------
uploaded_file = st.file_uploader("📂 Upload your CIC-IDS2019 CSV dataset", type=["csv"])
if uploaded_file is None:
    st.warning("⚠️ Please upload a dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# Identify label column
label_col = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()][0]
y = df[label_col].astype(str)
X = df.drop(columns=[label_col])
X = X.select_dtypes(include=[np.number]).fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# Encode and scale
le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(np.unique(y_enc))

X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if len(np.unique(y_train)) > 1:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# ---------- Model 1: CNN-BiLSTM-Attention ----------
s = int(math.ceil(math.sqrt(X_train.shape[1])))
def reshape_2d(X):
    if X.shape[1] < s*s:
        X = np.pad(X, ((0,0),(0,s*s - X.shape[1])), 'constant')
    return X.reshape((X.shape[0], s, s, 1)).astype(np.float32)

X_train_img = reshape_2d(X_train)
X_test_img = reshape_2d(X_test)

model1 = build_cnn_bilstm_attention((s, s, 1), n_classes)
model1.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
with st.spinner("🔹 Training Model 1 (CNN2D+BiLSTM+Attention)..."):
    model1.fit(X_train_img, y_train, validation_data=(X_test_img, y_test), epochs=2, batch_size=128, callbacks=[es], verbose=0)
p1 = model1.predict(X_test_img)
st.success("✅ Model 1 training complete.")

# ---------- Model 2: Conv1D-BiLSTM-Attention ----------
X_train_c1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_c1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model2 = build_conv1d_bilstm_attention(X_train_c1.shape[1], n_classes)
model2.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("🔹 Training Model 2 (Conv1D+BiLSTM+Attention)..."):
    model2.fit(X_train_c1, y_train, validation_data=(X_test_c1, y_test), epochs=2, batch_size=128, callbacks=[es], verbose=0)
p2 = model2.predict(X_test_c1)
st.success("✅ Model 2 training complete.")

# ---------- Model 3: Transformer Encoder ----------
d_model = 64
X_train_r = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_r = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
proj = layers.Dense(d_model, activation='relu')
X_train_proj = proj(X_train_r).numpy()
X_test_proj = proj(X_test_r).numpy()
model3 = build_transformer_encoder((X_train_proj.shape[1], d_model), n_classes)
model3.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("🔹 Training Model 3 (Transformer Encoder)..."):
    model3.fit(X_train_proj, y_train, validation_data=(X_test_proj, y_test), epochs=2, batch_size=128, callbacks=[es], verbose=0)
p3 = model3.predict(X_test_proj)
st.success("✅ Model 3 training complete.")

# ---------- Ensemble Result ----------
st.subheader("🤝 Final Ensemble Results (Soft Voting)")
p_ens = (p1 + p2 + p3) / 3.0
y_ens = np.argmax(p_ens, axis=1)
acc = accuracy_score(y_test, y_ens)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_ens, average='weighted', zero_division=0)

st.write(f"🎯 **Final Ensemble Accuracy:** {acc*100:.2f}%")
st.write(f"📊 Precision: {prec:.3f} | Recall: {rec:.3f} | F1-Score: {f1:.3f}")

st.markdown("""
### ✅ Summary
- Combines **CNN2D**, **Conv1D**, and **Transformer Encoder** models.  
- Uses **Attention mechanism** for deep feature extraction.  
- Ensemble voting ensures higher stability and better detection accuracy.  
""")

st.success("✅ Intrusion Detection Evaluation Completed Successfully!")
