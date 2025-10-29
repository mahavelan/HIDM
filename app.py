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

st.set_page_config(page_title="Hybrid IDS System", layout="wide")
st.title("🚨 Intrusion Detection Hybrid Deep Learning System")

st.markdown("""
### Models Used
- **Model 1:** CNN2D + BiLSTM + Attention  
- **Model 2:** Conv1D + BiLSTM + Attention  
- **Model 3:** Transformer Encoder  
- **Final Output:** Ensemble (Soft Voting of Models 1–3)
""")

# ---------------- Upload dataset ----------------
uploaded_file = st.file_uploader("📂 Upload the IDS dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.warning("⚠️ Please upload your dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"✅ Dataset loaded: {df.shape}")

label_col = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()][0]
y = df[label_col].astype(str)
X = df.drop(columns=[label_col])
X = X.select_dtypes(include=[np.number]).fillna(0)
X = X.replace([np.inf, -np.inf], 0)

le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(np.unique(y_enc))

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if len(np.unique(y_train)) > 1:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# ---------------- Model 1 ----------------
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
with st.spinner("Training Model 1..."):
    model1.fit(X_train_img, y_train, validation_data=(X_test_img, y_test), epochs=2, batch_size=128, callbacks=[es], verbose=0)
p1 = model1.predict(X_test_img)
st.success("✅ Model 1 Completed")

# ---------------- Model 2 ----------------
X_train_c1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_c1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model2 = build_conv1d_bilstm_attention(X_train_c1.shape[1], n_classes)
model2.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("Training Model 2..."):
    model2.fit(X_train_c1, y_train, validation_data=(X_test_c1, y_test), epochs=2, batch_size=128, callbacks=[es], verbose=0)
p2 = model2.predict(X_test_c1)
st.success("✅ Model 2 Completed")

# ---------------- Model 3 ----------------
d_model = 64
X_train_r = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_r = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
proj = layers.Dense(d_model, activation='relu')
X_train_proj = proj(X_train_r).numpy()
X_test_proj = proj(X_test_r).numpy()
model3 = build_transformer_encoder((X_train_proj.shape[1], d_model), n_classes)
model3.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("Training Model 3..."):
    model3.fit(X_train_proj, y_train, validation_data=(X_test_proj, y_test), epochs=2, batch_size=128, callbacks=[es], verbose=0)
p3 = model3.predict(X_test_proj)
st.success("✅ Model 3 Completed")

# ---------------- Ensemble ----------------
st.subheader("🤝 Ensemble Results")
p_ens = (p1 + p2 + p3) / 3.0
y_ens = np.argmax(p_ens, axis=1)
acc = accuracy_score(y_test, y_ens)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_ens, average='weighted', zero_division=0)
st.write(f"✅ **Final Ensemble Accuracy:** {acc*100:.2f}%")
st.write(f"📊 Precision: {prec:.3f} | Recall: {rec:.3f} | F1-Score: {f1:.3f}")

# ---------------- Summary ----------------
st.subheader("📈 Model Summary")
summary = pd.DataFrame({
    'Model': ['CNN2D+BiLSTM+Attention', 'Conv1D+BiLSTM+Attention', 'Transformer Encoder', 'Ensemble (Combined)'],
    'Description': [
        'Extracts spatial + temporal patterns',
        'Sequential pattern detection with less computation',
        'Captures global dependencies',
        'Weighted combination of Models 1–3'
    ]
})
st.dataframe(summary)
st.success("✅ Hybrid IDS Evaluation Completed Successfully!")
