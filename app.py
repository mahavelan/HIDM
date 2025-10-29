import streamlit as st
import pandas as pd
import numpy as np
import math, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers, optimizers, callbacks
from hybrid_models import build_cnn_bilstm_attention, build_conv1d_bilstm_attention, build_transformer_encoder

st.set_page_config(page_title="Hybrid IDS System", layout="wide")

st.title("🚨 Intrusion Detection Hybrid Models (CNN, LSTM, Transformer)")
st.markdown("**CIC-DDoS2019 / IDS2017 Dataset — Hybrid Deep Learning Models for Attack Detection**")

# -------------------- Upload Dataset --------------------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
if uploaded_file is None:
    st.warning("⚠️ Please upload the dataset CSV file to continue.")
    st.stop()

# -------------------- Load Data --------------------
df = pd.read_csv(uploaded_file)
st.success(f"✅ Loaded dataset with shape {df.shape}")

label_col = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()][0]
y = df[label_col].astype(str).str.upper()
X = df.drop(columns=[label_col])
X = X.select_dtypes(include=[np.number]).fillna(0)
X = X.replace([np.inf, -np.inf], 0)

le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(np.unique(y_enc))

# -------------------- Split, Scale, Balance --------------------
X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
if len(np.unique(y_train)) > 1:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# -------------------- Utility: Confusion Matrix --------------------
def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# -------------------- Model 1: CNN2D + BiLSTM + Attention --------------------
st.subheader("🧠 Model 1: CNN2D + BiLSTM + Attention")
s = int(math.ceil(math.sqrt(X_train.shape[1])))
def reshape_2d(X):
    if X.shape[1] < s*s:
        X = np.pad(X, ((0,0),(0,s*s - X.shape[1])), 'constant')
    return X.reshape((X.shape[0], s, s, 1)).astype(np.float32)
X_train_img = reshape_2d(X_train)
X_test_img = reshape_2d(X_test)
model1 = build_cnn_bilstm_attention((s, s, 1), n_classes)
model1.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
with st.spinner("Training Model 1..."):
    model1.fit(X_train_img, y_train, validation_data=(X_test_img, y_test), epochs=5, batch_size=128, callbacks=[es], verbose=0)
p1 = model1.predict(X_test_img)
y_pred1 = np.argmax(p1, axis=1)
acc1 = accuracy_score(y_test, y_pred1)
st.write(f"✅ Accuracy: **{acc1*100:.2f}%**")
plot_confusion(y_test, y_pred1, le.classes_, "Model 1 Confusion Matrix")

# -------------------- Model 2: Conv1D + BiLSTM + Attention --------------------
st.subheader("⚙️ Model 2: Conv1D + BiLSTM + Attention")
X_train_c1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_c1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model2 = build_conv1d_bilstm_attention(X_train_c1.shape[1], n_classes)
model2.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("Training Model 2..."):
    model2.fit(X_train_c1, y_train, validation_data=(X_test_c1, y_test), epochs=5, batch_size=128, callbacks=[es], verbose=0)
p2 = model2.predict(X_test_c1)
y_pred2 = np.argmax(p2, axis=1)
acc2 = accuracy_score(y_test, y_pred2)
st.write(f"✅ Accuracy: **{acc2*100:.2f}%**")
plot_confusion(y_test, y_pred2, le.classes_, "Model 2 Confusion Matrix")

# -------------------- Model 3: Transformer Encoder --------------------
st.subheader("🔁 Model 3: Transformer Encoder")
d_model = 64
X_train_r = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_r = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
proj = layers.Dense(d_model, activation='relu')
X_train_proj = proj(X_train_r).numpy()
X_test_proj = proj(X_test_r).numpy()
model3 = build_transformer_encoder((X_train_proj.shape[1], d_model), n_classes)
model3.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("Training Model 3..."):
    model3.fit(X_train_proj, y_train, validation_data=(X_test_proj, y_test), epochs=5, batch_size=128, callbacks=[es], verbose=0)
p3 = model3.predict(X_test_proj)
y_pred3 = np.argmax(p3, axis=1)
acc3 = accuracy_score(y_test, y_pred3)
st.write(f"✅ Accuracy: **{acc3*100:.2f}%**")
plot_confusion(y_test, y_pred3, le.classes_, "Model 3 Confusion Matrix")

# -------------------- Ensemble Model --------------------
st.subheader("🤝 Ensemble (Soft Voting of Models 1–3)")
p_ens = (p1 + p2 + p3) / 3.0
y_ens = np.argmax(p_ens, axis=1)
acc_ens = accuracy_score(y_test, y_ens)
st.write(f"✅ Ensemble Accuracy: **{acc_ens*100:.2f}%**")
plot_confusion(y_test, y_ens, le.classes_, "Ensemble Confusion Matrix")

# -------------------- Summary --------------------
st.subheader("📊 Model Accuracy Comparison")
summary = pd.DataFrame({
    'Model': ['CNN2D+BiLSTM+Attn', 'Conv1D+BiLSTM+Attn', 'Transformer', 'Ensemble'],
    'Accuracy (%)': [acc1*100, acc2*100, acc3*100, acc_ens*100]
})
st.dataframe(summary)
