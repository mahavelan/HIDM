import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from hybrid_models import build_cnn_bilstm_attention
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hybrid IDS System", layout="wide")

st.title("🚨 Cybersecurity Intrusion Detection System (IDS)")
st.markdown("**Detect and analyze DDoS attacks using a hybrid CNN + BiLSTM + Attention model**")

# --- Load Dataset ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

try:
    df = load_data("dataset.csv")
    st.success(f"✅ Loaded dataset with shape {df.shape}")
except Exception as e:
    st.error("❌ Dataset not found! Please add `dataset.csv` in the same folder.")
    st.stop()

# --- Preprocessing ---
label_col = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()][0]
y = df[label_col].astype(str).str.upper()
X = df.drop(columns=[label_col])

# Drop non-numeric
X = X.select_dtypes(include=[np.number]).fillna(0)
X = X.replace([np.inf, -np.inf], 0)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Scaling and balancing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
if len(np.unique(y_train)) > 1:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# Reshape for CNN
s = int(math.ceil(math.sqrt(X_train.shape[1])))
def reshape(X):
    if X.shape[1] < s*s:
        X = np.pad(X, ((0,0),(0,s*s - X.shape[1])), 'constant')
    return X.reshape((X.shape[0], s, s, 1)).astype(np.float32)

X_train_r = reshape(X_train)
X_test_r = reshape(X_test)

# --- Model Training ---
model = build_cnn_bilstm_attention((s, s, 1), len(np.unique(y_enc)))
model.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

with st.spinner("⏳ Training model... Please wait"):
    history = model.fit(X_train_r, y_train, validation_data=(X_test_r, y_test), epochs=5, batch_size=128, callbacks=[es], verbose=0)

st.success("✅ Model training completed!")

# --- Evaluation ---
preds = np.argmax(model.predict(X_test_r), axis=1)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0, output_dict=True)
cm = confusion_matrix(y_test, preds)

st.metric("Model Accuracy", f"{acc*100:.2f}%")

st.subheader("Classification Report")
st.dataframe(pd.DataFrame(report).T)

st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

# --- Plot Training Curves ---
st.subheader("Training Performance")
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Val Loss')
ax[0].legend(); ax[0].set_title('Loss Curve')
ax[1].plot(history.history['accuracy'], label='Train Acc')
ax[1].plot(history.history['val_accuracy'], label='Val Acc')
ax[1].legend(); ax[1].set_title('Accuracy Curve')
st.pyplot(fig)

# --- Real-time Prediction ---
st.subheader("🔎 Predict from New Input")
uploaded = st.file_uploader("Upload a small CSV sample for testing", type=['csv'])
if uploaded:
    new_df = pd.read_csv(uploaded)
    new_df = new_df.select_dtypes(include=[np.number]).fillna(0)
    new_df = scaler.transform(new_df)
    new_df_r = reshape(new_df)
    preds_new = np.argmax(model.predict(new_df_r), axis=1)
    preds_labels = le.inverse_transform(preds_new)
    st.write("Predictions:", preds_labels)
