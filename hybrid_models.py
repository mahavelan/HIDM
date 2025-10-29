# hybrid_models.py
# Smart Adaptive Intrusion Detection System (SA-IDS)
# Based on the Friday DDoS dataset and HIDM hybrid concept

import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Data Preprocessing
# ----------------------------
def preprocess_data(df, label_col=' Label'):
    if label_col not in df.columns:
        label_col = df.columns[-1]

    y = df[label_col].astype(str).str.strip().replace({'nan': np.nan, 'NaN': np.nan, '': np.nan}).fillna('BENIGN').str.upper()
    X = df.drop(columns=[label_col])

    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    X = np.clip(X.values.astype(np.float32), -1e9, 1e9)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, le, scaler

# ----------------------------
# CNN + BiLSTM Hybrid Model
# ----------------------------
def build_cnn_bilstm_attention(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2))(x)
    t = x.shape[1] * x.shape[2]
    x = layers.Reshape((t, x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    att = layers.Dense(1, activation='tanh')(x)
    att = layers.Flatten()(att)
    att = layers.Activation('softmax')(att)
    att = layers.RepeatVector(x.shape[-1])(att)
    att = layers.Permute([2, 1])(att)
    x = layers.multiply([x, att])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------------------
# Helper Functions
# ----------------------------
def reshape_for_cnn(X):
    n_samples, n_features = X.shape
    s = int(math.ceil(math.sqrt(n_features)))
    target_size = s * s
    if target_size > n_features:
        Xp = np.pad(X, ((0, 0), (0, target_size - n_features)), 'constant')
    else:
        Xp = X[:, :target_size]
    return Xp.reshape((n_samples, s, s, 1)).astype(np.float32)

def train_or_load_model(X, y, n_classes, epochs=3):
    X_reshaped = reshape_for_cnn(X)
    model = build_cnn_bilstm_attention(X_reshaped.shape[1:], n_classes)
    model.fit(X_reshaped, y, epochs=epochs, batch_size=64, verbose=2)
    return model

# ----------------------------
# Severity Scoring
# ----------------------------
def compute_severity_score(row):
    features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow IAT Mean', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean'
    ]
    total = 0
    count = 0
    for f in features:
        if f in row.index:
            total += abs(row[f])
            count += 1
    if count == 0:
        return 0
    score = min(100, (total / count) / 1000 * 100)
    return round(score, 2)

# ----------------------------
# Prediction with Severity
# ----------------------------
def predict_with_severity(model, X, le, df_original):
    X_reshaped = reshape_for_cnn(X)
    preds = model.predict(X_reshaped, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    labels = le.inverse_transform(y_pred)

    results = []
    for i, label in enumerate(labels):
        score = compute_severity_score(df_original.iloc[i])
        results.append({
            "Prediction": label,
            "Severity": score
        })
    return results
