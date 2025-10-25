# =====================================================
# hybrid_models.py
# Utility functions + Hybrid Deep Learning IDS Models
# =====================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------
# 1. Utility: Load dataset
# -----------------------------------------------------
def load_csv_auto(path):
    """Load a CSV file from path or uploaded Streamlit file."""
    if hasattr(path, "read"):
        return pd.read_csv(path)
    return pd.read_csv(path)


# -----------------------------------------------------
# 2. Preprocessing
# -----------------------------------------------------
def preprocess_df(df, label_col=None):
    """Preprocess CIC-DDoS2019 dataset."""
    if label_col is None:
        # Automatically find the label column
        possible = [c for c in df.columns if "Label" in c or "label" in c]
        label_col = possible[0] if possible else df.columns[-1]

    df = df.dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Handle categorical features if present
    X = pd.get_dummies(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, le


# -----------------------------------------------------
# 3. Attention Layer
# -----------------------------------------------------
class AttentionLayer(layers.Layer):
    """Simple additive attention layer for sequence data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


# -----------------------------------------------------
# 4. Model Builders
# -----------------------------------------------------
def build_cnn_bilstm_attention(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_conv1d_bilstm_attention(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 5, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_transformer_encoder(input_shape, n_classes, num_heads=2, ff_dim=64):
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    x = layers.Add()([x, inputs])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(input_shape[-1])(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -----------------------------------------------------
# 5. Training Utilities
# -----------------------------------------------------
def compile_and_train(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, le=None):
    preds = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_ if le else None)
    print("Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    return acc, cm, report


def plot_training_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------
# 6. K-Fold Evaluation
# -----------------------------------------------------
def kfold_validate(model_fn, n_classes, X_data, y, le=None, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data), 1):
        print(f"\n===== Fold {fold} =====")
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Reshape for 1D conv input
        X_train_r = np.expand_dims(X_train, axis=-1)
        X_val_r = np.expand_dims(X_val, axis=-1)

        model = model_fn(input_shape=(X_train_r.shape[1], 1), n_classes=n_classes)
        history = compile_and_train(model, X_train_r, y_train, X_val_r, y_val, epochs=3, batch_size=64)

        acc, cm, report = evaluate_model(model, X_val_r, y_val, le=le)
        acc_scores.append(acc)

    print("\n=== Cross-Validation Results ===")
    print("Average Accuracy:", np.mean(acc_scores))
    return acc_scores


# -----------------------------------------------------
# 7. Learning Curve (Overfitting Check)
# -----------------------------------------------------
def plot_learning_curve(model_fn, n_classes, X_data, y, epochs=8):
    """Train/validation accuracy vs. sample size."""
    train_sizes = np.linspace(0.2, 0.8, 4)
    train_acc, val_acc = [], []

    for frac in train_sizes:
        X_part, _, y_part, _ = train_test_split(X_data, y, train_size=frac, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(X_part, y_part, test_size=0.2, random_state=42)

        X_tr_r = np.expand_dims(X_tr, -1)
        X_val_r = np.expand_dims(X_val, -1)

        model = model_fn((X_tr_r.shape[1], 1), n_classes)
        history = model.fit(
            X_tr_r, y_tr,
            validation_data=(X_val_r, y_val),
            epochs=epochs, batch_size=64, verbose=0
        )

        train_acc.append(history.history["accuracy"][-1])
        val_acc.append(history.history["val_accuracy"][-1])

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_acc, "-o", label="Train Acc")
    plt.plot(train_sizes, val_acc, "-o", label="Val Acc")
    plt.xlabel("Training Set Fraction")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Overfitting Diagnostics)")
    plt.legend()
    plt.grid(True)
    plt.show()
