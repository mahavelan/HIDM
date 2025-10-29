# hybrid_models.py
# Model architecture definitions and helpers

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ---------------- Model 1: CNN2D + BiLSTM + Attention ----------------
def build_cnn_bilstm_attention(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Reshape((-1, x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    att = layers.TimeDistributed(layers.Dense(1, activation='tanh'))(x)
    att = layers.Flatten()(att)
    att = layers.Activation('softmax')(att)
    att = layers.RepeatVector(x.shape[-1])(att)
    att = layers.Permute([2,1])(att)
    x = layers.multiply([x, att])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

# ---------------- Model 2: Conv1D + BiLSTM + Attention ----------------
def build_conv1d_bilstm_attention(input_length, n_classes):
    inp = layers.Input(shape=(input_length,1))
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    att = layers.TimeDistributed(layers.Dense(1, activation='tanh'))(x)
    att = layers.Flatten()(att)
    att = layers.Activation('softmax')(att)
    att = layers.RepeatVector(x.shape[-1])(att)
    att = layers.Permute([2,1])(att)
    x = layers.multiply([x, att])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

# ---------------- Model 3: Transformer Encoder ----------------
def build_transformer_encoder(input_shape, n_classes, d_model=64, num_heads=4, dff=128):
    inp = layers.Input(shape=input_shape)  # (seq_len, d_model)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inp, inp)
    attn = layers.Dropout(0.1)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inp + attn)
    ffn = layers.Dense(dff, activation='relu')(out1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(0.1)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    x = layers.GlobalAveragePooling1D()(out2)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

# ---------------- Helper: compile model ----------------
def compile_model(model):
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------- Helper: ensemble ----------------
def ensemble_average(preds_list):
    """Average probability predictions from multiple models."""
    return np.mean(preds_list, axis=0)
