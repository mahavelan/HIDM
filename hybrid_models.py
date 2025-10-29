import math, tensorflow as tf
from tensorflow.keras import layers, models

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
    return models.Model(inp, out)

# ---------------- Model 2: Conv1D + BiLSTM + Attention ----------------
def build_conv1d_bilstm_attention(input_length, n_classes):
    inp = layers.Input(shape=(input_length,1))
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    attention = layers.TimeDistributed(layers.Dense(1, activation='tanh'))(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(x.shape[-1])(attention)
    attention = layers.Permute([2,1])(attention)
    x = layers.multiply([x, attention])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inp, out)

# ---------------- Model 3: Transformer Encoder ----------------
def build_transformer_encoder(input_shape, n_classes, d_model=64, num_heads=4, dff=128):
    inp = layers.Input(shape=input_shape)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inp, inp)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inp + attn_output)
    ffn = layers.Dense(dff, activation='relu')(out1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(0.1)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    x = layers.GlobalAveragePooling1D()(out2)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inp, out)
