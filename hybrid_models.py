import math, tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_bilstm_attention(input_shape, n_classes):
    """
    CNN + BiLSTM + Attention hybrid model for intrusion detection.
    - input_shape: (height, width, 1)
    - n_classes: number of output classes
    """
    inp = layers.Input(shape=input_shape)
    
    # --- CNN feature extraction ---
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2))(x)

    # --- Reshape to sequence format for BiLSTM ---
    t = x.shape[1] * x.shape[2]
    x = layers.Reshape((t, x.shape[-1]))(x)

    # --- BiLSTM for temporal pattern learning ---
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # --- Attention mechanism ---
    att = layers.Dense(1, activation='tanh')(x)
    att = layers.Flatten()(att)
    att = layers.Activation('softmax')(att)
    att = layers.RepeatVector(x.shape[-1])(att)
    att = layers.Permute([2, 1])(att)
    x = layers.multiply([x, att])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

    # --- Final dense layer ---
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    return model
