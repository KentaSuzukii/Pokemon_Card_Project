# 📁 Pokemon_Core/Image/modelling.py

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from Pokemon_Core.CardProcessing.augmentor import load_corner_dataframe
from Pokemon_Core.config import HARD_CODED_WIDTH, HARD_CODED_HEIGHT

# ─────────────────────────────────────────────────────────────────────────────
# 📌 CNN Model Training Module
# This module loads the reduced corner dataset, prepares it for training,
# builds a simple CNN, and trains it to classify the set_id based on corner crops.
# ─────────────────────────────────────────────────────────────────────────────

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Dataset Preprocessing ───────────────────────────────────────────────────
def preprocessing(path: str, train_frac=0.70, val_frac=0.15):
    """
    Loads and splits the dataset from JSON, encodes labels,
    normalizes images, and splits into train/val/test sets.
    """
    df = load_corner_dataframe(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['set_id'])
    y = to_categorical(df['target'])

    X = np.stack(df['corner'].values, axis=0).astype(np.float32) / 255.0
    if X.ndim == 3:
        X = X[..., np.newaxis]

    N = X.shape[0]
    n_train = int(train_frac * N)
    n_val = int((train_frac + val_frac) * N)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, le

# ─── TF Dataset Wrapping ─────────────────────────────────────────────────────
def make_dataset(X, y, batch_size=32, shuffle_buffer=None, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        buf = shuffle_buffer or X.shape[0]
        ds = ds.shuffle(buf, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ─── CNN Architecture ────────────────────────────────────────────────────────
def build_cnn(input_shape, num_classes):
    model = Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ─── Training and Evaluation ──────────────────────────────────────────────────
def train_symbols_model(data_path: str, batch_size=32, epochs=20):
    """
    Trains a CNN model on the given JSON dataset and evaluates its performance.
    Returns the model, training history, confusion matrix, and label encoder.
    """
    X_train, y_train, X_val, y_val, X_test, y_test, le = preprocessing(data_path)

    train_ds = make_dataset(X_train, y_train, batch_size, training=True)
    val_ds = make_dataset(X_val, y_val, batch_size, training=False)
    test_ds = make_dataset(X_test, y_test, batch_size, training=False)

    input_shape = (HARD_CODED_HEIGHT, HARD_CODED_WIDTH, 1)
    num_classes = y_train.shape[1]
    model = build_cnn(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_symbols_model.h5', save_best_only=True)
    ]

    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=cbs)

    y_pred_prob = model.predict(test_ds)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=le.classes_))

    return model, history, cm, le
