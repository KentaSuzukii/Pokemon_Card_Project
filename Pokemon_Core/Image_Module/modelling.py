import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle

from Pokemon_Core.Image_Module.augmentation import get_augment_data
from Pokemon_Core.Image_Module import HARD_CODED_WIDTH, HARD_CODED_HEIGHT

# ─── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocessing(path: str, train_frac=0.70, val_frac=0.15):
    """
    Loads augmented DataFrame, encodes labels, vectorizes & normalizes images,
    and splits into train/val/test NumPy arrays.
    """
    # 1) load & shuffle
    df = get_augment_data(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2) encode targets
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['set_id'])
    y = to_categorical(df['target'])

    # 3) stack images (grayscale, shape (H, W, 1))
    X = np.stack(df['corner'].values, axis=0).astype(np.float32)
    if X.ndim == 3:
        X = X[..., np.newaxis]

    # 4) train/val/test split
    N = X.shape[0]
    n_train = int(train_frac * N)
    n_val   = int((train_frac + val_frac) * N)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:],    y[n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test, le

# ─── MobileNetV2 Transfer Learning Model ───────────────────────────────────────
def build_mobilenet(input_shape, num_classes, N_unfreeze=30):
    """
    Builds a MobileNetV2 model for transfer learning.
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.models import Model

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    # Unfreeze last N layers except BatchNorm
    for layer in base_model.layers[-N_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def get_class_weights(y, le):
    y_integers = np.argmax(y, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(le.classes_)), y=y_integers)
    return dict(enumerate(class_weights))

def plot_history(history):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion(cm, labels):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def mobilenet_symbols_model(
    data_path: str,
    batch_size=32,
    epochs=25,
    N_unfreeze=30,
    img_shape=(160, 160)
):
    """
    Loads data, builds and trains MobileNetV2, and returns the trained model,
    history, confusion matrix, and label encoder.
    """
    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, le = preprocessing(data_path)

    # --- Grayscale to RGB, resize ---
    def to_rgb(imgs):
        return np.repeat(imgs, 3, axis=-1)

    def resize_imgs(imgs, target_shape):
        return np.array([cv2.resize(img, target_shape) for img in imgs])

    X_train_rgb = to_rgb(X_train)
    X_val_rgb   = to_rgb(X_val)
    X_test_rgb  = to_rgb(X_test)

    X_train_resized = resize_imgs(X_train_rgb, img_shape)
    X_val_resized   = resize_imgs(X_val_rgb, img_shape)
    X_test_resized  = resize_imgs(X_test_rgb, img_shape)

    # --- Build MobileNetV2 ---
    input_shape = (img_shape[0], img_shape[1], 3)
    num_classes = y_train.shape[1]
    model = build_mobilenet(input_shape, num_classes, N_unfreeze=N_unfreeze)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    class_weights = get_class_weights(y_train, le)

    # Callbacks
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_symbols_model.h5', save_best_only=True)
    ]

    # --- Train ---
    history = model.fit(
        X_train_resized, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_resized, y_val),
        callbacks=cbs,
        class_weight=class_weights,
        verbose=1
    )

    # --- Evaluate ---
    y_pred_prob = model.predict(X_test_resized)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    plot_history(history)
    plot_confusion(cm, le.classes_)

    # Save model and label encoder
    model.save("best_symbols_model.h5")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return model, history, cm, le
