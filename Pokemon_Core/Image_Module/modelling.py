import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns


from tensorflow.keras import layers, callbacks, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.utils.class_weight import compute_class_weight
import pickle

from Pokemon_Core.Image_Module.augmentation import get_augment_data
from Pokemon_Core.Image_Module import HARD_CODED_WIDTH, HARD_CODED_HEIGHT



# Import get_augment_data from the correct module
from Pokemon_Core.Image_Module.augmentation import load_corner_dataframe as get_augment_data
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


    X = np.stack(df['corner'].values, axis=0).astype(np.float32) / 255.0
    if X.ndim == 3:
        X = X[..., np.newaxis]


    # 3) stack & normalize images
    X = np.stack(df['corner'].values, axis=0).astype(np.float32) / 255.0
    # ensure channel dim
    if X.ndim == 3:
        X = X[..., np.newaxis]  # (N, H, W, 1)

    # 4) train/val/test split

    N = X.shape[0]
    n_train = int(train_frac * N)
    n_val   = int((train_frac + val_frac) * N)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:],    y[n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, le




# ─── Dataset Pipeline ──────────────────────────────────────────────────────────
def make_dataset(X, y, batch_size=32, shuffle_buffer=None, training=True):
    """
    Wraps NumPy arrays into a tf.data.Dataset, with optional shuffling and batching.
    """

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        buf = shuffle_buffer or X.shape[0]
        ds = ds.shuffle(buf, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Model Definition ──────────────────────────────────────────────────────────
def build_cnn(input_shape, num_classes):
    """
    Builds a small CNN with BatchNorm, Dropout, and 3×3 kernels.
    """

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


# ─── Training & Evaluation ─────────────────────────────────────────────────────
def symbols_model(data_path: str, batch_size=32, epochs=20):
    """
    Loads data, builds and trains the CNN, and returns the trained model,
    training history, test confusion matrix, and the label encoder.
    """
    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, le = preprocessing(data_path)

    # Datasets

    train_ds = make_dataset(X_train, y_train, batch_size, training=True)
    val_ds   = make_dataset(X_val,   y_val,   batch_size, training=False)
    test_ds  = make_dataset(X_test,  y_test,  batch_size, training=False)


    # Model

    input_shape = (HARD_CODED_HEIGHT, HARD_CODED_WIDTH, 1)
    num_classes = y_train.shape[1]
    model = build_cnn(input_shape, num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    class_weights = get_class_weights(y_train, le)


    # Callbacks

    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_symbols_model.h5', save_best_only=True)
    ]


    # Train

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=cbs,

        class_weight=class_weights,
        verbose=1
    )



    # Evaluate

    y_pred_prob = model.predict(test_ds)
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
