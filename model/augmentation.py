#!pip install albumentations opencv-python tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import albumentations as A

# ────────── CONFIG & SETUP ────────── #

# Import your constants
from model import (
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    HARD_CODED_WIDTH,
    HARD_CODED_HEIGHT,
    REDUCED_SET,
)

# 1. Define an Albumentations pipeline
augmenter = A.Compose([
    A.Rotate(limit=7, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    A.ShiftScaleRotate(
        shift_limit=0.02, scale_limit=0.02, rotate_limit=0,
        border_mode=cv2.BORDER_REFLECT_101, p=0.8
    ),
    A.GaussianBlur(blur_limit=(5,7), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
], p=1.0)

def albumentations_augment(img: np.ndarray) -> np.ndarray:
    """
    img: single-channel array shape (H, W, 1) or (H, W)
    returns: same shape, dtype=uint8
    """
    # ensure H×W×3 for Albumentations
    if img.ndim == 2:
        img3 = np.stack([img]*3, axis=-1)
    else:
        img3 = np.concatenate([img]*3, axis=-1)
    out = augmenter(image=img3)['image']
    # convert back to single-channel
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return gray[..., np.newaxis]  # shape (H, W, 1)

# ────────── DATA LOADING & PIPELINE ────────── #

def load_corner_dataframe(json_path: str) -> pd.DataFrame:
    """
    Load your reduced JSON (with 'corner' as lists), convert to arrays.
    Assumes you have already created and reduced your dataset.
    """
    df = pd.read_json(json_path)
    df['corner'] = df['corner'].apply(lambda v: np.array(v, dtype=np.uint8))
    df['label'] = df['set_id'].astype('category').cat.codes  # or any encoding you prefer
    return df

def make_tf_dataset(df: pd.DataFrame, batch_size=32, shuffle=True) -> tf.data.Dataset:
    """
    Converts DataFrame -> tf.data.Dataset with on-the-fly Albumentations augment.
    """
    # Extract arrays
    images = np.stack(df['corner'].values, axis=0)   # shape (N, H, W, 1)
    labels = df['label'].values                     # shape (N,)

    # Build base dataset
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    def _augment_tf(img, lab):
        # tf.numpy_function to call our numpy-based augmenter
        aug = tf.numpy_function(
            func=albumentations_augment,
            inp=[img],
            Tout=tf.uint8
        )
        # ensure proper shape & dtype
        aug = tf.reshape(aug, [H, W, 1])
        aug = tf.cast(aug, tf.float32) / 255.0
        return aug, lab

    # replace H, W with your actual HARD_CODED_HEIGHT/WIDTH
    H, W = HARD_CODED_HEIGHT, HARD_CODED_WIDTH

    # Map augmentation on the fly
    ds = ds.map(_augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ────────── USAGE ────────── #

if __name__ == "__main__":
    # 1. Load reduced dataset (left & right combined, or handle separately)
    df_reduced = load_corner_dataframe("path/to/dict_reduceddataset_left.json")
    # or concat left/right before this step

    # 2. Create tf.data.Dataset
    train_ds = make_tf_dataset(df_reduced, batch_size=64)

    # 3. Plug into your model
    model = ...  # your compiled tf.keras model expecting (H, W, 1) inputs
    model.fit(
        train_ds,
        epochs=20,
        steps_per_epoch= (len(df_reduced) // 64),
        # validation_data=val_ds, etc.
    )
