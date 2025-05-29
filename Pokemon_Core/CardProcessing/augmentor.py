# 📁 Pokemon_Core/Image/augmentation.py

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import albumentations as A

from Pokemon_Core.config import (
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    HARD_CODED_WIDTH,
    HARD_CODED_HEIGHT,
    REDUCED_SET,
)

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Image Augmentation Module
# This module defines image augmentation techniques using Albumentations,
# and provides methods to load and convert corner-labeled image data
# into TensorFlow datasets suitable for training.
# ─────────────────────────────────────────────────────────────────────────────

# ─── Augmentation Pipeline ────────────────────────────────────────────────────
# Defines a sequence of probabilistic augmentations to improve generalization
# of your model by simulating real-world image variations (rotations, blur, etc).

augmenter = A.Compose([
    A.Rotate(limit=7, border_mode=cv2.BORDER_REFLECT_101, p=0.8),  # small angle rotation
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=0,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.8),  # mild shift & scale
    A.GaussianBlur(blur_limit=(5, 7), p=0.5),                        # simulates defocus
    A.RandomBrightnessContrast(p=0.5),                               # handles lighting changes
    A.HueSaturationValue(p=0.3),                                     # color tweaks (less critical for grayscale)
], p=1.0)  # always apply the full augmentation pipeline

def albumentations_augment(img: np.ndarray) -> np.ndarray:
    """
    Applies Albumentations to a grayscale image. Converts to 3-channel for
    augmentation, then reverts to single-channel.

    Args:
        img (np.ndarray): (H, W) or (H, W, 1) grayscale image

    Returns:
        np.ndarray: (H, W, 1) augmented grayscale image
    """
    if img.ndim == 2:
        img3 = np.stack([img]*3, axis=-1)  # create 3 channels from 2D input
    else:
        img3 = np.concatenate([img]*3, axis=-1)  # duplicate channels if already has 1

    out = augmenter(image=img3)['image']
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return gray[..., np.newaxis]  # add back channel dimension

# ─── Dataset Creation ─────────────────────────────────────────────────────────

def load_corner_dataframe(json_path: str) -> pd.DataFrame:
    """
    Loads a reduced JSON dataset and converts corner images into arrays.
    Also encodes set_id into numeric labels for classification.

    Args:
        json_path (str): Path to the JSON file (left/right corners)

    Returns:
        pd.DataFrame: with 'corner', 'set_id', and 'label' columns
    """
    df = pd.read_json(json_path)
    df['corner'] = df['corner'].apply(lambda v: np.array(v, dtype=np.uint8))
    df['label'] = df['set_id'].astype('category').cat.codes
    return df

def make_tf_dataset(df: pd.DataFrame, batch_size=32, shuffle=True) -> tf.data.Dataset:
    """
    Converts a DataFrame into a TensorFlow dataset with real-time augmentation.

    Args:
        df (pd.DataFrame): DataFrame with 'corner' and 'label' columns
        batch_size (int): size of each training batch
        shuffle (bool): whether to shuffle the dataset each epoch

    Returns:
        tf.data.Dataset: ready for model training
    """
    images = np.stack(df['corner'].values, axis=0)  # (N, H, W, 1)
    labels = df['label'].values                     # (N,)

    H, W = HARD_CODED_HEIGHT, HARD_CODED_WIDTH

    def _augment_tf(img, lab):
        """
        Internal wrapper: applies Albumentations using tf.numpy_function,
        and reshapes + normalizes the result.
        """
        aug = tf.numpy_function(func=albumentations_augment, inp=[img], Tout=tf.uint8)
        aug = tf.reshape(aug, [H, W, 1])
        aug = tf.cast(aug, tf.float32) / 255.0
        return aug, lab

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.map(_augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
