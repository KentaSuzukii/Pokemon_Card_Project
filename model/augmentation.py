
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from model import NB_CARDS_PER_SET, HARD_CODED_HEIGHT, HARD_CODED_WIDTH

# ---------- Augmentation Functions ----------

def apply_blur(img: np.array) -> np.array:
    """Add random Gaussian blur to image."""
    img = cv2.convertScaleAbs(img)
    kernel_values = [5, 7]
    selected_kernel = random.choice(kernel_values)
    blurred = cv2.GaussianBlur(img, (selected_kernel, selected_kernel), 0)
    h, w = img.shape[:2]
    return blurred.reshape((h, w, 1))

def generate_augmented_image(image_np: np.array) -> np.array:
    """Augment image: rotation, shift, blur."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=7,
        width_shift_range=3/HARD_CODED_WIDTH,   # Changed to relative range for width
        height_shift_range=2/HARD_CODED_HEIGHT, # Relative for height
        preprocessing_function=apply_blur
    )
    # Ensure shape is (1, H, W, 1) for flow()
    if image_np.ndim == 3:
        image_np = np.expand_dims(image_np, axis=0)
    elif image_np.ndim == 2:
        image_np = image_np.reshape(1, image_np.shape[0], image_np.shape[1], 1)

    it = datagen.flow(image_np, batch_size=1, shuffle=True)
    augmented_image = next(it)[0]
    # Force dtype and shape
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
    h, w = augmented_image.shape[:2]
    return augmented_image.reshape((h, w, 1))

def transform_array(image_np: np.array) -> np.array:
    """Ensure each image is shape (height, width, 1)."""
    if image_np.ndim == 2:
        image_np = image_np.reshape((image_np.shape[0], image_np.shape[1], 1))
    elif image_np.ndim == 3 and image_np.shape[2] != 1:
        # If accidentally RGB, convert to grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).reshape(image_np.shape[0], image_np.shape[1], 1)
    # else assume already (h, w, 1)
    image_np = image_np.astype(np.uint8)
    return image_np

def squeeze_photo(im_array: np.array) -> np.array:
    """Remove batch dimension if present."""
    if im_array.ndim > 3:
        return np.squeeze(im_array, axis=0)
    return im_array

def visualize_aug_sample(df, n=3):
    """Show random pre/post augmentation for sanity-checking."""
    for _ in range(n):
        row = df.sample(1).iloc[0]
        orig = row['corner']
        aug = generate_augmented_image(transform_array(orig))
        plt.figure(figsize=(6,2))
        plt.subplot(1,2,1)
        plt.imshow(orig.squeeze(), cmap='gray')
        plt.title(f"Original ({row['set_id']})")
        plt.subplot(1,2,2)
        plt.imshow(aug.squeeze(), cmap='gray')
        plt.title("Augmented")
        plt.show()

def get_augment_data(dataset_path_name: str, verbose=True) -> pd.DataFrame:
    """
    Augments the number of bottom corner images in each set to NB_CARDS_PER_SET cards.
    Ensures all outputs are shape (height, width, 1) and uint8.
    """
    df = pd.read_json(dataset_path_name)
    # Convert all images to proper np.array and shape
    df['corner'] = df['corner'].apply(lambda v: transform_array(np.array(v)))

    set_size = df['set_id'].value_counts().reset_index()
    set_size.columns = ['set_id', 'count']
    set_size['num_of_aug'] = NB_CARDS_PER_SET - set_size['count']
    new_records = []

    for idx, row in set_size.iterrows():
        set_id = row['set_id']
        n_to_add = int(row['num_of_aug'])
        existing = df[df['set_id'] == set_id]
        if n_to_add <= 0:
            continue
        if verbose:
            print(f"Augmenting set {set_id}: adding {n_to_add} synthetic samples (from {len(existing)} base)")
        if len(existing) == 0:
            print(f"⚠️ Warning: No real samples for set {set_id}, skipping augmentation.")
            continue
        for i in range(n_to_add):
            # Randomly pick a base image for augmentation
            sample_row = existing.sample(1, random_state=None).iloc[0]
            image_ = sample_row['corner']
            augmented_image = generate_augmented_image(image_)
            new_records.append({
                'corner': augmented_image,
                'position': sample_row['position'],
                'set_id': sample_row['set_id'],
                'set_name': sample_row['set_name']
            })

    # Add new augmented records to the DataFrame
    if new_records:
        df_aug = pd.DataFrame(new_records)
        df = pd.concat([df, df_aug], axis=0, ignore_index=True)

    # Squeeze shape (if needed), double-check dtype/shape
    df['corner'] = df['corner'].apply(squeeze_photo)
    # Assert all shapes are correct
    shapes = df['corner'].apply(lambda x: x.shape).value_counts()
    if verbose:
        print("Image shape distribution after augmentation:\n", shapes)
    # Optionally assert all images are (H, W, 1)
    if not all([x.shape == (HARD_CODED_HEIGHT, HARD_CODED_WIDTH, 1) for x in df['corner']]):
        print("⚠️ Warning: Some images are not shape (height, width, 1). Please check for outliers.")

    # Ensure all are uint8 (or convert here)
    df['corner'] = df['corner'].apply(lambda x: x.astype(np.uint8))

    return df

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

