# 📁 Pokemon_Core/Image/prediction.py

import cv2
import numpy as np
import tensorflow as tf
import logging
from dataclasses import dataclass

from Pokemon_Core.config import (
    INITIAL_WIDTH, INITIAL_HEIGHT,
    HARD_CODED_WIDTH, HARD_CODED_HEIGHT,
    HIRES_WIDTH, HIRES_HEIGHT, RATIO,
    SETINFO
)
from Pokemon_Core.CardProcessing.card_aligner import deform_card

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Prediction & Preprocessing Module
# Prepares user-uploaded Pokémon card images by extracting bottom corners,
# and handles high-res cropping for OCR.
# Applies perspective correction via deform_card() if needed.
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Global Configuration for Image Dimensions ───────────────────────────────
@dataclass(frozen=True)
class CardConfig:
    init_w: int = INITIAL_WIDTH
    init_h: int = INITIAL_HEIGHT
    crop_w: int = HARD_CODED_WIDTH
    crop_h: int = HARD_CODED_HEIGHT
    ratio:  int = RATIO
    hires_w: int = HIRES_WIDTH
    hires_h: int = HIRES_HEIGHT

cfg = CardConfig()

# ─── Bottom Corner Cropping ───────────────────────────────────────────────────
def extract_corners(card_img: np.ndarray, cfg: CardConfig = cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies deformation and extracts both bottom-left and bottom-right grayscale crops.

    Args:
        card_img (np.ndarray): full card image in BGR format

    Returns:
        tuple[np.ndarray, np.ndarray]: left_crop, right_crop (normalized & batched)
    """
    aligned = deform_card(card_img)
    resized = cv2.resize(aligned, (cfg.init_w, cfg.init_h))
    h, w = cfg.init_h, cfg.init_w

    bottom_left  = resized[h - cfg.crop_h : h, 0 : cfg.crop_w]
    bottom_right = resized[h - cfg.crop_h : h, w - cfg.crop_w : w]

    gray_left  = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)

    # Normalize and reshape for model prediction: (1, H, W, 1)
    gray_left  = (gray_left.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
    gray_right = (gray_right.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]

    return gray_left, gray_right

# ─── High-Res Region Cropping for OCR ─────────────────────────────────────────
def extract_ocr_corner(card_img: np.ndarray, set_id: str, cfg: CardConfig = cfg) -> np.ndarray:
    """
    Based on set_id orientation, returns high-res bottom corner crop (for OCR).

    Args:
        card_img (np.ndarray): full image in BGR format
        set_id (str): predicted or known set ID (used to pick side)

    Returns:
        np.ndarray: cropped region (BGR)
    """
    aligned = deform_card(card_img)
    side = SETINFO[SETINFO[:, 0] == set_id][0, 3]  # 'left' or 'right'
    resized = cv2.resize(aligned, (cfg.hires_w, cfg.hires_h))
    h, w = cfg.hires_h, cfg.hires_w

    if side == 'left':
        return resized[h - cfg.crop_h*cfg.ratio : h, 0 : cfg.crop_w*cfg.ratio]
    else:
        return resized[h - cfg.crop_h*cfg.ratio : h, w - cfg.crop_w*cfg.ratio : w]

# ─── Utility ──────────────────────────────────────────────────────────────────
def load_model(path: str) -> tf.keras.Model:
    """
    Loads a TensorFlow Keras model from file.

    Args:
        path (str): Path to .h5 model file

    Returns:
        tf.keras.Model: compiled model
    """
    logger.info(f"Loading model from {path}")
    return tf.keras.models.load_model(path)

def predict_set_id(model: tf.keras.Model, corner: np.ndarray, label_encoder) -> str:
    """
    Predicts the set ID using the given model and a corner crop.

    Args:
        model (tf.keras.Model): Trained model
        corner (np.ndarray): Preprocessed image (1, H, W, 1)
        label_encoder (LabelEncoder): Used to decode class index

    Returns:
        str: predicted set ID
    """
    probs = model.predict(corner)
    class_idx = np.argmax(probs, axis=1)[0]
    return label_encoder.inverse_transform([class_idx])[0]
