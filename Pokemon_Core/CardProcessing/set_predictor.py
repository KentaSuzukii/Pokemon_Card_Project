# 📁 Pokemon_Core/CardProcessing/set_predictor.py

import os
import cv2
import pickle
import logging
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

from Pokemon_Core.config import (
    INITIAL_WIDTH, INITIAL_HEIGHT,
    HARD_CODED_WIDTH, HARD_CODED_HEIGHT,
    HIRES_WIDTH, HIRES_HEIGHT,
    RATIO, SETINFO
)
from Pokemon_Core.CardProcessing.card_aligner import deform_card

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Set Predictor Module
# Given a card image, this module aligns the card, extracts corners,
# runs the trained classifier, and prepares high-resolution crops for OCR.
# ─────────────────────────────────────────────────────────────────────────────

# 🛠️ Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 🖼️ Dimension Configuration Dataclass
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

# ─────────────────────────────────────────────────────────────────────────────
# 📥 Load Trained Assets (Model + Label Encoder)
# Loads both the Keras model and label encoder from the Models/ directory.
# Handles missing encoder file gracefully.
# ─────────────────────────────────────────────────────────────────────────────

def load_prediction_assets(model_path="Models/best_symbols_model.h5", encoder_path="Models/label_encoder.pkl"):
    """
    Loads the trained CNN model and label encoder for set prediction.

    Args:
        model_path (str): Path to .h5 Keras model file
        encoder_path (str): Path to .pkl LabelEncoder

    Returns:
        model (tf.keras.Model), label_encoder (LabelEncoder)
    """
    # 🧠 Load model
    logger.info(f"🔍 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 🧠 Load label encoder
    logger.info(f"🔍 Loading label encoder from: {encoder_path}")
    if not os.path.exists(encoder_path):
        logger.warning("⚠️ Label encoder file not found. Returning empty LabelEncoder.")
        return model, LabelEncoder()

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

# ─────────────────────────────────────────────────────────────────────────────
# 🔻 Corner Extraction for Classification
# Aligns the card and extracts the bottom-left and bottom-right grayscale corners.
# ─────────────────────────────────────────────────────────────────────────────

def extract_corners(card_img: np.ndarray, cfg: CardConfig = cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies alignment + crops left/right corners, then normalizes them.

    Returns:
        gray_left (np.ndarray): (1, H, W, 1)
        gray_right (np.ndarray): (1, H, W, 1)
    """
    aligned = deform_card(card_img)
    resized = cv2.resize(aligned, (cfg.init_w, cfg.init_h))
    h, w = cfg.init_h, cfg.init_w

    bottom_left  = resized[h - cfg.crop_h : h, 0 : cfg.crop_w]
    bottom_right = resized[h - cfg.crop_h : h, w - cfg.crop_w : w]

    gray_left  = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)

    gray_left  = (gray_left.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
    gray_right = (gray_right.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]

    return gray_left, gray_right

# ─────────────────────────────────────────────────────────────────────────────
# 🔮 Set Prediction from Single Corner
# Runs the model to get a predicted class and returns its label.
# ─────────────────────────────────────────────────────────────────────────────

def predict_set_id(model: tf.keras.Model, corner: np.ndarray, label_encoder: LabelEncoder) -> str:
    """
    Runs prediction and returns decoded set ID string.
    """
    probs = model.predict(corner, verbose=0)
    class_idx = np.argmax(probs, axis=1)[0]
    return label_encoder.inverse_transform([class_idx])[0]

# ─────────────────────────────────────────────────────────────────────────────
# 🔍 OCR Crop Extractor
# Uses set-specific info to extract the high-resolution OCR region.
# ─────────────────────────────────────────────────────────────────────────────

def extract_ocr_corner(card_img: np.ndarray, set_id: str, cfg: CardConfig = cfg) -> np.ndarray:
    """
    Crops high-resolution bottom corner region for OCR, based on set's expected corner side.
    """
    aligned = deform_card(card_img)
    resized = cv2.resize(aligned, (cfg.hires_w, cfg.hires_h))
    h, w = cfg.hires_h, cfg.hires_w

    side = SETINFO[SETINFO[:, 0] == set_id][0, 3]  # e.g., "left" or "right"

    if side == "left":
        return resized[h - cfg.crop_h * cfg.ratio : h, 0 : cfg.crop_w * cfg.ratio]
    else:
        return resized[h - cfg.crop_h * cfg.ratio : h, w - cfg.crop_w * cfg.ratio : w]
