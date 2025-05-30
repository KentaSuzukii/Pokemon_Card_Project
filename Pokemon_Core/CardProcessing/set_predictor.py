# 📁 Pokemon_Core/Image/set_predictor.py

import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import logging
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
# Given a user-uploaded image, this module:
# - Aligns the card via perspective correction
# - Extracts bottom corners for set prediction
# - Extracts high-res region for OCR (corner depends on predicted set)
# ─────────────────────────────────────────────────────────────────────────────

# ─── Logger Setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Card Image Dimension Configuration ──────────────────────────────────────
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

# ─── Load Trained Model and Label Encoder ─────────────────────────────────────
def load_prediction_assets(model_path="best_symbols_model.h5", encoder_path="label_encoder.pkl"):
    """
    Loads the trained CNN model and label encoder for set prediction.
    """
    logger.info(f"🔍 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    logger.info(f"🔍 Loading label encoder from: {encoder_path}")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

# ─── Bottom Corner Cropping for Prediction ────────────────────────────────────
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

# ─── Predict Set ID from a Single Corner ──────────────────────────────────────
def predict_set_id(model: tf.keras.Model, corner: np.ndarray, label_encoder: LabelEncoder) -> str:
    """
    Runs prediction and returns decoded set ID string.
    """
    probs = model.predict(corner, verbose=0)
    class_idx = np.argmax(probs, axis=1)[0]
    return label_encoder.inverse_transform([class_idx])[0]

# ─── High-Resolution Crop for OCR Use ─────────────────────────────────────────
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
