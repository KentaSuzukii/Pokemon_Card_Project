# 📁 Pokemon_Core/Image/ocr_text_detection.py

import cv2
import numpy as np
import easyocr
import re
import logging
from typing import Optional
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from Pokemon_Core.config import SETINFO
from Pokemon_Core.CardProcessing.set_predictor import extract_ocr_corner
from Pokemon_Core.CardProcessing.card_aligner import deform_card

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader globally (English only)
reader = easyocr.Reader(['en'], gpu=False)

# ─── Regex Pattern Setup ──────────────────────────────────────────────────────
# Build matchers based on set-specific total card counts
_TOTAL = {
    sid: str(int(SETINFO[SETINFO[:, 0] == sid, 4][0]))
    for sid in SETINFO[:, 0]
}
_PAT_STRICT = {
    sid: re.compile(rf"^(\d{{1,3}})/{re.escape(total)}$")
    for sid, total in _TOTAL.items()
}
_PAT_ANYSLASH = re.compile(r"^(\d{1,3})/(\d+)$")

# ─────────────────────────────────────────────────────────────────────────────
# 📌 OCR Card Number Extraction (Fully Upgraded Version)
# Includes deform_card, regex tiers, white-on-dark detection, and blob fallback
# ─────────────────────────────────────────────────────────────────────────────

def extract_card_number(card_img: np.ndarray, set_id: str) -> Optional[str]:
    """
    Extract the card number from a given Pokémon card image.

    Steps:
    1. Deform the card image to a top-down view.
    2. Crop the bottom-right OCR region.
    3. Enhance contrast and detect inversion.
    4. Use EasyOCR to extract text.
    5. Use multiple regex patterns and heuristics to extract the number.

    Args:
        card_img (np.ndarray): Full image of the card (BGR)
        set_id (str): Set ID string (used for regex matching)

    Returns:
        str: Detected card number or "0" if not found
    """
    aligned_img = deform_card(card_img)
    crop = extract_ocr_corner(aligned_img, set_id)

    # Convert to grayscale using PIL for white-on-dark detection
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    pil = ImageEnhance.Contrast(pil).enhance(1.5)

    # Detect and invert white-on-dark text
    arr = np.array(pil, dtype=np.float32) / 255.0
    bld = pil.filter(ImageFilter.GaussianBlur(3))
    groove = (np.array(bld, dtype=np.float32) / 255.0).var() - arr.var() < 0
    if groove:
        pil = ImageOps.invert(pil)

    # Resize and binarize
    gray_img = np.array(pil, dtype=np.uint8)
    upscaled = cv2.resize(gray_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # OCR pass 1
    ocr_res = reader.readtext(bgr, allowlist="0123456789/", batch_size=1)
    candidates = [t.strip() for _, t, conf in ocr_res if conf > 0.2 and t.strip()]

    # Regex Tier 1: strict match like "101/198"
    pat_total = _PAT_STRICT.get(set_id, _PAT_ANYSLASH)
    for t in candidates:
        m = pat_total.match(t)
        if m:
            return m.group(1).lstrip("0") or "0"

    # Regex Tier 2: general slash match
    for t in candidates:
        m = _PAT_ANYSLASH.match(t)
        if m:
            return m.group(1).lstrip("0") or "0"

    # Regex Tier 3: digit sequence fallback
    runs = re.findall(r"\d+", "".join(candidates))
    if runs:
        best = max(runs, key=len)
        return (best[-3:] if len(best) > 3 else best).lstrip("0") or "0"

    # Blob fallback
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape
    blobs = [
        (x, binary[y:y+h, x:x+w])
        for c in cnts
        for x, y, w, h in [cv2.boundingRect(c)]
        if 0.2*H < h < 0.9*H and 0.05*W < w < 0.5*W
    ]
    blobs.sort(key=lambda b: b[0])  # sort left to right

    text = ""
    for _, blob in blobs:
        sub = cv2.resize(blob, (28, 28), interpolation=cv2.INTER_CUBIC)
        sub_bgr = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        r = reader.readtext(sub_bgr, allowlist="0123456789/", batch_size=1)
        if r and r[0][1].strip():
            text += r[0][1].strip()

    # Final regex attempt from blob result
    m = _PAT_ANYSLASH.match(text)
    if m:
        return m.group(1).lstrip("0") or "0"

    runs = re.findall(r"\d+", text)
    if runs:
        best = max(runs, key=len)
        return (best[-3:] if len(best) > 3 else best).lstrip("0") or "0"

    logger.warning("❌ All OCR strategies failed")
    return "0"
