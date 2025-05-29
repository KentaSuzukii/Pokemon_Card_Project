import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import easyocr, pytesseract

from model import SETINFO
from model.prediction import card_ocr_crop

# ─── EasyOCR reader (reuse one) ────────────────────────────────────────────────
_OCR = easyocr.Reader(["en"], gpu=False, verbose=False)

# ─── Precompute each set’s “total” and regex for strict slash → total ─────────
_TOTAL = {
    sid: str(int(SETINFO[SETINFO[:,0]==sid, 4][0]))
    for sid in SETINFO[:,0]
}
_PAT_STRICT = {
    sid: re.compile(rf"^(\d{{1,3}})/{re.escape(total)}$")
    for sid, total in _TOTAL.items()
}
# loose “any slash” pattern (1–3 digits before slash, any digits after)
_PAT_ANYSLASH = re.compile(r"^(\d{1,3})/(\d+)$")


def _preprocess_corner(img: np.ndarray, set_id: str) -> np.ndarray:
    """
    1) Grab the high-res bottom corner via card_ocr_crop
    2) Gray → PIL → boost contrast → invert if white-on-dark
    Returns a uint8 2D array ready for upscaling/threshold.
    """
    corner = card_ocr_crop(img, set_id)                    # BGR hires crop
    gray   = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)      # → uint8 gray
    pil    = Image.fromarray(gray)
    pil    = ImageEnhance.Contrast(pil).enhance(1.5)

    # determine “groove” (white text on dark bg) by blur-variance trick
    arr = np.array(pil, dtype=np.float32) / 255.0
    bld = pil.filter(ImageFilter.GaussianBlur(3))
    groove = (np.array(bld, dtype=np.float32)/255.0).var() - arr.var() < 0
    if groove:
        pil = ImageOps.invert(pil)

    return np.array(pil, dtype=np.uint8)


def get_pokeid(img: np.ndarray, set_id: str) -> str:
    """
    1) Preprocess corner → upscale×2 + Otsu threshold
    2) EasyOCR once → candidates
    3a) Try strict “123/TOTAL” → return left side
    3b) Try any “123/xyz” → return left side
    3c) Fallback→longest digit run capped to 3
    4) If no candidate at all, do blob segmentation + per-blob EasyOCR, then retry slash + run
    """
    # ── step 1: crop, preprocess, upscale & binarize ─────────────────────────────
    gray2 = _preprocess_corner(img, set_id)
    up2   = cv2.resize(gray2, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(up2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bgr2  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # ── step 2: EasyOCR on the full patch ────────────────────────────────────────
    ocr_res    = _OCR.readtext(bgr2, allowlist="0123456789/", batch_size=1)
    candidates = [t.strip() for _, t, conf in ocr_res if conf > 0.2 and t.strip()]

    # ── step 3a: strict “123/TOTAL” match ────────────────────────────────────────
    pat_total = _PAT_STRICT[set_id]
    for t in candidates:
        m = pat_total.match(t)
        if m:
            return m.group(1).lstrip("0") or "0"

    # ── step 3b: any “123/xyz” match ─────────────────────────────────────────────
    for t in candidates:
        m = _PAT_ANYSLASH.match(t)
        if m:
            return m.group(1).lstrip("0") or "0"

    # ── step 3c: longest digit-run (up to 3 digits) ──────────────────────────────
    runs = re.findall(r"\d+", "".join(candidates))
    if runs:
        best = max(runs, key=len)
        best = best[-3:] if len(best) > 3 else best
        return best.lstrip("0") or "0"

    # ── step 4: blob fallback (if full OCR found nothing) ────────────────────────
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W    = bw.shape
    blobs   = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # only reasonably tall & narrow blobs
        if 0.2*H < h < 0.9*H and 0.05*W < w < 0.5*W:
            blobs.append((x, bw[y:y+h, x:x+w]))
    blobs.sort(key=lambda b: b[0])

    text = ""
    for _, blob in blobs:
        sub   = cv2.resize(blob, (28,28), interpolation=cv2.INTER_CUBIC)
        sub_b = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        r     = _OCR.readtext(sub_b, allowlist="0123456789/", batch_size=1)
        if r and r[0][1].strip():
            text += r[0][1].strip()

    # retry slash on blob-OCR result
    m = _PAT_ANYSLASH.match(text)
    if m:
        return m.group(1).lstrip("0") or "0"

    # final digit-run fallback on blob text
    runs = re.findall(r"\d+", text)
    if runs:
        best = max(runs, key=len)
        best = best[-3:] if len(best) > 3 else best
        return best.lstrip("0") or "0"

    # nothing found
    return "0"
