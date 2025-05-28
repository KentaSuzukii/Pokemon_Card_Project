import re
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import easyocr

from model import SETINFO, INITIAL_HEIGHT, INITIAL_WIDTH

# ─── EasyOCR Reader ────────────────────────────────────────────────────────────
# reuse one Reader instance for speed
_EASY_READER = easyocr.Reader(["en"], gpu=False, verbose=False)

# ─── Original Helpers ──────────────────────────────────────────────────────────
def is_groove(img: Image.Image) -> bool:
    orig_arr   = np.array(img, dtype="float32") / 255
    eroded_arr = np.array(
        img.filter(ImageFilter.GaussianBlur(3))
           .filter(ImageFilter.MaxFilter(15)),
        dtype="float32"
    ) / 255
    return (eroded_arr.var() - orig_arr.var()) < 0

def preproc_clean(data: list) -> np.ndarray:
    _ = np.array(data)
    return np.expand_dims(_, axis=2).astype("uint8")

def get_id_coords(set_id: str) -> tuple[int,int,int,int]:
    id_coord = None
    # (same hard-coded mapping as you already have)…
    if set_id in ("sv3","sv4","sv3pt5","sv2"):
        id_coord = (285,75,550,125)
    elif set_id in ("swsh9","swsh6","swsh12pt5","swsh10","swsh45"):
        id_coord = (280,75,540,125)
    elif set_id == "sm4":
        id_coord = (190,70,359,90)
    elif set_id in ("dv1","g1"):
        id_coord = (210,85,380,110)
    elif set_id == "xy1":
        id_coord = (130,80,330,100)
    elif set_id in ("xy2","xy3"):
        id_coord = (150,90,335,110)
    elif set_id in ("xy4","xy6","xy7"):
        id_coord = (150,90,335,115)
    elif set_id in ("dp1","dp2"):
        id_coord = (210,100,400,150)
    if id_coord is None:
        raise ValueError(f"Invalid set_id: {set_id}")
    return id_coord

def add_contrast(img: Image.Image, low: float = 0.1, high: float = 0.95) -> Image.Image:
    np_num = np.array(img, dtype=np.float32) / 255
    np_btm = np.quantile(np_num, low)
    np_num = np.clip(np_num - np_btm, 0, 1)
    np_top = np.quantile(np_num, high)
    np_num = np.clip(np_num / np_top, 0, 1) * 255
    return Image.fromarray(np_num.astype(np.uint8))

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from model.prediction import card_ocr_crop
from model import INITIAL_WIDTH, INITIAL_HEIGHT

def ocr_preprocessor(img: np.ndarray, set_id: str) -> Image.Image:
    """
    1) Extract the bottom-corner in high-res with card_ocr_crop().
    2) Convert to grayscale PIL.Image, boost contrast & invert if needed.
    """
    # 1a) If img is the raw PIL or NumPy hires card, turn to BGR array
    if isinstance(img, Image.Image):
        card_np = np.array(img)[:,:,::-1]  # PIL->RGB->BGR
    else:
        card_np = img

    # 1b) Get the perfect bottom-corner at high resolution
    corner_bgr = card_ocr_crop(card_np, set_id)  # BGR uint8

    # 2) Prep for OCR: gray → PIL
    gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)

    # 3) Contrast & invert if needed
    pil = ImageEnhance.Contrast(pil).enhance(1.0)
    if is_groove(pil):
        pil = ImageOps.invert(pil)
        pil = add_contrast(pil, 0.2, 0.97)
        pil = ImageOps.invert(pil)
    else:
        pil = add_contrast(pil, 0.2, 0.97)
        pil = ImageOps.invert(pil)

    return pil




# reuse one EasyOCR reader
_EASY_READER = easyocr.Reader(["en"], gpu=False, verbose=False)

# helper from your debug cell
def pick_strict_slash(texts: list[str]) -> str | None:
    """
    Return the first text matching 1–3 digits / >=1 digit, else None.
    """
    pat = re.compile(r"^\d{1,3}/\d+$")
    for t in texts:
        if pat.match(t):
            return t
    return None

def get_pokeid(img: np.ndarray, set_id: str) -> str:
    """
    1) Preprocess via ocr_preprocessor → PIL crop.
    2) Run EasyOCR.
    3) Pick first strict slash‐pattern match (1–3 digits before /, >=1 after).
    4) Fallback to longest digit‐run.
    5) Strip leading zeros.
    """
    # 1) get your crop
    pil_crop = ocr_preprocessor(img, set_id)
    arr = np.array(pil_crop)
    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # 2) OCR
    results = _EASY_READER.readtext(
        bgr,
        allowlist="0123456789/",
        batch_size=1
    )
    texts = [t for _, t, _ in results]

    # 3) strict slash first
    slash = pick_strict_slash(texts)
    if slash:
        num = slash.split("/", 1)[0]
    else:
        # 4) fallback → longest digit run in all OCR texts
        runs = re.findall(r"\d+", "".join(texts))
        num = max(runs, key=len) if runs else ""

    # 5) drop leading zeros, but not all
    return num.lstrip("0") or "0"
