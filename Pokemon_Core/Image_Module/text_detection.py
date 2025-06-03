import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract
import easyocr
import re

# --- Minimal SETINFO Example: expand as needed ---
# For reference-style cropping and totals.
SETINFO = {
    # set_id      (coords),                total,   side
    "dv1":        ((210, 85, 380, 110),     "21",   "right"),
    "swsh9":      ((280, 75, 540, 125),     "186",  "left"),
    "swsh45":     ((280, 75, 540, 125),     "73",   "left"),
    "swsh6":      ((280, 75, 540, 125),     "233",  "left"),
    "swsh12pt5":  ((280, 75, 540, 125),     "160",  "left"),
    "xy1":        ((130, 80, 330, 100),     "146",  "right"),
    "xy2":        ((150, 90, 335, 110),     "110",  "right"),
    "xy3":        ((150, 90, 335, 110),     "114",  "right"),
    "g1":         ((210, 85, 380, 110),     "117",  "right"),
    "xy4":        ((150, 90, 335, 115),     "124",  "right"),
    "xy6":        ((150, 90, 335, 115),     "112",  "right"),
    "xy7":        ((150, 90, 335, 115),     "100",  "right"),
    "dp1":        ((210, 100, 400, 150),    "130",  "right"),
    "dp2":        ((210, 100, 400, 150),    "124",  "right"),
    "sm4":        ((190, 70, 359, 90),      "126",  "left"),
    "swsh10":     ((280, 75, 540, 125),     "216",  "left"),
    "sv4":        ((285, 75, 550, 125),     "266",  "left"),
    "sv3pt5":     ((285, 75, 550, 125),     "207",  "left"),
    "sv3":        ((285, 75, 550, 125),     "230",  "left"),
    "sv2":        ((285, 75, 550, 125),     "279",  "left"),
}


# --- Reference: Preprocessing & Cropping ---
def is_groove(img: Image.Image) -> bool:
    orig_arr = np.array(img, dtype="float32") / 255
    eroded_arr = np.array(img.filter(ImageFilter.GaussianBlur(3)).filter(ImageFilter.MaxFilter(15)), dtype="float32") / 255
    orig_var = orig_arr.var()
    eroded_var = eroded_arr.var()
    return eroded_var - orig_var < 0

def add_contrast(img: Image, low: float = 0.1, high: float = 0.95) -> Image:
    np_num = np.array(img, dtype=np.float32) / 255
    np_btm = np.quantile(np_num, low)
    np_num -= np_btm
    np_num = np.clip(np_num, 0, 1)
    np_top = np.quantile(np_num, high)
    np_num = np.clip(np_num / np_top, 0, 1) * 255
    return Image.fromarray(np_num.astype(np.uint8))

def get_id_coords(set_id: str, img_shape=None, tight=False):
    """
    Returns the crop rectangle for the set_id.
    If img_shape is provided, will adapt if you resize, but assumes standard (600x825).
    If tight=True, still uses the original box (to avoid too-narrow crops).
    """
    if set_id not in SETINFO:
        raise ValueError(f"Invalid set_id: {set_id}")
    # Always use the original crop for robustness
    crop_w, crop_h = 200, 72
    img_w, img_h = 600, 825
    if img_shape is not None:
        img_h, img_w = img_shape[:2]
    _, _, side = SETINFO[set_id]
    if side == "left":
        a, b, c, d = 0, img_h - crop_h, crop_w, img_h
    else:
        a, b, c, d = img_w - crop_w, img_h - crop_h, img_w, img_h
    return a, b, c, d




def ocr_preprocessor(img_input: np.ndarray, set_id: str) -> Image:
    # Ensure input shape is (H, W, C)
    if img_input.ndim == 2:
        img_input = np.expand_dims(img_input, axis=2)
    if img_input.shape[2] == 1:
        img_input = np.repeat(img_input, 3, axis=2)
    gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray_img)
    a, b, c, d = get_id_coords(set_id, img_input.shape, tight=True)
    img_contrast = img.crop((a, b, c, d))
    contrast_enhancer = ImageEnhance.Contrast(img_contrast)
    img_contrast = contrast_enhancer.enhance(1)
    im_offset = np.clip(np.array(img_contrast), 0, 255).astype("uint8")
    im_offset = Image.fromarray(im_offset)
    if is_groove(im_offset):  # Black Text
        im_offset = ImageOps.invert(im_offset)
        im_offset = add_contrast(im_offset, 0.2, 0.97)
        im_offset = ImageOps.invert(im_offset)
    else:  # White Text
        im_offset = add_contrast(im_offset, 0.2, 0.97)
        im_offset = ImageOps.invert(im_offset)
    return im_offset


# --- Reference: OCR using pytesseract (fallback to pyocr if desired) ---
def ocr_text(img: np.ndarray) -> str:
    # img can be np.ndarray or PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    result = pytesseract.image_to_string(img, config="--psm 6 -c tessedit_char_whitelist=0123456789/")
    # Optionally, retry up to 10x with random brightness/contrast tweaks if result is blank
    if not result.strip():
        brightness_enhancer = ImageEnhance.Brightness(img)
        contrast_enhancer = ImageEnhance.Contrast(img)
        for _ in range(10):
            img_bright = brightness_enhancer.enhance(1 + np.random.uniform(-0.2, 0.2))
            result = pytesseract.image_to_string(img_bright, config="--psm 6 -c tessedit_char_whitelist=0123456789/")
            if result.strip():
                break
            img_contrast = contrast_enhancer.enhance(1 + np.random.uniform(-0.2, 0.2))
            result = pytesseract.image_to_string(img_contrast, config="--psm 6 -c tessedit_char_whitelist=0123456789/")
            if result.strip():
                break
    return result

def extract_number_before_slash_or_cardtotal(text: str, set_id: str) -> str:
    sequence = SETINFO[set_id][1] if set_id in SETINFO else ""
    cleaned_text = re.sub(r"[^0-9/]", "", text)
    if "/" in cleaned_text:
        match = re.search(r"(\d+)(?=/)", cleaned_text)
        if match:
            return match.group(1)
        else:
            return ""
    else:
        pattern = rf"(\d+)(?={sequence}(?!\d))"
        match = re.search(pattern, cleaned_text)
        if match:
            return match.group(1)[:-1]
        else:
            number_match = re.search(r"\d+", cleaned_text)
            if number_match:
                return number_match.group(0)
            else:
                return ""

def is_valid_pokeid(candidate: str, set_id: str) -> bool:
    """Return True if the candidate looks like a valid pokeid (e.g. 2-3 digits)."""
    # Require at least two digits (change this if your smallest sets really have single-digit pokeids)
    return bool(re.fullmatch(r"\d{2,3}", candidate))


def clean_pokeid(pokeid: str, set_id: str) -> str:
    if " " in pokeid:
        pokeid = pokeid.split(" ", 1)[1]
    pokeid = extract_number_before_slash_or_cardtotal(pokeid, set_id)
    pokeid = re.sub(r"[^A-Za-z0-9]", "", pokeid)
    # Add set-specific pokeid cleaning if needed, simplified here:
    if pokeid != "":
        if set_id in ("dv1",):
            if len(pokeid) > 2:
                pokeid = pokeid[:2]
            if len(pokeid) == 2 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]
        # Add more set-specific logic as needed!
    return pokeid

# --- Your EasyOCR fallback with blob fallback and regex ---
EASY_OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
def card_ocr_crop(card: np.ndarray, set_id: str) -> np.ndarray:
    # This version: just crops bottom left/right, adjust as needed for your images
    coords, _, side = SETINFO.get(set_id, ((0,0,0,0), "", "left"))
    h, w = card.shape[:2]
    if side == "left":
        return card[h-72:h, 0:200]
    else:
        return card[h-72:h, w-200:w]

def _preprocess_corner(img: np.ndarray, set_id: str) -> np.ndarray:
    corner = card_ocr_crop(img, set_id)
    gray   = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    pil    = Image.fromarray(gray)
    pil    = ImageEnhance.Contrast(pil).enhance(1.5)
    arr = np.array(pil, dtype=np.float32) / 255.0
    bld = pil.filter(ImageFilter.GaussianBlur(3))
    groove = (np.array(bld, dtype=np.float32)/255.0).var() - arr.var() < 0
    if groove:
        pil = ImageOps.invert(pil)
    return np.array(pil, dtype=np.uint8)

def get_pokeid_easyocr(img: np.ndarray, set_id: str) -> str:
    gray2 = _preprocess_corner(img, set_id)
    up2   = cv2.resize(gray2, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(up2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bgr2  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    ocr_res = EASY_OCR.readtext(bgr2, allowlist="0123456789/", batch_size=1)

    print("DEBUG: EasyOCR raw:", ocr_res)
    # ---- 1. Try all EasyOCR results for left-of-slash number (up to 3 digits) ----
    for _, text, conf in ocr_res:
        print("DEBUG: Candidate:", text, "Conf:", conf)
        if conf > 0.2:
            match = re.search(r"(\d{1,3})/\d+", text)
            if match:
                print("DEBUG: Matched regex:", match.group(1))
                return match.group(1).lstrip("0") or "0"

    # The rest remains the same...
    # ... (keep strict/legacy and blob fallback as they are)
    # Strict “123/TOTAL” match
    if set_id in SETINFO:
        total = SETINFO[set_id][1]
        pat_total = re.compile(rf"^(\d{{1,3}})/{re.escape(total)}$")
        for _, text, conf in ocr_res:
            if conf > 0.2:
                m = pat_total.match(text)
                if m:
                    print("DEBUG: Matched total:", m.group(1))
                    return m.group(1).lstrip("0") or "0"

    # Any “123/xyz” match
    pat_anyslash = re.compile(r"^(\d{1,3})/(\d+)$")
    for _, text, conf in ocr_res:
        if conf > 0.2:
            m = pat_anyslash.match(text)
            if m:
                print("DEBUG: Matched anyslash:", m.group(1))
                return m.group(1).lstrip("0") or "0"

    # Longest digit run
    candidates = [text.strip() for _, text, conf in ocr_res if conf > 0.2 and text.strip()]
    runs = re.findall(r"\d+", "".join(candidates))
    print("DEBUG: Fallback runs:", runs)
    if runs:
        best = max(runs, key=len)
        best = best[-3:] if len(best) > 3 else best
        print("DEBUG: Longest run fallback:", best)
        return best.lstrip("0") or "0"

    # Blob fallback (as before)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bw.shape
    blobs = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 0.2*H < h < 0.9*H and 0.05*W < w < 0.5*W:
            blobs.append((x, bw[y:y+h, x:x+w]))
    blobs.sort(key=lambda b: b[0])
    text = ""
    for _, blob in blobs:
        sub = cv2.resize(blob, (28, 28), interpolation=cv2.INTER_CUBIC)
        sub_b = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        r = EASY_OCR.readtext(sub_b, allowlist="0123456789/", batch_size=1)
        if r and r[0][1].strip():
            text += r[0][1].strip()
    m = pat_anyslash.match(text)
    if m:
        print("DEBUG: Matched anyslash in blob:", m.group(1))
        return m.group(1).lstrip("0") or "0"
    runs = re.findall(r"\d+", text)
    print("DEBUG: Fallback runs in blob:", runs)
    if runs:
        best = max(runs, key=len)
        best = best[-3:] if len(best) > 3 else best
        print("DEBUG: Longest run fallback in blob:", best)
        return best.lstrip("0") or "0"
    return "0"



# --- Unified main function: call this! ---
def get_pokeid(card_img: np.ndarray, set_id: str) -> str:
    """
    Extracts the Pokémon ID using reference pipeline, with EasyOCR fallback.
    """
    # 1. Reference pipeline: hard-coded crop, tesseract, set-specific cleaning
    try:
        crop = ocr_preprocessor(card_img, set_id)
        tesseract_result = ocr_text(np.array(crop))
        cleaned = clean_pokeid(tesseract_result, set_id)
        # Only accept if result is plausible (2–3 digits)
        if cleaned and cleaned not in ["", "0", "None"] and is_valid_pokeid(cleaned, set_id):
            print(f"[Reference OCR succeeded] Set={set_id}, ID={cleaned}")
            return cleaned
        else:
            print(f"[Reference OCR rejected] Set={set_id}, ID={cleaned}")
    except Exception as e:
        print(f"[Reference OCR pipeline error]: {e}")
    # 2. Fallback: your robust EasyOCR pipeline
    try:
        fallback_id = get_pokeid_easyocr(card_img, set_id)
        if fallback_id and fallback_id not in ["", "0", "None"]:
            print(f"[Fallback OCR succeeded] Set={set_id}, ID={fallback_id}")
            return fallback_id
    except Exception as e:
        print(f"[Fallback OCR pipeline error]: {e}")
    print(f"[Both OCR pipelines failed] Set={set_id}, returning '0'")
    return "0"


# --- Example usage ---
if __name__ == "__main__":
    # Load your image as a numpy array, e.g. via OpenCV
    img_path = "YOUR_IMAGE_PATH.jpg"
    set_id = "sv3"  # or whatever set you're using
    card_img = cv2.imread(img_path)
    pokeid = get_pokeid(card_img, set_id)
    print("Extracted Pokémon ID:", pokeid)
