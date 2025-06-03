import logging
from dataclasses import dataclass
import cv2
import numpy as np
import requests
import requests_cache
from tenacity import retry, stop_after_attempt, wait_exponential

from Pokemon_Core.Image_Module import(
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    HARD_CODED_WIDTH,
    HARD_CODED_HEIGHT,
    RATIO,
    HIRES_WIDTH,
    HIRES_HEIGHT,
    SETINFO)

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ─── HTTP Caching ─────────────────────────────────────────────────────────────
# Caches GET requests in SQLite for 24 h to avoid rate-limits and speed up retries
requests_cache.install_cache('pokedex_cache', backend='sqlite', expire_after=86400)

# ─── Configuration ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CardConfig:
    init_w: int     = INITIAL_WIDTH
    init_h: int     = INITIAL_HEIGHT
    crop_w: int     = HARD_CODED_WIDTH
    crop_h: int     = HARD_CODED_HEIGHT
    ratio:  int     = RATIO
    hires_w: int    = HIRES_WIDTH
    hires_h: int    = HIRES_HEIGHT

cfg = CardConfig()

# ─── Prediction Preprocessing ─────────────────────────────────────────────────
def card_prediction_processing(
    card: np.ndarray,
    cfg
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize to (init_w, init_h), crop bottom corners, convert to grayscale,
    normalize to [0,1], and add batch+channel dims → shapes (1, H, W, 1).
    Returns (left_corner, right_corner).
    """

    # Input check
    if not isinstance(card, np.ndarray):
        raise ValueError("Input image must be a NumPy array. Use np.array(img) if you loaded with PIL.")

    # 1) Resize to standard size
    card_img = cv2.resize(card, (cfg.init_w, cfg.init_h))

    # 2) Slice bottom-left & bottom-right corners
    h, w = cfg.init_h, cfg.init_w
    bl = card_img[h - cfg.crop_h : h, 0 : cfg.crop_w]
    br = card_img[h - cfg.crop_h : h, w - cfg.crop_w : w]

    # 3) Convert to grayscale
    gl = cv2.cvtColor(bl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)

    # 4) Normalize and reshape to (1, H, W, 1)
    gl = (gl.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
    gr = (gr.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]

    # Output shape check
    if gl.shape != (1, cfg.crop_h, cfg.crop_w, 1) or gr.shape != (1, cfg.crop_h, cfg.crop_w, 1):
        print(f"Warning: Output shape mismatch! Left: {gl.shape}, Right: {gr.shape}")
        return None

    return gl, gr


# ─── OCR Cropping ──────────────────────────────────────────────────────────────
def card_ocr_crop(
    card: np.ndarray,
    set_id: str,
    cfg: CardConfig = cfg
) -> np.ndarray:
    """
    Resize to high-res (hires_w, hires_h), then crop the one bottom corner
    (scaled by ratio) based on whether the icon lives on 'left' or 'right'.
    """
    side = SETINFO[SETINFO[:,0] == set_id][0, 3]  # 'left' or 'right'
    card_img = cv2.resize(card, (cfg.hires_w, cfg.hires_h))
    h, w = cfg.hires_h, cfg.hires_w
    if side == 'left':
        return card_img[h - cfg.crop_h*cfg.ratio : h, 0 : cfg.crop_w*cfg.ratio]
    else:
        return card_img[h - cfg.crop_h*cfg.ratio : h, w - cfg.crop_w*cfg.ratio : w]

# ─── API Fetching with Retries & Caching ───────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def get_card_info(set_id: str, poke_id: str) -> tuple[str | None, float | None, str | None]:
    """
    Fetch card metadata from Pokemon TCG API with retries and caching:
      - rarity (e.g. 'Rare Holo')
      - averageSellPrice
      - high-res image URL
    """
    url = f'https://api.pokemontcg.io/v2/cards/{set_id}-{poke_id}'
    logger.info(f"Fetching card info from {url}")
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json().get('data', {})
    rarity       = data.get('rarity')
    market_price = data.get('cardmarket', {}).get('prices', {}).get('averageSellPrice')
    image_url    = data.get('images', {}).get('large')
    return rarity, market_price, image_url
