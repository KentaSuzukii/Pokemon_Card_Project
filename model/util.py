import pandas as pd
import numpy as np
import requests
import cv2

from model import (
    SETINFO,
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    HARD_CODED_WIDTH,
    HARD_CODED_HEIGHT,
    REDUCED_SET,
)

def download_image(session: requests.Session, s_id: str, card_index: int) -> np.ndarray | None:
    """Fetches a high-res PNG from the Pokémon TCG API and returns it as a BGR OpenCV image."""
    url = f'https://images.pokemontcg.io/{s_id}/{card_index}_hires.png'
    try:
        resp = session.get(url, timeout=5)
        resp.raise_for_status()
        # Decode PNG bytes into BGR image
        img_array = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except requests.RequestException:
        print(f"⚠️ Failed to fetch {url}")
        return None

def crop_corners(img: np.ndarray, corner_w: int, corner_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the bottom-left and bottom-right grayscale crops of size (corner_h×corner_w)."""
    h_img, w_img = img.shape[:2]
    bl = img[h_img - corner_h : h_img, 0 : corner_w]
    br = img[h_img - corner_h : h_img, w_img - corner_w : w_img]
    gray_bl = cv2.cvtColor(bl, cv2.COLOR_BGR2GRAY)
    gray_br = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
    return gray_bl, gray_br

def make_record(
    corner_img: np.ndarray, position: str, set_id: str, set_name: str
) -> dict:
    """Packages one corner crop into a record suitable for a DataFrame row."""
    return {
        'corner':   corner_img,
        'position': position,
        'set_id':   set_id,
        'set_name': set_name,
    }

def create_dataset() -> pd.DataFrame:
    """
    Downloads every card in every set, crops its two bottom corners,
    labels them, and returns a DataFrame with columns:
      - corner (grayscale image array)
      - position ('left' or 'right')
      - set_id (or 'no')
      - set_name (or 'no')
    """
    session = requests.Session()
    records: list[dict] = []

    for s_id, count, set_name, side, _ in SETINFO:
        total = int(count)
        print(f"▶️ Processing set {s_id} ({total} cards)")
        for i in range(1, total + 1):
            img = download_image(session, s_id, i)
            if img is None:
                continue

            resized = cv2.resize(img, (INITIAL_WIDTH, INITIAL_HEIGHT))
            gray_left, gray_right = crop_corners(resized, HARD_CODED_WIDTH, HARD_CODED_HEIGHT)

            if side == 'left':
                records.append(make_record(gray_left,  'left',  s_id,      set_name))
                records.append(make_record(gray_right, 'right', 'no',      'no'))
            else:
                records.append(make_record(gray_left,  'left',  'no',      'no'))
                records.append(make_record(gray_right, 'right', s_id,      set_name))

    return pd.DataFrame.from_records(records)

def reduce_side(df: pd.DataFrame, side: str, reduced_set: int = REDUCED_SET) -> pd.DataFrame:
    """
    Keeps at most `reduced_set` samples per set_id for the given side.
    Returns a balanced DataFrame.
    """
    df_side = df[df['position'] == side]
    grouped = df_side.groupby('set_id', group_keys=False)
    sampled = grouped.apply(
        lambda g: g if len(g) <= reduced_set else g.sample(n=reduced_set, random_state=42)
    )
    return sampled.reset_index(drop=True)

def reduce_dataset(json_path: str) -> None:
    """
    Loads the full dataset JSON, balances left/right sides separately,
    and writes two reduced JSON files.
    """
    df_full = pd.read_json(json_path)

    df_left  = reduce_side(df_full, 'left')
    df_left.to_json('../../raw_data/dict_reduceddataset_left.json')

    df_right = reduce_side(df_full, 'right')
    df_right.to_json('../../raw_data/dict_reduceddataset_right.json')
