# 📁 Pokemon_Core/Image/util.py

import os
import cv2
import requests
import pandas as pd
import numpy as np

from Pokemon_Core.config import (
    SETINFO,
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    HARD_CODED_WIDTH,
    HARD_CODED_HEIGHT,
    REDUCED_SET
)

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Utility Module for Image Dataset Creation
# Includes functions to download card images, crop corners, label them,
# and reduce the dataset for training.
# ─────────────────────────────────────────────────────────────────────────────

def download_image(session: requests.Session, s_id: str, card_index: int) -> np.ndarray | None:
    """
    Downloads high-resolution card image from Pokémon TCG API.

    Args:
        session (requests.Session): reusable session
        s_id (str): set ID (e.g. 'sv3pt5')
        card_index (int): card number in set

    Returns:
        np.ndarray | None: BGR OpenCV image, or None if failed
    """
    url = f'https://images.pokemontcg.io/{s_id}/{card_index}_hires.png'
    try:
        resp = session.get(url, timeout=5)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except requests.RequestException:
        print(f"⚠️ Failed to fetch {url}")
        return None

def crop_corners(img: np.ndarray, corner_w: int, corner_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the bottom-left and bottom-right corners from a resized image.

    Args:
        img (np.ndarray): full image (H, W, 3)
        corner_w (int): corner crop width
        corner_h (int): corner crop height

    Returns:
        tuple[np.ndarray, np.ndarray]: grayscale bottom-left and right corners
    """
    h_img, w_img = img.shape[:2]
    bl = img[h_img - corner_h : h_img, 0 : corner_w]
    br = img[h_img - corner_h : h_img, w_img - corner_w : w_img]
    return cv2.cvtColor(bl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)

def make_record(corner_img: np.ndarray, position: str, set_id: str, set_name: str) -> dict:
    """
    Formats a single corner crop into a labeled training dictionary.
    """
    return {
        'corner':   corner_img,
        'position': position,
        'set_id':   set_id,
        'set_name': set_name,
    }

def create_dataset() -> pd.DataFrame:
    """
    Main loop to construct the full corner-based dataset.
    Downloads each card, resizes it, crops both corners, and labels them.

    Returns:
        pd.DataFrame: dataset with columns: corner, position, set_id, set_name
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
    Limits each set_id to a maximum number of samples (per side).
    Helps balance the dataset for training.
    """
    df_side = df[df['position'] == side]
    grouped = df_side.groupby('set_id', group_keys=False)
    sampled = grouped.apply(
        lambda g: g if len(g) <= reduced_set else g.sample(n=reduced_set, random_state=42)
    )
    return sampled.reset_index(drop=True)

def reduce_dataset(json_path: str) -> None:
    """
    Reduces a full dataset to smaller left/right balanced versions and saves them.

    Args:
        json_path (str): path to full_dataset.json
    """
    df_full = pd.read_json(json_path)

    df_left  = reduce_side(df_full, 'left')
    df_left.to_json('Data/Processed/dict_reduceddataset_left.json')

    df_right = reduce_side(df_full, 'right')
    df_right.to_json('Data/Processed/dict_reduceddataset_right.json')
