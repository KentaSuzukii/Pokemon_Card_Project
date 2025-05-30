import os
import numpy as np
import pandas as pd
import requests
import cv2
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from model import (
    SETINFO,
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    HARD_CODED_WIDTH,
    HARD_CODED_HEIGHT,
    REDUCED_SET,
)

# ---------------------- Core Download/Crop Functions ----------------------

def download_image(session: requests.Session, s_id: str, card_index: int) -> np.ndarray | None:
    """Fetches a high-res PNG from the Pokémon TCG API and returns it as a BGR OpenCV image."""
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
    """Returns the bottom-left and bottom-right grayscale crops of size (corner_h×corner_w)."""
    h_img, w_img = img.shape[:2]
    bl = img[h_img - corner_h : h_img, 0 : corner_w]
    br = img[h_img - corner_h : h_img, w_img - corner_w : w_img]
    gray_bl = cv2.cvtColor(bl, cv2.COLOR_BGR2GRAY)
    gray_br = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
    return gray_bl, gray_br

def make_record(corner_img: np.ndarray, position: str, set_id: str, set_name: str) -> dict:
    """Packages one corner crop into a record suitable for a DataFrame row."""
    return {
        'corner':   corner_img,
        'position': position,
        'set_id':   set_id,
        'set_name': set_name,
    }

# ---------------------- Dataset Builder with Progress and Robustness ----------------------

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
        for i in tqdm(range(1, total + 1), desc=f"Set {s_id}"):
            img = download_image(session, s_id, i)
            if img is None:
                continue
            try:
                resized = cv2.resize(img, (INITIAL_WIDTH, INITIAL_HEIGHT))
                gray_left, gray_right = crop_corners(resized, HARD_CODED_WIDTH, HARD_CODED_HEIGHT)
            except cv2.error:
                print(f"⚠️ OpenCV error on set {s_id} card {i}")
                continue

            if side == 'left':
                records.append(make_record(gray_left,  'left',  s_id,      set_name))
                records.append(make_record(gray_right, 'right', 'no',      'no'))
            else:
                records.append(make_record(gray_left,  'left',  'no',      'no'))
                records.append(make_record(gray_right, 'right', s_id,      set_name))

    return pd.DataFrame.from_records(records)

# ---------------------- Memory-Efficient Disk Storage (Optional) ----------------------

def save_crops_to_disk(df, out_dir="crops"):
    os.makedirs(out_dir, exist_ok=True)
    new_records = []
    for idx, row in df.iterrows():
        fname = f"{row['position']}_{row['set_id']}_{idx}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, row['corner'])
        new_records.append({
            "img_path": path,
            "position": row["position"],
            "set_id": row["set_id"],
            "set_name": row["set_name"]
        })
    return pd.DataFrame(new_records)

# ---------------------- Class Balancing ----------------------

def reduce_side(df: pd.DataFrame, side: str, reduced_set: int = REDUCED_SET, verbose=True) -> pd.DataFrame:
    """
    Keeps at most `reduced_set` samples per set_id for the given side.
    Returns a balanced DataFrame.
    """
    df_side = df[df['position'] == side]
    grouped = df_side.groupby('set_id', group_keys=False)
    sampled = grouped.apply(
        lambda g: g if len(g) <= reduced_set else g.sample(n=reduced_set, random_state=42)
    )
    sampled = sampled.reset_index(drop=True)
    if verbose:
        print(f"Sample count per class for '{side}':\n{sampled['set_id'].value_counts()}")
    return sampled

# ---------------------- Model Ready Preprocessing ----------------------

def prepare_X_y(df, img_key='corner'):
    X = np.stack([np.expand_dims(img, -1) if img.ndim == 2 else img for img in df[img_key].values])
    X = X.astype('float32') / 255.
    y = df['set_id'].values
    return X, y

# ---------------------- Class Weights (for Imbalance) ----------------------

def compute_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict

# ---------------------- Example Usage (Commented for Notebooks/Scripts) ----------------------

if __name__ == "__main__":
    # 1. Build the full dataset
    df_full = create_dataset()
    print("Total records:", len(df_full))

    # 2. Optionally balance per side and/or save crops to disk
    df_left  = reduce_side(df_full, 'left')
    df_right = reduce_side(df_full, 'right')

    # 3. If needed, save balanced crops
    # df_left_disk = save_crops_to_disk(df_left, out_dir="crops_left")
    # df_right_disk = save_crops_to_disk(df_right, out_dir="crops_right")

    # 4. Prepare data for ML
    X_left, y_left = prepare_X_y(df_left)
    X_right, y_right = prepare_X_y(df_right)

    # 5. Compute class weights
    class_weight_left = compute_class_weights(y_left)
    print("Class weights (left):", class_weight_left)

    # Now you're ready to train a model!
