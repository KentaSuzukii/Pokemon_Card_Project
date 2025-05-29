import os
import pandas as pd
import numpy as np
import requests
import cv2
from PIL import Image
from io import BytesIO

from Pokemon_Core.config import (
    SETINFO, REDUCED_SET,
    INITIAL_WIDTH, INITIAL_HEIGHT,
    HARD_CODED_WIDTH, HARD_CODED_HEIGHT
)

def create_dataset(save_images: bool = True) -> pd.DataFrame:
    """Creates training dataset with bottom corners of Pokémon cards."""
    dataset_df = pd.DataFrame(columns=['corner', 'position', 'set_id', 'set_name'], index=[0])
    k = 0

    if save_images:
        os.makedirs("Data/Processed/Images", exist_ok=True)

    for j in range(SETINFO.shape[0]):
        s_id = SETINFO[j, 0]
        print(f'On-going set: {s_id}')

        for i in range(1, int(SETINFO[j, 1]) + 1):
            image_url = f'https://images.pokemontcg.io/{s_id}/{str(i)}_hires.png'
            response_card = requests.get(image_url)

            if response_card.status_code == 200:
                image = Image.open(BytesIO(response_card.content))

                # Optional: save full card image
                # save_dir = "Data/Raw/Full_Cards"
                # os.makedirs(save_dir, exist_ok=True)
                # image.save(f"{save_dir}/card_{s_id}_{str(i)}.png")

                card_image = np.array(image)
                new_card = cv2.resize(card_image, (INITIAL_WIDTH, INITIAL_HEIGHT))

                h, w, d = new_card.shape
                bottomleft = new_card[h-HARD_CODED_HEIGHT:, :HARD_CODED_WIDTH, :]
                bottomright = new_card[h-HARD_CODED_HEIGHT:, w-HARD_CODED_WIDTH:, :]

                grayleft = cv2.cvtColor(bottomleft, cv2.COLOR_BGR2GRAY)
                grayright = cv2.cvtColor(bottomright, cv2.COLOR_BGR2GRAY)

                if SETINFO[j, 3] == 'left':
                    dataset_df.loc[k] = [grayleft.tolist(), 'left', s_id, SETINFO[j, 2]]; k += 1
                    dataset_df.loc[k] = [grayright.tolist(), 'right', 'no', 'no']; k += 1
                    if save_images:
                        cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-left.jpg", grayleft)
                        cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-right.jpg", grayright)
                else:
                    dataset_df.loc[k] = [grayleft.tolist(), 'left', 'no', 'no']; k += 1
                    dataset_df.loc[k] = [grayright.tolist(), 'right', s_id, SETINFO[j, 2]]; k += 1
                    if save_images:
                        cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-left.jpg", grayleft)
                        cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-right.jpg", grayright)
            else:
                print(f"❌ Failed: {s_id}-{i} | Status {response_card.status_code}")

    dataset_df.to_json("Data/Processed/full_dataset.json")
    print("✅ Dataset saved at Data/Processed/full_dataset.json")
    return dataset_df


def reduce_dataset(path: str) -> None:
    """Reduces dataset to max REDUCED_SET entries per set+side and saves as JSON."""
    df = pd.read_json(path)

    def df_side(df: pd.DataFrame, setinfo: np.array, side: str) -> pd.DataFrame:
        setinfo = setinfo[setinfo[:, 3] == side]
        set_list = np.append(setinfo[:, 0], 'no')

        df_small = pd.DataFrame()
        for s_id in set_list:
            filtered = df[(df['set_id'] == s_id) & (df['position'] == side)]
            if len(filtered) > REDUCED_SET:
                sampled = filtered.sample(REDUCED_SET, random_state=42)
            else:
                sampled = filtered
            df_small = pd.concat([df_small, sampled], ignore_index=True)
        return df_small

    df_left = df_side(df, SETINFO, side='left')
    df_right = df_side(df, SETINFO, side='right')

    df_left.to_json('Data/Processed/dict_reduceddataset_left.json')
    df_right.to_json('Data/Processed/dict_reduceddataset_right.json')
