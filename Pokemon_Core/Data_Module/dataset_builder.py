import os
import pandas as pd
import numpy as np
import requests
import cv2
from PIL import Image
from io import BytesIO

from Pokemon_Core.Data_Module.config import (
    SETINFO, REDUCED_SET,
    INITIAL_WIDTH, INITIAL_HEIGHT,
    HARD_CODED_WIDTH, HARD_CODED_HEIGHT
)


def create_dataset(save_images: bool = False) -> pd.DataFrame:
    """
    Creates a dataset of bottom corners of Pokémon cards for image classification.

    Parameters:
        save_images (bool): If True, saves the processed grayscale corners as .jpg files.

    Returns:
        dataset_df (pd.DataFrame): DataFrame containing:
            - 'corner': Pixel values of the grayscale bottom-left or bottom-right corner.
            - 'position': Either 'left' or 'right'.
            - 'set_id': Set ID string (e.g., 'dv1').
            - 'set_name': Human-readable name of the set.
    """
    # Prepare empty dataset
    dataset_df = pd.DataFrame(columns=['corner', 'position', 'set_id', 'set_name'], index=[0])
    k = 0  # Row index

    # Output directory for JSONs
    json_output_dir = "Data/Json"
    os.makedirs(json_output_dir, exist_ok=True)

    # Optional: image output directory
    # if save_images:
    #     os.makedirs("Data/Processed/Images", exist_ok=True)

    for j in range(SETINFO.shape[0]):
        s_id = SETINFO[j, 0]  # Set ID (e.g., 'dv1')
        print(f'On-going set: {s_id}')

        for i in range(1, int(SETINFO[j, 1]) + 1):
            # Construct image URL (hi-res version)
            image_url = f'https://images.pokemontcg.io/{s_id}/{str(i)}_hires.png'
            response_card = requests.get(image_url)

            if response_card.status_code == 200:
                # Load image into memory
                image = Image.open(BytesIO(response_card.content))
                card_image = np.array(image)

                # Resize to standard dimensions for consistency
                new_card = cv2.resize(card_image, (INITIAL_WIDTH, INITIAL_HEIGHT))
                h, w, d = new_card.shape

                # Extract bottom-left and bottom-right corners
                bottomleft = new_card[h-HARD_CODED_HEIGHT:, :HARD_CODED_WIDTH, :]
                bottomright = new_card[h-HARD_CODED_HEIGHT:, w-HARD_CODED_WIDTH:, :]

                # Convert both corners to grayscale
                grayleft = cv2.cvtColor(bottomleft, cv2.COLOR_BGR2GRAY)
                grayright = cv2.cvtColor(bottomright, cv2.COLOR_BGR2GRAY)

                # Depending on corner side, tag correctly
                if SETINFO[j, 3] == 'left':
                    dataset_df.loc[k] = [grayleft.tolist(), 'left', s_id, SETINFO[j, 2]]; k += 1
                    dataset_df.loc[k] = [grayright.tolist(), 'right', 'no', 'no']; k += 1
                    # if save_images:
                    #     cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-left.jpg", grayleft)
                    #     cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-right.jpg", grayright)
                else:
                    dataset_df.loc[k] = [grayleft.tolist(), 'left', 'no', 'no']; k += 1
                    dataset_df.loc[k] = [grayright.tolist(), 'right', s_id, SETINFO[j, 2]]; k += 1
                    # if save_images:
                    #     cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-left.jpg", grayleft)
                    #     cv2.imwrite(f"Data/Processed/Images/{s_id}-{i}-right.jpg", grayright)

            else:
                print(f"❌ Failed: {s_id}-{i} | Status {response_card.status_code}")

    # Save the full dataset to JSON
    json_path = os.path.join(json_output_dir, "full_dataset.json")
    dataset_df.to_json(json_path)
    print(f"✅ Dataset saved at {json_path}")
    return dataset_df


def reduce_dataset(path: str) -> None:
    """
    Reduces the full dataset to a balanced number of examples per set and side.

    Parameters:
        path (str): Path to the full_dataset.json file.

    Saves:
        - Data/Json/dict_reduceddataset_left.json
        - Data/Json/dict_reduceddataset_right.json
    """
    df = pd.read_json(path)

    def df_side(df: pd.DataFrame, setinfo: np.array, side: str) -> pd.DataFrame:
        """
        Filters and balances dataset for a given side ('left' or 'right').

        Returns:
            A DataFrame containing up to REDUCED_SET entries for each set.
        """
        setinfo = setinfo[setinfo[:, 3] == side]  # Only relevant side
        set_list = np.append(setinfo[:, 0], 'no')  # Include 'no' class for opposite side

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

    os.makedirs("Data/Json", exist_ok=True)
    df_left.to_json("Data/Json/dict_reduceddataset_left.json")
    df_right.to_json("Data/Json/dict_reduceddataset_right.json")
    print("✅ Reduced datasets saved in Data/Json/")
