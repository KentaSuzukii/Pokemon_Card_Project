import os
import matplotlib.pyplot as plt
import requests
import pandas as pd
from PIL import Image
from io import BytesIO


path = "/Users/suzukikenta/code/Pokemon_Card_Project/Notebooks/Kenta"
model_df = pd.read_csv(os.path.join(path, "prediction_model.csv"))

def over_or_undervalued(card_id, set_id):
    new_df = model_df[(model_df["card_id"] == card_id) & (model_df["set_id"] == set_id)]
    print(new_df["over_or_under_valued_log"].values[0])

over_or_undervalued("1", "base1")

def recommendation_best_card(budget, type, generation):
    poke_types = model_df["single_type"].unique().tolist()
    generations = model_df["generation"].unique().tolist()

    filtered_df = model_df[
        (model_df["market_price"] <= budget) &
        ((model_df["single_type"] == type) | (type not in poke_types)) &
        ((model_df["generation"] == generation) | (generation not in generations))
    ]

    # Filter to best predicted price
    filtered_df = filtered_df[filtered_df["predicted_price"] == filtered_df["predicted_price"].max()]

    # Final output
    filtered_df = filtered_df[['set_id', 'card_id', 'market_price', 'predicted_price', 'predicted_price_adjusted']]
    filtered_df = filtered_df.sort_values(by='market_price', ascending=True)

    url = f"https://images.pokemontcg.io/{filtered_df['set_id'].iloc[0]}/{filtered_df['card_id'].iloc[0]}_hires.png"

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')  # Optional: hides axis
    plt.show()

    print(f"The predicted price is {filtered_df['predicted_price'].values[0]:.1f}EUR. The market price is {round(filtered_df['market_price'].values[0],1)} EUR.")

recommendation_best_card(1000, "Fire", "")


def recommendation_biggest_margin_card(budget, type, generation):
    poke_types = model_df["single_type"].unique().tolist()
    generations = model_df["generation"].unique().tolist()

    filtered_df = model_df[
        (model_df["market_price"] <= budget) &
        ((model_df["single_type"] == type) | (type not in poke_types)) &
        ((model_df["generation"] == generation) | (generation not in generations))
    ]

    # Filter to best predicted price
    diff = (filtered_df["predicted_price"] - filtered_df["market_price"])
    filtered_df = filtered_df[diff == diff.max()]

    # Final output
    filtered_df = filtered_df[['set_id', 'card_id', 'market_price', 'predicted_price', 'predicted_price_adjusted']]
    filtered_df = filtered_df.sort_values(by='market_price', ascending=True)

    url = f"https://images.pokemontcg.io/{filtered_df['set_id'].iloc[0]}/{filtered_df['card_id'].iloc[0]}_hires.png"

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')  # Optional: hides axis
    plt.show()

    print(f"The predicted price is {filtered_df['predicted_price'].values[0]:.1f}EUR. The market price is {round(filtered_df['market_price'].values[0],1)} EUR.")

recommendation_biggest_margin_card(1000, "Fire", "")
