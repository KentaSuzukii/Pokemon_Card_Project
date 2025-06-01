import os
import matplotlib.pyplot as plt
import requests
import pandas as pd
from PIL import Image
from io import BytesIO


script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "../../Data/Raw/prediction_model.csv")
path_1 = os.path.join(script_dir, "../../Data/Raw/market_price.csv")
model_df = pd.read_csv(path)
market_df = pd.read_csv(path_1)


def market_predicted_price(card_id, set_id):
    market_match = market_df[(market_df['card_id'] == card_id) & (market_df['set_id'] == set_id)]
    model_match = model_df[(model_df['card_id'] == card_id) & (model_df['set_id'] == set_id)]

    # If no market data is found
    if market_match.empty:
        return f"❌ No market data found for card {card_id} in set {set_id}."

    market_price = market_match['market_price'].values[0]

    # If no model prediction is found
    if model_match.empty:
        return f"Market Price: {market_price:.2f} EUR. We don't have the predicted price for this card."

    predicted_price = model_match['predicted_price'].values[0]
    valuation = model_match['over_or_under_valued_log'].values[0]

    url = f"https://images.pokemontcg.io/{market_match['set_id'].iloc[0]}/{market_match['card_id'].iloc[0]}_hires.png"

    response = requests.get(url)
    img_0 = Image.open(BytesIO(response.content))

    return f"Market Price: {market_price:.2f} EUR, Predicted Price: {predicted_price:.2f} EUR, Under/Over Valued: {valuation}", img_0


def recommendation_best_card(budget, poke_type, generation):
    poke_types = model_df["single_type"].unique().tolist()
    generations = model_df["generation"].unique().tolist()

    filtered_df = model_df[
        (model_df["market_price"] <= budget) &
        ((model_df["single_type"] == poke_type) | (poke_type not in poke_types)) &
        ((model_df["generation"] == generation) | (generation not in generations))
    ]

    filtered_df = filtered_df[filtered_df["predicted_price"] == filtered_df["predicted_price"].max()]

    filtered_df = filtered_df.sort_values(by='market_price', ascending=True)

    url = f"https://images.pokemontcg.io/{filtered_df['set_id'].iloc[0]}/{filtered_df['card_id'].iloc[0]}_hires.png"

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    predicted_price = filtered_df['predicted_price'].values[0]
    market_price = filtered_df['market_price'].values[0]

    text_output = f"The predicted price is {predicted_price:.1f} EUR. The market price is {market_price:.1f} EUR."

    # Return both text and image
    return text_output, img



def recommendation_biggest_margin_card(budget, poke_type, generation):
    poke_types = model_df["single_type"].unique().tolist()
    generations = model_df["generation"].unique().tolist()

    filtered_df = model_df[
        (model_df["market_price"] <= budget) &
        ((model_df["single_type"] == poke_type) | (poke_type not in poke_types)) &
        ((model_df["generation"] == generation) | (generation not in generations))
    ]

    diff = (filtered_df["predicted_price"] - filtered_df["market_price"])
    filtered_df = filtered_df[diff == diff.max()]

    filtered_df = filtered_df.sort_values(by='market_price', ascending=True)

    url = f"https://images.pokemontcg.io/{filtered_df['set_id'].iloc[0]}/{filtered_df['card_id'].iloc[0]}_hires.png"

    response = requests.get(url)
    img_1 = Image.open(BytesIO(response.content))

    predicted_price = filtered_df['predicted_price'].values[0]
    market_price = filtered_df['market_price'].values[0]

    text_output_1 = f"The predicted price is {predicted_price:.1f} EUR. The market price is {market_price:.1f} EUR."

    # Return both text and image
    return text_output_1, img_1
