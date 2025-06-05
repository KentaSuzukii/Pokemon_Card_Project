import streamlit as st
from Pokemon_Core.Prediction_Module.prediction_model import recommendation_biggest_margin_card, recommendation_best_card, market_predicted_price
from Pokemon_Core.Image_Module.modelling import recognize_card_from_photo
import pandas as pd
import sys
from PIL import Image
import io
import tensorflow as tf
import pickle
import requests
from io import BytesIO
import base64



def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    base64_img = base64.b64encode(data).decode()

    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        height: 100vh;
    }}

    /* Light mode */
    @media (prefers-color-scheme: light) {{
        section.main {{
            background-color: rgba(255, 255, 255, 0.7);
            color: #222222;
        }}
    }}

    /* Dark mode */
    @media (prefers-color-scheme: dark) {{
        section.main {{
            background-color: rgba(30, 30, 30, 0.7);
            color: #f0f0f0;
        }}
    }}

    section.main {{
        padding: 1rem 2rem;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def app():
    #Image Detection
    set_background("Data/Image/background.jpg")

    st.title("Overvalued? UnderValued?")

    st.markdown(
    '<p style="color: blue; font-size: 30px;">Upload and Display an Image</p>',
    unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    @st.cache_resource
    def load_models():
        model_left = tf.keras.models.load_model("Data/Processed/Organized_Images/Left/mobilenet_left_model.h5")
        with open("Data/Processed/Organized_Images/Left/le_left.pkl", "rb") as f:
            le_left = pickle.load(f)
        model_right = tf.keras.models.load_model("Data/Processed/Organized_Images/Right/mobilenet_right_model.h5")
        with open("Data/Processed/Organized_Images/Right/le_right.pkl", "rb") as f:
            le_right = pickle.load(f)
        return model_left, le_left, model_right, le_right
    model_left, le_left, model_right, le_right = load_models()

    if uploaded_file is not None:
        result = recognize_card_from_photo(uploaded_file, model_left, le_left, model_right, le_right)

        if result is not None:
            card_id_i = result.get("poke_id")
            set_id_i = result.get("set_id")

            if card_id_i is None or set_id_i is None:
                st.error("Recognition incomplete: missing 'poke_id' or 'set_id'. Please enter them manually.")
            else:
                try:
                    prediction_result = market_predicted_price(card_id_i, set_id_i)

                    if isinstance(prediction_result, tuple):
                        market_price_0,predicted_price_0,valuation_0, img_0 = prediction_result
                        buffered = io.BytesIO()
                        img_0.save(buffered, format="PNG")  # PIL.Imageの場合
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        if valuation_0 == "overvalued":
                            st.markdown(f'<p style="color: red; font-size: 20px;">Market Price:{market_price_0:.2f}EUR</p>',
    unsafe_allow_html=True)
                            st.markdown(f'<p style="color: red; font-size: 20px;">Expected Price:{predicted_price_0:.2f}EUR</p>',
    unsafe_allow_html=True)
                            st.markdown(
    f"""
    <div style="border: 3px solid red; display: inline-block;">
        <img src="data:image/png;base64,{img_base64}" width="500" />
        <p style="text-align: center;">Pokémon Card Image</p>
    </div>
    """,
    unsafe_allow_html=True
)
                        else:
                            st.markdown(f'<p style="color: green; font-size: 20px;">Market Price:{market_price_0:.2f}EUR</p>',
    unsafe_allow_html=True)
                            st.markdown(f'<p style="color: green; font-size: 20px;">Expected Price:{predicted_price_0:.2f}EUR</p>',
    unsafe_allow_html=True)
                            st.markdown(
    f"""
    <div style="border: 3px solid green; display: inline-block;">
        <img src="data:image/png;base64,{img_base64}" width="500" />
        <p style="text-align: center;">Pokémon Card Image</p>
    </div>
    """,
    unsafe_allow_html=True
)
                        st.markdown(f"**Card ID:** {card_id_i} | **Set ID:** {set_id_i}")
                    else:
                        st.markdown(prediction_result)

                except Exception as e:
                    st.error("An error occurred during price prediction.")
                    st.exception(e)

        else:
            st.warning("Card not recognized. Some sets may not be supported yet.")
    else:
        st.info("Please upload a valid Pokémon card image.")


    st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)


    #If the image scanning doesn't work

    st.markdown(
    '<p style="color: blue; font-size: 30px;">Manual Input</p>',
    unsafe_allow_html=True)

    card_id = str(st.number_input("Enter your card id", min_value=1, value=1))

    path = "Data/Raw/pokemon_set_ids_and_names.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"CSV file not found at {path}. Please check the path and file location.")
        st.stop()

    set_id = st.selectbox("Enter a Set ID", df["set_id"])
    selected_name = df[df["set_id"] == set_id]["set_name"].values[0]
    st.write(f"Set Name: **{selected_name}**")

    if st.button("Get Market and Predicted Price"):
        try:
            result = market_predicted_price(card_id, set_id)

            # Handle both string-only and (text, image) return types
            if isinstance(result, tuple):
                market_price_1,predicted_price_1,valuation_1, img_1 = result
                buffered = io.BytesIO()
                img_1.save(buffered, format="PNG")  # PIL.Imageの場合
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                if valuation_1 == "overvalued":
                    st.markdown(f'<p style="color: red; font-size: 20px;">Market Price:{market_price_1:.2f}EUR</p>',
unsafe_allow_html=True)
                    st.markdown(f'<p style="color: red; font-size: 20px;">Expected Price:{predicted_price_1:.2f}EUR</p>',
unsafe_allow_html=True)
                    st.markdown(
f"""
<div style="border: 3px solid red; display: inline-block;">
<img src="data:image/png;base64,{img_base64}" width="500" />
<p style="text-align: center;">Pokémon Card Image</p>
</div>
""",
unsafe_allow_html=True
)
                else:
                    st.markdown(f'<p style="color: green; font-size: 20px;">Market Price:{market_price_1:.2f}EUR</p>',
unsafe_allow_html=True)
                    st.markdown(f'<p style="color: green; font-size: 20px;">Expected Price:{predicted_price_1:.2f}EUR</p>',
unsafe_allow_html=True)
                    st.markdown(
f"""
<div style="border: 3px solid green; display: inline-block;">
<img src="data:image/png;base64,{img_base64}" width="500" />
<p style="text-align: center;">Pokémon Card Image</p>
</div>
""",
unsafe_allow_html=True
)
            else:
                st.markdown(result)

        except Exception as e:
            st.markdown(f"Oops, something went wrong, please check your inputs, especailly the set id")


    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
    st.write("All Pokémon Set IDs and Names")
    st.dataframe(df)

    st.markdown("You can refer to this website for set symbols and set IDs: https://www.justinbasil.com/guide/appendix1")

import os
print(os.path.exists("Data/Image/background.jpg"))
