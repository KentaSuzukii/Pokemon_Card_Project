import streamlit as st
from Pokemon_Core.Prediction_Module.prediction_model import recommendation_biggest_margin_card, recommendation_best_card, market_predicted_price
import pandas as pd
import base64
import time
import re

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
    section.main {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1rem 2rem;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def app():
    ### Best Card Withing Budget ###

    set_background("Data/Image/background.jpg")

    st.title("Pokémon Cards Recommendation")

    st.markdown(
    '<p style="color: blue; font-size: 30px;">Best Card within Budget</p>',
    unsafe_allow_html=True)

    budget = st.number_input("Enter your budget (EUR)", min_value=0, value=100,key='budget')

    poke_type = st.selectbox("Select Pokémon type", [
    "None", "Fire", "Water", "Grass", "Psychic", "Colorless", "Fighting",
        "Lightning", "Darkness", "Metal", "Dragon", "Fairy", "None"])

    generation = st.selectbox("Select Generation", ["None", "First", "Second", "Third","Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "None"])

    if st.button("Get Recommendation"):
        market_price,predicted_price, image = recommendation_best_card(budget, poke_type, generation)

        placeholder = st.empty()

        if predicted_price > 800:  # your video condition
            with open("Data/Video/masterball.mp4", "rb") as video_file:
                video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            video_html = f"""
            <video width="500" autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            """
            placeholder.markdown(video_html, unsafe_allow_html=True)
            time.sleep(10)

        # Now replace the entire placeholder with both text and image
        with placeholder.container():
            st.markdown(f'<p style="color: green; font-size: 20px;">Market Price:{market_price:.2f}EUR</p>',
    unsafe_allow_html=True)
            st.markdown(f'<p style="color: green; font-size: 20px;">Predicted Price:{predicted_price:.2f}EUR</p>',
    unsafe_allow_html=True)

            st.image(image, caption="Recommended Pokémon Card", width=500)



    ### Card with the Biggest Margin ###
    st.markdown(
    '<p style="color: blue; font-size: 30px;">Biggest Margin Card within Budget</p>',
    unsafe_allow_html=True)

    budget_1 = st.number_input("Enter your budget (EUR)", min_value=0, value=100,key="budget_1")

    poke_type_1 = st.selectbox(
        "Select Pokémon type",
        ["None", "Fire", "Water", "Grass", "Psychic", "Colorless", "Fighting",
        "Lightning", "Darkness", "Metal", "Dragon", "Fairy", "None"]
        ,key="poke_type_1")

    generation_1 = st.selectbox(
        "Select Generation",
        ["None","First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth"],
        key="generation_1"
    )


    if st.button("Get Recommendation", key="margin_card_button"):
        market_price, predicted_price, image_1 = recommendation_biggest_margin_card(budget_1, poke_type_1, generation_1)

        placeholder = st.empty()

        if predicted_price > 800:
            with open("Data/Video/masterball.mp4", "rb") as video_file:
                video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            video_html = f"""
            <video width="700" autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            """
            placeholder.markdown(video_html, unsafe_allow_html=True)
            time.sleep(8)

        # Replace video with result
        with placeholder.container():
            st.markdown(f'<p style="color: green; font-size: 20px;">Market Price:{market_price:.2f}EUR</p>',
    unsafe_allow_html=True)
            st.markdown(f'<p style="color: green; font-size: 20px;">Predicted Price:{predicted_price:.2f}EUR</p>',
    unsafe_allow_html=True)

            st.image(image_1, caption="Recommended Pokémon Card", width=500)
