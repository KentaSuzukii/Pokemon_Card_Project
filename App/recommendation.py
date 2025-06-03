import streamlit as st
from Pokemon_Core.Prediction_Module.prediction_model import recommendation_biggest_margin_card, recommendation_best_card, market_predicted_price
import pandas as pd
import base64
import time
import re

def app():
    ### Best Card Withing Budget ###
    st.header("Recommendation for Pokémon Cards")

    st.subheader("Best Card within Budget")

    budget = st.number_input("Enter your budget (EUR)", min_value=0, value=100,key='budget')

    poke_type = st.selectbox("Select Pokémon type", [
    "None", "Fire", "Water", "Grass", "Psychic", "Colorless", "Fighting",
        "Lightning", "Darkness", "Metal", "Dragon", "Fairy", "None"])

    generation = st.selectbox("Select Generation", ["None", "First", "Second", "Third","Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "None"])

    if st.button("Get Recommendation"):
        text_output, image = recommendation_best_card(budget, poke_type, generation)

        placeholder = st.empty()

        match = re.search(r"The predicted price is ([\d.]+) EUR", text_output)
        predicted_price = float(match.group(1)) if match else 0

        if predicted_price > 800:  # your video condition
            with open("Data/Video/masterball.mp4", "rb") as video_file:
                video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            video_html = f"""
            <video width="700" autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            """
            placeholder.markdown(video_html, unsafe_allow_html=True)
            time.sleep(12)

        # Now replace the entire placeholder with both text and image
        with placeholder.container():
            st.markdown(text_output)
            st.image(image, caption="Recommended Pokémon Card", use_container_width=True)



    ### Card with the Biggest Margin ###
    st.subheader("Biggest Margin Card")

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
        text_output_1, image_1 = recommendation_biggest_margin_card(budget_1, poke_type_1, generation_1)

        placeholder = st.empty()

        # Extract predicted price from the text
        match = re.search(r"The predicted price is ([\d.]+) EUR", text_output_1)
        predicted_price = float(match.group(1)) if match else 0

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
            time.sleep(12)

        # Replace video with result
        with placeholder.container():
            st.markdown(text_output_1)
            st.image(image_1, caption="Recommended Pokémon Card", use_container_width=True)
