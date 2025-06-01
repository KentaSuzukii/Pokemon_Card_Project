import streamlit as st
from Pokemon_Core.Prediction_Module import recommendation_biggest_margin_card, recommendation_best_card, market_predicted_price
import pandas as pd



# Load your dataframe here
model_df = pd.read_csv("Data/market_price.csv")




### Over or Under Valued Cards ###
st.header("Over or Under Valued Cards")

card_id = str(st.number_input("Enter your card id", min_value=0, value=250))
set_id = st.text_input("Enter your set ID", value="swsh1")

if st.button("Get Market and Predicted Price"):
    result = market_predicted_price(card_id, set_id)

    # Handle both string-only and (text, image) return types
    if isinstance(result, tuple):
        text_output_0, image_0 = result
        st.markdown(text_output_0)
        st.image(image_0, caption="Pokémon Card Image")
    else:
        st.markdown(result)





### Best Card Withing Budget ###
st.header("Pokémon Budget Card Recommendation (Best Card Within Budget)")

budget = st.number_input("Enter your budget (EUR)", min_value=0, value=1500)

poke_type = st.selectbox("Select Pokémon type", [
   "None", "Fire", "Water", "Grass", "Psychic", "Colorless", "Fighting",
    "Lightning", "Darkness", "Metal", "Dragon", "Fairy", "None"])

generation = st.selectbox("Select Generation", ["None", "First", "Second", "Third","Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "None"])

if st.button("Get Recommendation"):
    # Call function with your data frame and inputs
    text_output, image = recommendation_best_card(budget, poke_type, generation)

    # Display text
    st.markdown(text_output)

    # Display image
    st.image(image, caption="Recommended Pokémon Card")





### Card with the Biggest Margin ###
st.header("Pokémon Budget Card Recommendation (Biggest Margin Between Predicted and Market Price)")

budget_1 = st.number_input("Enter your budget (EUR)", min_value=0, value=50)

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


if st.button("Get Recommendation",key="biggest_margin_button"):
    # Call function with your data frame and inputs
    text_output_1, image_1 = recommendation_biggest_margin_card(budget_1, poke_type_1, generation_1)

    # Display text
    st.markdown(text_output_1)

    # Display image
    st.image(image_1, caption="Recommended Pokémon Card")
