import streamlit as st
from Pokemon_Core.Prediction_Module.prediction_model import recommendation_biggest_margin_card, recommendation_best_card, market_predicted_price
from Pokemon_Core.Image_Module.modelling import recognize_card_from_photo
import pandas as pd
import sys
from PIL import Image
import io
import tensorflow as tf
import pickle


def app():
    #Image Detection
    st.title("Overvalued? UnderValued?")

    st.subheader("Upload and Display an Image")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read image file as bytes
        bytes_data = uploaded_file.read()

        # Open image with PIL
        image = Image.open(io.BytesIO(bytes_data))

        # Display image in the app
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # @st.cache_resource
    # def load_models():
    #     model_left = tf.keras.models.load_model("Data/Processed/Organized_Images/Left/mobilenet_left_model.h5")
    #     with open("Data/Processed/Organized_Images/Left/le_left.pkl", "rb") as f:
    #         le_left = pickle.load(f)
    #     model_right = tf.keras.models.load_model("Data/Processed/Organized_Images/Right/mobilenet_right_model.h5")
    #     with open("Data/Processed/Organized_Images/Right/le_right.pkl", "rb") as f:
    #         le_right = pickle.load(f)
    #     return model_left, le_left, model_right, le_right

    # model_left, le_left, model_right, le_right = load_models()



    a = recognize_card_from_photo(uploaded_file,model_left, le_left, model_right, le_right)

    card_id_i = a["poke_id"]
    set_id_i = a["set_id"]

    st.markdown(f"**Card ID:** {card_id_i} | **Set ID:** {set_id_i}")

    st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)

    #If the image scanning doesn't work

    st.subheader("If the scanning didn't work, you can enter the card id and set id manually")

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
                text_output_0, image_0 = result
                st.markdown(text_output_0)
                st.image(image_0, caption="Pokémon Card Image")
            else:
                st.markdown(result)

        except Exception as e:
            st.markdown(f"Oops, something went wrong, please check your inputs, especailly the set id")


    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
    st.write("All Pokémon Set IDs and Names")
    st.dataframe(df)

    st.markdown("You can refer to this website for set symbols and set IDs: https://www.justinbasil.com/guide/appendix1")
