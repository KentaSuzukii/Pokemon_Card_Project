import streamlit as st
from PIL import Image

def app():
    # Main image (header)
    main_image = Image.open("Data/Image/logo.jpg")
    st.image(main_image, width=600)

    # Two columns = two tab buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image("Data/Image/boy.jpg", width=170)
        if st.button("Find your Pokémon’s true strength!"):
            st.session_state.page_selection = "Find your Pokémon’s true strength!"
            st.rerun()

    with col2:
        st.image("Data/Image/girl.jpg", width=170)
        if st.button("Get help to choose your best card!"):
            st.session_state.page_selection = "Get help to choose your best card!"
            st.rerun()
