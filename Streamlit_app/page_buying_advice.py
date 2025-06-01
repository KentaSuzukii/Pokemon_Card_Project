# Streamlit_app/page_buying_advice.py

import streamlit as st

def show_buying_advice_page():
    st.subheader("💰 Suggest undervalued cards to buy")
    st.info("Input your preferences to find the best card deal.")

    budget = st.number_input("Enter your budget (in USD):", min_value=1)
    card_type = st.text_input("Optional: Pokémon type (e.g. Water, Fire, Electric):")
    generation = st.text_input("Optional: Card generation (e.g. Gen 1, Gen 2):")

    if st.button("🔎 Find undervalued card"):
        if budget:
            st.success("Searching for cards...")
            # 🔌 Plug in your prediction logic here
        else:
            st.warning("Please enter your budget.")
