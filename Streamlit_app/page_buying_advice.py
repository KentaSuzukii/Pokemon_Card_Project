# Streamlit_app/page_buying_advice.py

import streamlit as st
from Pokemon_Core.Prediction_Module.price_predictor import get_best_card

def show_buying_advice_page():
    st.subheader("💰 Suggest undervalued cards to buy")
    st.info("Input your preferences to find the best card deal.")

    # ─── User Inputs ───────────────────────────────────────────────
    budget = st.number_input("Enter your budget (in USD):", min_value=1)
    card_type = st.text_input("Optional: Pokémon type (e.g. Water, Fire, Electric):")
    generation = st.text_input("Optional: Card generation (e.g. Gen 1, Gen 2):")

    # ─── Trigger Search ─────────────────────────────────────────────
    if st.button("🔎 Find undervalued card"):
        if not budget:
            st.warning("Please enter your budget.")
            return

        with st.spinner("Searching for cards..."):
            try:
                best_card = get_best_card(budget, card_type, generation)

                if best_card is not None and not best_card.empty:
                    st.success("✅ Here's your best undervalued card:")
                    st.dataframe(best_card.reset_index(drop=True))
                else:
                    st.warning("No undervalued card found under your filters.")

            except Exception as e:
                st.error("⚠️ An error occurred while searching.")
                st.exception(e)
