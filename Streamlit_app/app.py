# Streamlit_app/app.py

import streamlit as st
from Streamlit_app.page_identifier import show_identifier_page
from Streamlit_app.page_buying_advice import show_buying_advice_page
from Streamlit_app.ui_helpers import inject_custom_css

# Page setup
st.set_page_config(page_title="Pokémon Card Analyzer", layout="wide")
inject_custom_css("Streamlit_app/CSS/style.css")

# Header
st.markdown("<h1 style='text-align: center;'>📸 Pokémon Card Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload cards or find great deals on undervalued Pokémon cards!</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio(
    "Choose an option:",
    ("📷 Identify Card from Image", "💰 Find Undervalued Cards to Buy")
)

# Route to appropriate page
if page == "📷 Identify Card from Image":
    show_identifier_page()
else:
    show_buying_advice_page()
