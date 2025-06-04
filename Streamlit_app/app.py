# Streamlit_app/app.py

import os
import sys

# Fix module import path so that 'Pokemon_Core' is recognized
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import streamlit.web.cli as stcli
from page_identifier import show_identifier_page
from page_buying_advice import show_buying_advice_page
from ui_helpers import inject_custom_css

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

# Made a separate run_streamlit.py file for the below code and it is placed in Scripts folder
# if __name__ == "__main__":
#     port = os.environ.get("PORT", "8501")
#     sys.argv = [
#         "streamlit", "run", "Streamlit_app/app.py",
#         "--server.port", port,
#         "--server.address", "0.0.0.0",  # REQUIRED for Cloud Run!
#         "--server.enableCORS=false"
#     ]
#     sys.exit(stcli.main())
