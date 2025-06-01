# Streamlit_app/ui_helpers.py

import streamlit as st

def inject_custom_css(css_path):
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file not found.")
