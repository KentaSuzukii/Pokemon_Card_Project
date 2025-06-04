import streamlit as st
from app import home, over_under_valued, recommendation

PAGES = {
    "Home": home,
    "Find your Pokémon’s true strength!": over_under_valued,
    "Get help to choose your best card!": recommendation
}

def main():
    # Default to "Home" if nothing is selected yet
    if "page_selection" not in st.session_state:
        st.session_state.page_selection = "Home"

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()),
                                 index=list(PAGES.keys()).index(st.session_state.page_selection))

    st.session_state.page_selection = selection
    PAGES[selection].app()

if __name__ == "__main__":
    main()
