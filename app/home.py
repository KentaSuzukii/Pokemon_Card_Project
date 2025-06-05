import base64
import streamlit as st

def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def app():
    # ロゴ画像（上）
    main_image_b64 = get_base64_image("Data/Image/logo.jpg")
    st.markdown(
        f"""
        <div style="margin-left: 20%; width: fit-content;">
            <img src="data:image/jpg;base64,{main_image_b64}" width="400" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 2カラム（左右）
    col1, col2 = st.columns(2)

    with col1:
        boy_image_b64 = get_base64_image("Data/Image/boy.jpg")
        st.markdown(
            f"""
            <div style="padding-right: 60px; text-align: center;">
                <img src="data:image/jpg;base64,{boy_image_b64}" width="170" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Find your Pokémon’s true strength!"):
            st.session_state.page_selection = "Find your Pokémon’s true strength!"
            st.experimental_rerun()

    with col2:
        girl_image_b64 = get_base64_image("Data/Image/girl.jpg")
        st.markdown(
            f"""
            <div style="padding-right: 60px; text-align: center;">
                <img src="data:image/jpg;base64,{girl_image_b64}" width="170" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Get help to choose your best card!"):
            st.session_state.page_selection = "Get help to choose your best card!"
            st.experimental_rerun()
