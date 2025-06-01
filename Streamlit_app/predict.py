# Streamlit_app/predict.py

import numpy as np
from PIL import Image
from Pokemon_Core.Image_Module.prediction import predict_card_set
from Pokemon_Core.Image_Module.text_detection import read_card_number

def identify_card_from_image(uploaded_image: Image.Image):
    """
    Takes a PIL image, runs the full pipeline:
    - Predict set ID
    - Try OCR for card number
    Returns a dictionary with results.
    """
    # Convert image to numpy array (as required by model)
    np_img = np.array(uploaded_image)

    # Step 1: Predict card set
    set_id = predict_card_set(np_img)

    # Step 2: Try OCR to get card number
    card_number = read_card_number(np_img)

    # Step 3: Return result
    return {
        "set_id": set_id,
        "card_number": card_number
    }
