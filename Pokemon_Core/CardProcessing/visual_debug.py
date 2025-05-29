# 📁 Pokemon_Core/Image/draw.py

import cv2
import numpy as np
from Pokemon_Core.config import DEBUG_MODE

def draw_prediction(
    image: np.ndarray,
    set_id: str = None,
    card_number: str = None,
    label: str = None,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Draws prediction information (set ID, card number, label) on the image.

    Args:
        image (np.ndarray): Input image (BGR).
        set_id (str, optional): Predicted or known set ID.
        card_number (str, optional): OCR-extracted card number.
        label (str, optional): Additional label or prediction text.
        color (tuple): BGR color for the overlay text and box.

    Returns:
        np.ndarray: Image with annotation overlay.
    """
    overlay = image.copy()
    H, W = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    y_offset = 40

    items = []
    if set_id:
        items.append(f"Set: {set_id}")
    if card_number:
        items.append(f"Card #: {card_number}")
    if label:
        items.append(f"Label: {label}")

    for idx, text in enumerate(items):
        cv2.putText(
            overlay,
            text,
            (20, y_offset + idx * 40),
            font,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA
        )

    return overlay

def show_if_debug(image: np.ndarray, win_name: str = "DEBUG VIEW") -> None:
    """
    Displays the image if DEBUG_MODE is True.

    Args:
        image (np.ndarray): Image to show.
        win_name (str): Window name.
    """
    if DEBUG_MODE:
        cv2.imshow(win_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
