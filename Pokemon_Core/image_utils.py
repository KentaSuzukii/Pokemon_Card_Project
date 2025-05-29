# =========================
# 📁 Pokemon_Core/image_utils.py
# =========================

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple

# Automatically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "Data", "Processed", "Images")


def load_image(filename: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image from the Images folder using an absolute path.

    Args:
        filename (str): Name of the image file (e.g., 'dv1-1.jpg')
        grayscale (bool): Whether to load in grayscale (default: True)

    Returns:
        np.ndarray: Loaded image
    """
    base_dir = os.path.dirname(__file__)  # e.g., Pokemon_Core/
    path = os.path.abspath(os.path.join(base_dir, "..", "Data", "Processed", "Images", filename))

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)

    if image is None:
        raise FileNotFoundError(f"Image not found at ABSOLUTE path: {path}")

    return image



def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the given size.

    Args:
        image (np.ndarray): The input image
        size (tuple): (width, height)

    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, size)


def show_image(image: np.ndarray, title: str = "Image") -> None:
    """
    Show an image using matplotlib (auto handles grayscale or color).

    Args:
        image (np.ndarray): Image to display
        title (str): Window title
    """
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()
