# =========================
# 📁 Pokemon_Core/image_utils.py
# =========================

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple


# Folder where processed corner images are saved
IMAGE_FOLDER = "Data/Processed/Images/"


def load_image(filename: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image from the Images folder.

    Args:
        filename (str): Name of the image file (e.g., 'dv1-1.jpg')
        grayscale (bool): Whether to load in grayscale (default: True)

    Returns:
        np.ndarray: Loaded image
    """
    path = os.path.join(IMAGE_FOLDER, filename)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)

    if image is None:
        raise FileNotFoundError(f"Image not found at: {path}")

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
