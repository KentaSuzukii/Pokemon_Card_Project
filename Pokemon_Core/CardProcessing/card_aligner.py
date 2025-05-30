import cv2
import numpy as np
from Pokemon_Core.config import HIRES_WIDTH, HIRES_HEIGHT, DEBUG_MODE

def detect_card_contour(image: np.ndarray) -> np.ndarray:
    """
    Detects the largest quadrilateral contour in the image, assumed to be the card.

    Args:
        image (np.ndarray): Input BGR image

    Returns:
        np.ndarray: 4 corner points sorted [top-left, top-right, bottom-left, bottom-right]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return sort_corners(pts)

    raise ValueError("❌ Could not detect card contour.")

def sort_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sorts the 4 corner points in the order:
    top-left, top-right, bottom-left, bottom-right
    """
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.float32([top_left, top_right, bottom_left, bottom_right])

def deform_card(image: np.ndarray) -> np.ndarray:
    """
    Automatically detects and warps the Pokémon card to a standard size.

    Args:
        image (np.ndarray): Original BGR image of the card

    Returns:
        np.ndarray: Warped and aligned card image
    """
    try:
        src_pts = detect_card_contour(image)
    except ValueError as e:
        print(e)
        return image  # fallback: return original image

    dst_pts = np.float32([
        [0, 0],
        [HIRES_WIDTH - 1, 0],
        [0, HIRES_HEIGHT - 1],
        [HIRES_WIDTH - 1, HIRES_HEIGHT - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (HIRES_WIDTH, HIRES_HEIGHT))

    if DEBUG_MODE:
        debug_img = image.copy()
        for pt in src_pts.astype(int):
            cv2.circle(debug_img, tuple(pt), 8, (0, 255, 0), -1)
        cv2.imshow("Card Contour Detection", debug_img)
        cv2.imshow("Warped Output", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped
