# 📁 Scripts/test_predictor.py

import sys
import cv2
import argparse
from Pokemon_Core.CardProcessing.set_predictor import (
    load_prediction_assets,
    extract_corners,
    predict_set_id,
    extract_ocr_corner
)

def main(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return

    # Load model and encoder
    model, label_encoder = load_prediction_assets()

    # Extract corners
    gray_left, gray_right = extract_corners(img)

    # Predict on both corners
    pred_left = predict_set_id(model, gray_left, label_encoder)
    pred_right = predict_set_id(model, gray_right, label_encoder)

    # Print prediction
    print(f"🔎 Predicted set (left corner):  {pred_left}")
    print(f"🔎 Predicted set (right corner): {pred_right}")

    # Optional: Show OCR crop
    ocr_crop = extract_ocr_corner(img, pred_right)
    cv2.imshow("OCR Crop Region", ocr_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokémon Card Set Prediction")
    parser.add_argument("image", help="Path to the card image")
    args = parser.parse_args()
    main(args.image)
