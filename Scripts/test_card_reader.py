# 📁 Scripts/test_card_reader.py

import cv2
import argparse
from Pokemon_Core.CardProcessing.card_number_reader import extract_card_number

def main(image_path: str, set_id: str):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image from {image_path}")
        return

    # Run OCR pipeline
    card_number = extract_card_number(img, set_id)
    print(f"✅ Predicted Card Number: {card_number}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR card number extraction.")
    parser.add_argument("image_path", type=str, help="Path to the Pokémon card image")
    parser.add_argument("set_id", type=str, help="Set ID for layout guidance (e.g., sv3)")
    args = parser.parse_args()

    main(args.image_path, args.set_id)
