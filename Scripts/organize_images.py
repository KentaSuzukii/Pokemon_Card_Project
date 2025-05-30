# =========================
# 📁 Scripts/organize_images.py
# =========================

"""
This script organizes the processed Pokémon card corner images into structured subfolders
based on their position ('left' or 'right') and their set ID.

📦 Input:
- Images from: Data/Processed/Images/
- Filenames follow the format: <set_id>-<card_number>-<position>.jpg
  e.g., sv3pt5-5-left.jpg

🎯 Output:
- Organized images saved into:
    - Data/Processed/Organized_Images/left/<set_id>/
    - Data/Processed/Organized_Images/right/<set_id>/

🔧 How it works:
- For each image, the script:
    1. Parses the set ID and position from the filename.
    2. Creates a subdirectory based on the set ID inside the corresponding 'left' or 'right' folder.
    3. Moves the image to its new location.

👨‍💻 Usage:
Run from the project root:
    python Scripts/organize_images.py
"""

import os
import shutil

SOURCE_FOLDER = "Data/Processed/Images"
DEST_FOLDER = "Data/Processed/Organized_Images"

# Ensure destination folders exist
for side in ["left", "right"]:
    os.makedirs(os.path.join(DEST_FOLDER, side), exist_ok=True)

# Loop over all images
for filename in os.listdir(SOURCE_FOLDER):
    if not filename.endswith(".jpg"):
        continue

    parts = filename.split("-")
    if len(parts) != 3:
        print(f"Skipping malformed file: {filename}")
        continue

    set_id, number, side_ext = parts
    side = side_ext.split(".")[0]

    dest_dir = os.path.join(DEST_FOLDER, side, set_id)
    os.makedirs(dest_dir, exist_ok=True)

    src = os.path.join(SOURCE_FOLDER, filename)
    dst = os.path.join(dest_dir, filename)

    shutil.copy(src, dst)

print("✅ Images organized by set and label.")
