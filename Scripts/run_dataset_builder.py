# =========================
# 📁 Scripts/run_dataset_builder.py
# =========================

from Pokemon_Core.dataset_builder import create_dataset

if __name__ == "__main__":
    print("📦 Creating Pokémon card training dataset...")
    create_dataset(save_images=True)
    print("✅ Full dataset created and saved.")
