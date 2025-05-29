# =========================
# 📁 Scripts/run_dataset_reduction.py
# =========================

from Pokemon_Core.dataset_builder import reduce_dataset

if __name__ == "__main__":
    print("✂️ Reducing full dataset into balanced training sets...")
    reduce_dataset("Data/Processed/full_dataset.json")
    print("✅ Reduced datasets saved (left & right corners).")
