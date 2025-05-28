from Pokemon_Core.dataset_builder import create_dataset
import pandas as pd
import os

# ✅ Ensure output directory exists
output_path = "Data/Processed"
os.makedirs(output_path, exist_ok=True)

# 🛠 Create the full dataset
print("📦 Creating Pokémon card training dataset...")
df = create_dataset()

# 💾 Save to disk
save_path = os.path.join(output_path, "full_dataset.json")
df.to_json(save_path)
print(f"✅ Dataset saved at {save_path}")
