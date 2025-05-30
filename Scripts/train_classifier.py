# 📁 Scripts/train_classifier.py

# The below 2 lines are added because my laptop has old GPU and there was some mismatch with the versions, etc. This forces CPU usage.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from Pokemon_Core.CardProcessing.set_classifier import train_symbols_model

def main():
    parser = argparse.ArgumentParser(description="Train Set Classifier CNN model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/Processed/dict_reduceddataset_left.json",
        help="Path to the JSON dataset containing corner crops."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    model, history, cm, le = train_symbols_model(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
