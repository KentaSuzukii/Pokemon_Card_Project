# Streamlit_app/price_predictor.py

import pandas as pd
import os

# ─────────────────────────────────────────────
# 📂 Load prediction_model.csv safely
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', '..', 'Data', 'Raw', 'prediction_model.csv')

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ CSV not found at {CSV_PATH}")

# Ensure card_id is string
df['card_id'] = df['card_id'].astype(str)

# ─────────────────────────────────────────────
# 🔍 1. Predict values for a specific card
# ─────────────────────────────────────────────
def predict_card_value(set_id, card_id):
    result = df[(df['set_id'] == set_id) & (df['card_id'] == str(card_id))]
    if result.empty:
        return {"error": "Card not found."}
    return result.iloc[0].to_dict()


# ─────────────────────────────────────────────
# 💰 2. Highest predicted price under budget
# ─────────────────────────────────────────────
def get_best_card_by_price(budget, card_type='', generation=''):
    query = df[df['market_price'] <= budget]
    if card_type:
        query = query[query['single_type'].str.lower() == card_type.lower()]
    if generation:
        query = query[query['generation'].str.lower() == generation.lower()]
    if query.empty:
        return None
    return query.sort_values('predicted_price', ascending=False).head(1)


# ─────────────────────────────────────────────
# 📈 3. Largest margin (predicted - market)
# ─────────────────────────────────────────────
def get_best_card_by_margin(budget, card_type='', generation=''):
    query = df[df['market_price'] <= budget]
    if card_type:
        query = query[query['single_type'].str.lower() == card_type.lower()]
    if generation:
        query = query[query['generation'].str.lower() == generation.lower()]
    if query.empty:
        return None
    query = query.copy()
    query['margin'] = query['predicted_price'] - query['market_price']
    return query.sort_values('margin', ascending=False).head(1)


# ─────────────────────────────────────────────
# 🧠 4. Main function for Streamlit integration
# Currently uses margin method by default
# ─────────────────────────────────────────────
def get_best_card(budget, card_type='', generation=''):
    return get_best_card_by_margin(budget, card_type, generation)
