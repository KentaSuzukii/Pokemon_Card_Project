
import pandas as pd
import os

# Load prediction CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'Raw', 'prediction_model.csv')
df = pd.read_csv(CSV_PATH)

# Ensure card_id is str
df['card_id'] = df['card_id'].astype(str)

def predict_card_value(set_id, card_id):
    """Returns prediction data for a specific card"""
    result = df[(df['set_id'] == set_id) & (df['card_id'] == card_id)]
    if result.empty:
        return {"error": "Card not found."}

    row = result.iloc[0]
    return {
        "set_id": row['set_id'],
        "card_id": row['card_id'],
        "market_price": row['market_price'],
        "predicted_price": row['predicted_price'],
        "adjusted_price": row['predicted_price_adjusted'],
        "valuation": row['over_or_under_valued_log']
    }

def get_best_card_by_price(budget, card_type='', generation=''):
    """Returns card with the highest predicted price under budget"""
    query = df[df['market_price'] <= budget]
    if card_type:
        query = query[query['single_type'].str.lower() == card_type.lower()]
    if generation:
        query = query[query['generation'].str.lower() == generation.lower()]
    if query.empty:
        return {"error": "No cards match the filters within budget."}

    best = query.sort_values('predicted_price', ascending=False).iloc[0]
    return {
        "set_id": best['set_id'],
        "card_id": best['card_id'],
        "predicted_price": best['predicted_price'],
        "market_price": best['market_price'],
        "adjusted_price": best['predicted_price_adjusted'],
        "valuation": best['over_or_under_valued_log']
    }

def get_best_card_by_margin(budget, card_type='', generation=''):
    """Returns card with largest positive margin (predicted - market) under budget"""
    query = df[df['market_price'] <= budget]
    if card_type:
        query = query[query['single_type'].str.lower() == card_type.lower()]
    if generation:
        query = query[query['generation'].str.lower() == generation.lower()]
    if query.empty:
        return {"error": "No cards match the filters within budget."}

    query = query.copy()
    query['margin'] = query['predicted_price'] - query['market_price']
    best = query.sort_values('margin', ascending=False).iloc[0]
    return {
        "set_id": best['set_id'],
        "card_id": best['card_id'],
        "predicted_price": best['predicted_price'],
        "market_price": best['market_price'],
        "adjusted_price": best['predicted_price_adjusted'],
        "valuation": best['over_or_under_valued_log'],
        "margin": best['margin']
    }
