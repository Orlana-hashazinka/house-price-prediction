# ================================
# Imports
# ================================
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def load_model(name):
    path = BASE_DIR / "model" / f"best_{name}.joblib"
    print(path)
    return joblib.load(path)

def load_importance():
    return pd.read_csv(BASE_DIR / "model" / "permutation_importance.csv")

# ================================
# Charger la ligne de référence
# ================================
def load_reference():
    return pd.read_csv(BASE_DIR / "model" / "reference_input.csv")

# ================================
# Préparer les données
# ================================
def prepare_input(reference, user_inputs):
    input_df = reference.copy()
    for key, value in user_inputs.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value
    return input_df

# ================================
# Prédiction
# ================================
def predict_price(model, input_df):
    prediction_log = model.predict(input_df)[0]
    return np.expm1(prediction_log)

# ================================
# Statistiques marché (train.csv)
# ================================
def load_market_data():
    try:
        return pd.read_csv(BASE_DIR / "data" / "train.csv")
    except FileNotFoundError:
        return None

def get_market_stats(data):
    if data is None:
        return None
    return {
        "median": data["SalePrice"].median(),
        "mean":   data["SalePrice"].mean(),
        "q25":    data["SalePrice"].quantile(0.25),
        "q75":    data["SalePrice"].quantile(0.75),
    }