import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
import time

# === PARAMÈTRES ===
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 96)
DATA_PATH = "data/full_data.csv"
MODEL_PATH = "backend/models/sarimax_model_final.pkl"
MAX_ITER = 20
GTOL = 1e-3

# === 1. Chargement des données ===
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.dropna(subset=["Consommation"])
df["month"] = df["Datetime"].dt.month
series = df["Consommation"]
mean_val = series.mean()
std_val = series.std()
series_norm = (series - mean_val) / std_val
X = pd.get_dummies(df["month"], prefix="month")
series_norm = series_norm.astype(float)
X = X.astype(float)

print(f"📦 {len(series_norm)} points utilisés pour l'entraînement complet")

# === 2. Entraînement ===
print("🔧 Entraînement SARIMAX (optimisé)...")
start_time = time.time()
model = SARIMAX(series_norm,
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                exog=X,
                enforce_stationarity=False,
                enforce_invertibility=False)

result = model.fit(disp=True, maxiter=MAX_ITER)
print(f"✅ Entraînement terminé en {time.time() - start_time:.2f} sec")

# === 3. Nettoyage du modèle pour pickle ===
def strip_result(r):
    r.model.data.orig_endog = None
    r.model.data.orig_exog = None
    r.model.endog = None
    r.model.exog = None
    r.data = None
    return r

light_result = strip_result(result)
model_pack = {
    "model": light_result,
    "mean": mean_val,
    "std": std_val,
    "exog_columns": X.columns.tolist()
}

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_pack, f)

print(f"💾 Modèle SARIMAX sauvegardé dans {MODEL_PATH}")
