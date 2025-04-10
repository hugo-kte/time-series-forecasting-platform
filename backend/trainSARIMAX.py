import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
import pickle
import time

# === PARAMÈTRES ===
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 96)
TRAIN_RATIO = 0.8
DATA_PATH = "data/full_data.csv"
MODEL_PATH = "backend/models/sarimax_model.pkl"
MAX_ITER = 35
GTOL = 1e-4

# === 1. Chargement et préparation des données ===
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.dropna(subset=["Consommation"])
df["month"] = df["Datetime"].dt.month
series = df["Consommation"]

# Normalisation
mean_val = series.mean()
std_val = series.std()
series_norm = (series - mean_val) / std_val

# Variables exogènes : mois encodés
X = pd.get_dummies(df["month"], prefix="month")

# Split train/test
train_size = int(len(series_norm) * TRAIN_RATIO)
train, test = series_norm[:train_size], series_norm[train_size:]
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

# Convertir en float pour éviter les bugs avec des booléens
train = train.astype(float)
test = test.astype(float)
X_train = X_train.astype(float)
X_test = X_test.astype(float)



print(f"📦 {len(train)} points pour l'entraînement, {len(test)} pour le test")
print("🔧 Entraînement SARIMAX (avec arrêt anticipé)...")

# === 2. Entraînement du modèle SARIMAX ===
start_time = time.time()
model = SARIMAX(train,
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                exog=X_train,
                enforce_stationarity=False,
                enforce_invertibility=False)

result = model.fit(disp=True, maxiter=MAX_ITER)
duration = time.time() - start_time
print(f"✅ Entraînement terminé en {duration:.2f} secondes")

# === 3. Analyse post-optimisation ===
print("📈 Résultat de l'optimisation :")
print(result.mle_retvals)

# === 4. Nettoyage du modèle avant sauvegarde ===
def strip_model_result(result):
    result.model.data.orig_endog = None
    result.model.data.orig_exog = None
    result.model.endog = None
    result.model.exog = None
    result.data = None
    return result

light_result = strip_model_result(result)

model_pack = {
    "model": light_result,
    "mean": mean_val,
    "std": std_val,
    "exog_columns": X_train.columns.tolist()
}

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_pack, f)

print(f"💾 Modèle SARIMAX sauvegardé dans {MODEL_PATH}")

# === 5. Prédiction sur la période de test ===
with open(MODEL_PATH, "rb") as f:
    loaded_pack = pickle.load(f)

model = loaded_pack["model"]
mean_val = loaded_pack["mean"]
std_val = loaded_pack["std"]
exog_columns = loaded_pack["exog_columns"]

X_test_full = pd.get_dummies(df["month"].iloc[-len(test):], prefix="month")
for col in exog_columns:
    if col not in X_test_full.columns:
        X_test_full[col] = 0
X_test_full = X_test_full[exog_columns]

pred_norm = model.predict(start=len(train), end=len(train) + len(test) - 1, exog=X_test_full)
pred_real = pred_norm * std_val + mean_val
test_real = test * std_val + mean_val
train_tail_real = train[-len(test):] * std_val + mean_val

# === 6. Affichage graphique ===
plt.figure(figsize=(12, 5))
plt.plot(test_real.values, label="Données réelles", color="blue")
plt.plot(pred_real.values, label="Prévisions", color="orange")
plt.plot(train_tail_real.values, label="Train (derniers)", color="green", alpha=0.5)
plt.title(f"Modèle SARIMAX - Consommation (order={ORDER}, seasonal={SEASONAL_ORDER})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
