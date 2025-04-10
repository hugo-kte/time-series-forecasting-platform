import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
import pickle

# === PARAMÈTRES ===
ORDER = (1, 1, 1)         # ARIMA(p,d,q)
SEASONAL_ORDER = (1, 1, 1, 96)  # (P, D, Q, s) → ici s=96 (jour complet si données toutes les 15 min)
TRAIN_RATIO = 0.8

# === 1. Charger les données ===
df = pd.read_csv("data/full_data.csv")
series = df["Consommation"].dropna()

# === 2. Normalisation manuelle ===
mean_val = series.mean()
std_val = series.std()
series_norm = (series - mean_val) / std_val

# === 3. Split train/test ===
train_size = int(len(series_norm) * TRAIN_RATIO)
train, test = series_norm[:train_size], series_norm[train_size:]

print("📦 Données prêtes :", len(train), "train |", len(test), "test")

# === 4. Entraînement du modèle SARIMAX ===
print("🔧 Entraînement SARIMAX...")
model = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)
print("✅ Modèle SARIMAX entraîné")

# === 5. Sauvegarde avec pickle ===
os.makedirs("backend/models", exist_ok=True)
with open("backend/models/sarimax_model.pkl", "wb") as f:
    pickle.dump(result, f)
print("💾 Modèle SARIMAX sauvegardé dans backend/models/sarimax_model.pkl")

# === 6. Rechargement et prédiction ===
with open("backend/models/sarimax_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
pred_norm = loaded_model.predict(start=len(train), end=len(series_norm) - 1)

# === 7. Dénormalisation ===
test_real = test * std_val + mean_val
pred_real = pred_norm * std_val + mean_val
train_tail_real = train[-len(test):] * std_val + mean_val

# === 8. Affichage comparatif ===
plt.figure(figsize=(12, 5))
plt.plot(test_real.values, label="Données réelles", color="blue")
plt.plot(pred_real.values, label="Prévisions", color="orange")
plt.plot(train_tail_real.values, label="Train (derniers)", color="green", alpha=0.5)
plt.title(f"Modèle SARIMAX - Consommation (order={ORDER}, seasonal_order={SEASONAL_ORDER})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
