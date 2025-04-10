import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import os

# === PARAMÈTRES ===
MA_ORDER = 10                # Pas trop élevé pour garder la stabilité
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

print("✅ Données préparées :", len(train), "train |", len(test), "test")

# === 4. Entraînement du modèle MA (ARIMA avec p=0, d=0, q=MA_ORDER) ===
model = ARIMA(train, order=(0, 0, MA_ORDER))
result = model.fit(method_kwargs={"maxiter": 30})  # ✅ fonctionne sans erreur
print("✅ Modèle MA entraîné")

# === 5. Sauvegarde du modèle ===
os.makedirs("backend/models", exist_ok=True)
result.save("backend/models/ma_model.pkl")
print("✅ Modèle MA sauvegardé dans backend/models/ma_model.pkl")

# === 6. Rechargement et prédiction ===
loaded_model = ARIMAResults.load("backend/models/ma_model.pkl")
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
plt.title(f"Modèle MA (q={MA_ORDER}) - Consommation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
