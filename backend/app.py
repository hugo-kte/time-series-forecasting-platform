from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

MODEL_DIR = "backend/models"
DATA_PATH = "data/full_data.csv"

# === Chargement des infos de normalisation ===
df = pd.read_csv(DATA_PATH)
series = df["Consommation"].dropna().astype(float)
mean_val = series.mean()
std_val = series.std()
series_norm = (series - mean_val) / std_val

@app.route("/")
def home():
    return jsonify({"message": "Bienvenue sur la plateforme de prévision 👋"})

@app.route("/predict/<model_name>")
def predict(model_name):
    steps = int(request.args.get("steps", 96))  # Par défaut : 1 jour

    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        return jsonify({"error": f"Modèle '{model_name}' introuvable."}), 404

    try:
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)

        # Si c’est un dict (SARIMAX avec normalisation stockée)
        if isinstance(model_obj, dict):
            model = model_obj["model"]
            mean_val_local = model_obj["mean"]
            std_val_local = model_obj["std"]
            start = len(model.data.endog)
            end = start + steps - 1
            pred_norm = model.predict(start=start, end=end)
            pred_real = pred_norm * std_val_local + mean_val_local
        else:
            # AR/MA/ARMA simples (modèle déjà normalisé à l'entraînement)
            start = len(series_norm)
            end = start + steps - 1
            pred_norm = model_obj.predict(start=start, end=end)
            pred_real = pred_norm * std_val + mean_val

        return jsonify({"predictions": pred_real.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
