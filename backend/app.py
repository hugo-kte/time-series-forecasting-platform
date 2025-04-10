# backend/app.py
from flask import Flask, jsonify, request
from utils import load_data
from models import ARModel, MAModel, ARMAModel, SARIMAXModel

app = Flask(__name__)

# Instanciation des modèles
models = {
    "ar": ARModel(),
    "ma": MAModel(),
    "arma": ARMAModel(),
    "sarimax": SARIMAXModel()
}

@app.route("/train/<model_name>", methods=["POST"])
def train(model_name):
    if model_name not in models:
        return jsonify({"error": "Modèle non supporté"}), 400
    df = load_data()
    series = df["Consommation"]
    try:
        models[model_name].train(series)
        return jsonify({"message": f"Modèle {model_name.upper()} entraîné avec succès."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/<model_name>", methods=["GET"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": "Modèle non supporté"}), 400
    steps = int(request.args.get("steps", 96))
    try:
        preds = models[model_name].predict(steps=steps)
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
