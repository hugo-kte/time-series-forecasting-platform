# backend/app.py
from flask import Flask, jsonify, request
from model import ARModel
from utils import load_data

app = Flask(__name__)
model = ARModel()

@app.route("/train", methods=["POST"])
def train():
    df = load_data()
    model.train(df)
    return jsonify({"message": "Modèle AR entraîné avec succès."})

@app.route("/predict", methods=["GET"])
def predict():
    steps = int(request.args.get("steps", 96))
    try:
        preds = model.predict(steps=steps)
        return jsonify({"predictions": preds.tolist()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
