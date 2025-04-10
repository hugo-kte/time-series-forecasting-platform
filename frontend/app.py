import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

#BACKEND_URL = "http://backend:8000"  # Si Dockerisé
BACKEND_URL = "http://localhost:8000"  # Pour test en local

st.title("Prévisions de Consommation")

model = st.selectbox("Choisir un modèle", ["ar", "ma", "arma", "sarimax"])

if st.button("Entraîner le modèle"):
    res = requests.post(f"{BACKEND_URL}/train/{model}")
    st.success(res.json().get("message", "Entraînement terminé"))

if st.button("Prédire"):
    steps = st.slider("Nombre de pas à prédire", 1, 96, 48)
    res = requests.get(f"{BACKEND_URL}/predict/{model}?steps={steps}")
    if res.status_code == 200:
        preds = res.json()["predictions"]
        st.line_chart(pd.Series(preds, name="Prévision"))
    else:
        st.error(res.json().get("error", "Erreur inconnue"))

if st.button("Voir les métriques"):
    res = requests.get(f"{BACKEND_URL}/metrics/{model}?steps=48")
    if res.status_code == 200:
        st.write(res.json())
    else:
        st.error(res.json().get("error", "Erreur inconnue"))
