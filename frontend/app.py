import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration de base ===
BACKEND_URL = "http://localhost:8000"
MODELS = ["ar", "ma", "arma", "sarimax"]

st.set_page_config(page_title="Prévisions Électriques", layout="wide")
st.title("⚡ Plateforme de Prévision de la Consommation Électrique")

# === Sidebar : sélection ===
st.sidebar.header("Configuration")
model = st.sidebar.selectbox("Modèle à utiliser", MODELS)
steps = st.sidebar.slider("Nombre de pas à prédire", 24, 288, 96)  # entre 6h et 3 jours

if st.sidebar.button("Lancer la prévision"):
    try:
        st.info("⏳ Prévision en cours...")
        res = requests.get(f"{BACKEND_URL}/predict/{model}?steps={steps}")
        res.raise_for_status()

        preds = res.json().get("predictions", [])
        if preds:
            st.success(f"✅ Prévision réalisée avec le modèle {model.upper()}")

            df_pred = pd.DataFrame(preds, columns=["Prévision"])
            df_pred.index.name = "Index"

            # === Affichage ===
            st.line_chart(df_pred)

            with st.expander("Voir les données brutes"):
                st.dataframe(df_pred)

        else:
            st.warning("Aucune donnée prédite.")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion au back-end : {e}")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
