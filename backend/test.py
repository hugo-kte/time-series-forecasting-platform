import logging
import pandas as pd
from models import SARIMAXModel

# Configuration basique des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Lecture du fichier CSV 'backend/full_data.csv'")
data = pd.read_csv('backend/full_data.csv')

# On utilise la colonne 'Consommation' qui est présente dans le CSV
if 'Consommation' not in data.columns:
    logging.error("La colonne 'Consommation' n'a pas été trouvée. Colonnes disponibles: %s", data.columns.tolist())
    raise KeyError("'Consommation' column is missing in the CSV file.")

logging.info("Extraction de la série temporelle depuis la colonne 'Consommation'")
series = data["Consommation"]

train_size = int(len(series) * 0.8)
train_series = series.iloc[:train_size]
logging.info("Sélection de %d points pour l'entraînement", train_size)

logging.info("Initialisation du modèle SARIMAX")
model = SARIMAXModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), save_path='sarimax_model.pkl')

logging.info("Début de l'entraînement du modèle")
model.train(train_series)
logging.info("Entraînement terminé et modèle sauvegardé (si save_path est défini)")

logging.info("Début de la prédiction pour 10 périodes")
forecast = model.predict(steps=10)
logging.info("Prédiction terminée")
logging.info("Prévisions : %s", forecast)

print("Prévisions :", forecast)