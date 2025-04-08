# backend/model.py
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

class ARModel:
    def __init__(self, lags=48):
        self.lags = lags
        self.model = None
        self.fitted_model = None

    def train(self, df):
        df = df.sort_values("Datetime")
        self.model = AutoReg(df["Consommation"], lags=self.lags, old_names=False)
        self.fitted_model = self.model.fit()

    def predict(self, steps=96):
        if self.fitted_model is None:
            raise ValueError("Le modèle n'est pas encore entraîné.")
        return self.fitted_model.predict(start=len(self.fitted_model.model.endog),
                                         end=len(self.fitted_model.model.endog) + steps - 1)
