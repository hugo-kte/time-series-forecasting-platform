import os
import pickle
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Correction pour SARIMAX

class ARModel:
    def __init__(self, lags=48, save_path=None):
        self.lags = lags
        self.save_path = save_path
        self.model = None
        self.fitted = None

    def train(self, series):
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.fitted = pickle.load(f)
        else:
            self.model = AutoReg(series, lags=self.lags, old_names=False)
            self.fitted = self.model.fit()
            if self.save_path:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.fitted, f)

    def predict(self, steps):
        return self.fitted.predict(
            start=len(self.fitted.model.endog),
            end=len(self.fitted.model.endog) + steps - 1
        )

class MAModel:
    def __init__(self, order=(0, 0, 1), save_path=None):
        self.order = order
        self.save_path = save_path
        self.model = None
        self.fitted = None

    def train(self, series):
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.fitted = pickle.load(f)
        else:
            self.model = ARIMA(series, order=self.order)
            self.fitted = self.model.fit()
            if self.save_path:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.fitted, f)

    def predict(self, steps):
        return self.fitted.forecast(steps=steps)

class ARMAModel:
    def __init__(self, order=(2, 0, 2), save_path=None):
        self.order = order
        self.save_path = save_path
        self.model = None
        self.fitted = None

    def train(self, series):
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.fitted = pickle.load(f)
        else:
            self.model = ARIMA(series, order=self.order)
            self.fitted = self.model.fit()
            if self.save_path:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.fitted, f)

    def predict(self, steps):
        return self.fitted.forecast(steps=steps)

class SARIMAXModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 96), save_path=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.save_path = save_path
        self.model = None
        self.fitted = None

    def train(self, series):
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.fitted = pickle.load(f)
        else:
            self.model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted = self.model.fit(disp=False)
            if self.save_path:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.fitted, f)

    def predict(self, steps):
        return self.fitted.forecast(steps=steps)
