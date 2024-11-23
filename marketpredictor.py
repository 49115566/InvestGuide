import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from marketdata import MarketData

class MarketPredictor:
    def __init__(self, ticker, start_date, end_date, n_outputs, n_epochs, n_trials, save_path = None):
        self.data_control = MarketData(ticker, start_date, end_date)
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.n_trials = n_trials
        self.save_path = save_path
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None

    def preprocess_data(self):
        data = self.data_control.data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        self.scaler = scaler
        X, y = [], []
        for i in range(len(data_scaled) - self.n_steps - self.n_outputs + 1):
            X.append(data_scaled[i:i+self.n_steps])
            y.append(data_scaled[i+self.n_steps:i+self.n_steps+self.n_outputs])
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def create_default_model(self, trial):
        self.model = Sequential()
        self.model.add(Input(shape=(self.data_control.data.shape[1])))
        for i in range(trial.suggest_int('n_layers', 1, 3)):
            self.model.add(LSTM(trial.suggest_int('n_units', 50, 200), return_sequences=True))
            self.model.add(Dropout(trial.suggest_float('dropout', 0.1, 0.5)))
        self.model.add(LSTM(trial.suggest_int('n_units', 50, 200)))
        self.model.add(Dropout(trial.suggest_float('dropout', 0.1, 0.5)))
        self.model.add(Dense(self.n_outputs))
        self.model.compile(optimizer='adam', loss='mse')

    def set_model(self, model):
        self.model = model

    def save_model(self):
        self.model.save(self.save_path)

    def load_model(self):
        self.model = load_model(self.save_path)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.n_epochs, validation_data=(self.X_test, self.y_test), callbacks=[EarlyStopping(patience=10)])