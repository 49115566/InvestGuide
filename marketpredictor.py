import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from marketdata import MarketData

class MarketPredictor:
    def __init__(self, ticker, start_date, end_date, added_features, n_epochs, X_scaler=None, X_train=None, X_validate=None, X_test=None, y_scaler = None, y_train=None, y_validate=None, y_test=None, n_steps=15, model=None):
        self.data_control = MarketData(ticker, start_date, end_date)
        self.data_control.add_features(added_features)
        self.data_control.fix_missing_values()
        self.data_control.engineer.add_custom_feature(self.data_control.data, lambda data: data['Close'].pct_change(), 'pct_change').add_custom_feature(self.data_control.data, lambda data: data['pct_change'].shift(-1), 'pct_change_next')
        self.data_control.data.dropna(inplace=True)
        self.n_epochs = n_epochs
        self.X_scaler = X_scaler
        self.X_train = X_train
        self.X_validate = X_validate
        self.X_test = X_test
        self.y_scaler = y_scaler
        self.y_train = y_train
        self.y_validate = y_validate
        self.y_test = y_test
        self.n_steps = n_steps
        self.model = model

    def preprocess_data(self):
        self.X_scaler = self.X_scaler or MinMaxScaler()
        self.y_scaler = self.y_scaler or MinMaxScaler()
        
        # Scale X values
        X_data = self.data_control.data.drop(columns=['pct_change_next'])
        X_scaled = self.X_scaler.fit_transform(X_data)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_data.columns)
        
        # Scale y values
        y_data = self.data_control.data['pct_change_next'].values.reshape(-1, 1)
        y_scaled = self.y_scaler.fit_transform(y_data)
        
        if self.X_train is None or self.X_validate is None or self.X_test is None or self.y_train is None or self.y_validate is None or self.y_test is None:
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(X_scaled_df, y_scaled, test_size=0.2, shuffle=False)
            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X_temp, y_temp, test_size=0.25, shuffle=False)
        
        self.X_train = np.array([self.X_train[i:i+self.n_steps] for i in range(len(self.X_train)-self.n_steps)])
        self.X_validate = np.array([self.X_validate[i:i+self.n_steps] for i in range(len(self.X_validate)-self.n_steps)])
        self.X_test = np.array([self.X_test[i:i+self.n_steps] for i in range(len(self.X_test)-self.n_steps)])
        self.y_train = np.array(self.y_train[self.n_steps:])
        self.y_validate = np.array(self.y_validate[self.n_steps:])
        self.y_test = np.array(self.y_test[self.n_steps:])
        
    def create_default_model(self, n_layers=1, n_units=100, dropout=0.2):
        self.model = Sequential()
        self.model.add(Input(shape=(self.n_steps, self.X_train.shape[2])))
        for i in range(n_layers):
            self.model.add(LSTM(n_units, return_sequences=True))
            self.model.add(Dropout(dropout))
        self.model.add(LSTM(n_units))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(1))  # Fixed output shape of 1
        self.model.compile(optimizer='adam', loss='mse')  # or use loss=Huber()

    def set_model(self, model):
        self.model = model

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, save_path):
        self.model = load_model(save_path)

    def save_data(self, save_path):
        self.data_control.to_feather(save_path)

    def load_data(self, save_path):
        self.data_control.data = self.data_control.read_feather(save_path)

    def train_model(self, patience=50):
        self.model.fit(self.X_train, self.y_train, epochs=self.n_epochs, validation_data=(self.X_validate, self.y_validate), callbacks=[EarlyStopping(patience=patience)])

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))  # Reshape y_pred to match the scaler's expected input
        y_test = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1))  # Reshape y_test to match the scaler's expected input
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, r2
    
    def predict_tomorrow(self):
        data = self.data_control.data.drop(columns=['pct_change_next'])
        data_scaled = self.X_scaler.transform(data)
        X = np.array([data_scaled[-self.n_steps:]])
        y_pred = self.model.predict(X)
        y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        return y_pred
