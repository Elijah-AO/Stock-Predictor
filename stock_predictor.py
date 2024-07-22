# TODO: argparse
import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import save_model
from sklearn.preprocessing import MinMaxScaler
class StockPredictor():
    def __init__(self, ticker, start='2000-01-01', end=datetime.today().strftime('%Y-%m-%d'), model="LSTM"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.model = model
        self.model_data = None
        self.model_path = f'models/{self.ticker}_{self.model}.keras'
        self.csv_path = f'data/{self.ticker}.csv'


    def get_data(self):
        if not os.path.exists(self.csv_path):
            print("Downloading ticker data")
            data = yf.download(self.ticker, start=self.start, end=self.end)
            data.to_csv(self.csv_path)
        else:
            print("Retrieving data from CSV")
        self.data = pd.read_csv(f"data/{self.ticker}.csv")



    def get_model(self):
        if not os.path.exists(self.model_path):
            self.train()
        else:
            self.model = load_model(f'{self.ticker}_{self.model}.keras')

    def preprocessing(self, train_ratio=0.8):
        df = self.data.copy()
        self.dates = df['Date']
        df['Date'] = pd.to_datetime(df['Date'])
        X = df.drop(['Adj Close', 'Close', 'Date'], axis=1)
        y = df['Adj Close']

        train_size = int(len(df) * train_ratio)

        self.X_train = X.iloc[:train_size]
        self.X_test = X.iloc[train_size:]
        self.y_train = y.iloc[:train_size]
        self.y_test = y.iloc[train_size:]

        self.dates_train = self.dates.iloc[:train_size]
        self.dates_test = self.dates.iloc[train_size:]

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X_train = pd.DataFrame(self.scaler_X.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler_X.transform(self.X_test), columns=self.X_test.columns)

        self.y_train = pd.DataFrame(self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)))



    def train_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        self.y_pred_scaled = model.predict(self.X_test)

        self.y_pred = self.scaler_y.inverse_transform(self.y_pred_scaled)

        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.mean(self.y_test) * 100

        print(f'Error: {nrmse: .3f}%')
        self.model_data = model
        return nrmse  

    def train_lstm(self):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(5, 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        X_train = np.array(self.X_train)
        y_train = np.array(self.y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model.fit(X_train, y_train, batch_size=10, epochs=10)

        save_model(model, f'models/{self.ticker}_{self.model}.keras')

        X_test = np.array(self.X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        self.y_pred_scaled = model.predict(X_test)

        self.y_pred = self.scaler_y.inverse_transform(self.y_pred_scaled)

        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.mean(self.y_test) * 100

        print(f'Error: {nrmse: .3f}%')
        self.model_data = model


    def train(self):
        match self.model:
            case "LSTM":
                self.train_lstm()
            case "Regression":
                self.train_regression()
            case _:
                raise ValueError("Invalid model name")

    def plot_data(self, predictions=False):
        plt.xticks(range(0, len(self.data['Date']), 500), self.data['Date'].loc[::500], rotation=45)
        plt.title(f'{self.ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        if predictions:
            plt.plot(self.dates_test, self.y_test, label='Actual')
            plt.plot(self.dates_test, self.y_pred, label='Predicted')

        else:
            plt.plot(self.data['Date'], self.data['Adj Close'])

        plt.legend()
        plt.show()

    def plot_predictions(self):
            plt.figure(figsize=(12, 6))
            plt.plot(self.dates_test, self.y_test, label='Actual')
            plt.plot(self.dates_test, self.y_pred, label='Predicted')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.title('Actual vs Predicted Prices for ' + self.ticker)
            plt.legend()
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=len(self.dates_test) // 200))
            plt.show()

    def predict_next(self):
        last = self.data.iloc[-1].drop(['Adj Close', 'Close', 'Date'])
        last = self.scaler_X.transform([last])
        last = np.reshape(last, (last.shape[0], last.shape[1], 1))
        prediction = self.model_data.predict(last)
        prediction = self.scaler_y.inverse_transform(prediction)
        print(f'Next price: {prediction[0][0]}')
        
