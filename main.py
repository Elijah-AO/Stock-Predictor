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
        self.model = "Regression"
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
        X = df.drop(['Adj Close', 'Date'], axis=1)
        y = df['Adj Close']

        train_size = int(len(df) * train_ratio)

        self.X_train = X.iloc[:train_size]
        self.X_test = X.iloc[train_size:]
        self.y_train = y.iloc[:train_size]
        self.y_test = y.iloc[train_size:]

        self.dates_train = self.dates.iloc[:train_size]
        self.dates_test = self.dates.iloc[train_size:]

    def train_regression(self):
        model = LinearRegression()

        model.fit(self.X_train, self.y_train)

        self.y_pred = model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.mean(self.y_test) * 100

        print(f'Error: {nrmse: .3f}%')

    def train_lstm(self):
        pass

    def train(self):
        match self.model:
            case "LSTM":
                self.train_lstm()
            case "Regression":
                self.train_regression()
            case _:
                raise ValueError("Invalid model")

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


def main(ticker):
    predictor = StockPredictor(ticker)
    predictor.get_data()
    predictor.preprocessing()
    predictor.train()
    predictor.plot_data(predictions=True)

if __name__ == "__main__":
    main('AAPL')

# TODO: MinMax Scaler vs standard scaler
