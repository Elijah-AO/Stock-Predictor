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
class StockPredictor():
    def __init__(self, ticker, start='2000-01-01', end=datetime.today().strftime('%Y-%m-%d')):
        self.ticker = ticker
        self.start = start
        self.end = end 
    
    def download_data(self):
        data = yf.download(self.ticker, start=self.start, end=self.end)
        self.csv_path = f'{self.ticker}.csv'
        data.to_csv(self.csv_path)

    def load_data(self):
        if not f"{self.ticker}.csv" in os.listdir() :
            self.download_data()
            print("downloadesd")
            
        df = pd.read_csv(f"{self.ticker}.csv")
        self.dates = df['Date']
        print(df.head)
        df['Date'] = pd.to_datetime(df['Date'])
        X = df.drop(['Adj Close','Close', 'Date'], axis=1)
        y = df['Adj Close']
        
        train_size = int(len(df) * 0.8)
        
        self.X_train = X.iloc[:train_size]
        self.X_test = X.iloc[train_size:]
        self.y_train = y.iloc[:train_size]
        self.y_test = y.iloc[train_size:]

        self.dates_train = self.dates.iloc[:train_size]
        self.dates_test = self.dates.iloc[train_size:]
        
    def train_regression(self):
        self.model = LinearRegression()
        
        self.model.fit(self.X_train, self.y_train)
        
        self.y_pred = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.mean(self.y_test) * 100  
        
        print(f'Error: {nrmse: .3f}%')
    
        
    def plot_predictions(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates_test, self.y_test, label='Actual')
        plt.plot(self.dates_test, self.y_pred, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.title('Actual vs Predicted Adjusted Close Prices')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.show()
        
        
def main(ticker):
    predictor = StockPredictor(ticker)
    predictor.load_data()
    predictor.train_regression()
    predictor.plot_predictions()


if __name__ == "__main__":
    '''parser = argparse.ArgumentParser(
        prog='Stock Predictor',
        description='Stock Predictor')
    parser.add_argument('ticker', type=str, help='The ticker of the stock that you would like to predict')
    args = parser.parse_args()
    main(args.ticker)'''
    main("NVDA")
