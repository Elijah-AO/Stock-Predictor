import yfinance as yf
import pandas as pd
from datetime import datetime

class StockPredictor():
    def __init__(self) -> None:
        pass
        
    
    
    
today = datetime.today().strftime('%Y-%m-%d')
print(today)
ticker = 'ABDP.L'
stock_data = yf.download(ticker, start='2010-01-01', end=today)
print(len(stock_data))
print(stock_data.tail())

if __name__ == "__main__":
    pass