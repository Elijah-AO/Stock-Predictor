from stock_predictor import StockPredictor
import argparse


def main(ticker, model):
    stock_predictor = StockPredictor(ticker=ticker, model=model)
    stock_predictor.get_data()
    stock_predictor.preprocessing()
    stock_predictor.train_regression()
    stock_predictor.plot_predictions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="The stock ticker to predict")
    parser.add_argument("--model", help="The model to use for prediction", default="lstm")
    args = parser.parse_args()
    main(args.ticker, args.model)

