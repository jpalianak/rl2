import yfinance as yf
import pandas as pd


def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    # df = df['Close'][ticker].to_frame()  # type: ignore
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']  # type: ignore
    # df = df.rename(columns={ticker: "Close"})
    return df
