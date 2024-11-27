import yfinance as yf
import pandas as pd

class DataFetcher:
    """
    A class used to fetch historical stock data for a given ticker symbol.

    Methods
    -------
    fetch(ticker: str, start_date: str, end_date: str) -> pd.DataFrame
        Fetches historical stock data for the specified ticker symbol between the given start and end dates.
    """
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical stock data for the specified ticker symbol between the given start and end dates.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the stock to fetch data for.
        start_date : str
            The start date for the data fetch in 'YYYY-MM-DD' format.
        end_date : str
            The end date for the data fetch in 'YYYY-MM-DD' format.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the historical stock data with columns for Open, High, Low, Close, Adj Close, and Volume.
        """
        data = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.columns.name = None
        return data