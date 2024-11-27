from typing import Optional, List
import pandas as pd
from datafetcher import DataFetcher
from dataengineer import DataEngineer
from datafiller import DataFiller
from datavisualizer import DataVisualizer
from datafilemanager import DataFileManager

class MarketData:
    """
    A class to manage market data fetching, feature engineering, missing value handling, visualization, and file operations.

    Attributes
    ----------
    ticker : str
        The ticker symbol of the stock.
    start_date : str
        The start date for fetching data in 'YYYY-MM-DD' format.
    end_date : str
        The end date for fetching data in 'YYYY-MM-DD' format.
    fetcher : DataFetcher, optional
        An instance of DataFetcher to fetch the data (default is None, which initializes a new DataFetcher).
    data : pd.DataFrame, optional
        The market data (default is None).
    engineer : DataEngineer
        An instance of DataEngineer for feature engineering.
    filler : DataFiller
        An instance of DataFiller for handling missing values.
    visualizer : DataVisualizer
        An instance of DataVisualizer for plotting data.
    filemanager : DataFileManager
        An instance of DataFileManager for file operations.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str, fetcher: Optional[DataFetcher] = None, fetch_data: bool = True, engineer: Optional[DataEngineer] = None, filler: Optional[DataFiller] = None, visualizer: Optional[DataVisualizer] = None, filemanager: Optional[DataFileManager] = None, data: Optional[pd.DataFrame] = None):
        """
        Initializes the MarketData class with the given parameters.

        Notes
        -----
        If fetch_data is True, it will fetch the data using the DataFetcher instance.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.fetcher = fetcher or DataFetcher()
        if fetch_data:
            self.fetch_data()
        else:
            self.data = data
        self.engineer = engineer or DataEngineer()
        self.filler = filler or DataFiller()
        self.visualizer = visualizer or DataVisualizer()
        self.filemanager = filemanager or DataFileManager()

    def fetch_data(self) -> None:
        """
        Fetches the market data for the specified ticker and date range.

        Notes
        -----
        This method uses the DataFetcher instance to fetch the data.
        """
        self.data = self.fetcher.fetch(self.ticker, self.start_date, self.end_date)

    def fetch_data_with_features(self, features: List[str]) -> None:
        """
        Fetches the market data and adds the specified features.

        Parameters
        ----------
        features : list
            A list of feature names to add to the data.

        Notes
        -----
        This method first fetches the data and then adds the specified features using the DataEngineer instance.
        """
        self.fetch_data()
        self.add_features(features)

    def fetch_data_with_all_features(self, exclude: List[str] = []) -> None:
        """
        Fetches the market data and adds all available features, excluding specified ones.

        Parameters
        ----------
        exclude : list, optional
            A list of feature names to exclude (default is an empty list).

        Notes
        -----
        This method first fetches the data and then adds all available features using the DataEngineer instance, excluding the specified ones.
        """
        self.fetch_data()
        self.add_all_features(exclude=exclude)

    def add_features(self, features: List[str]) -> None:
        """
        Adds the specified features to the data.

        Parameters
        ----------
        features : list
            A list of feature names to add to the data.

        Notes
        -----
        This method uses the DataEngineer instance to add the specified features to the data.
        """
        self.engineer.add_features(data=self.data, features=features, ticker=self.ticker, start_date=self.start_date, end_date=self.end_date)

    def add_all_features(self, exclude: List[str] = []) -> None:
        """
        Adds all available features to the data, excluding specified ones.

        Parameters
        ----------
        exclude : list, optional
            A list of feature names to exclude (default is an empty list).

        Notes
        -----
        This method uses the DataEngineer instance to add all available features to the data, excluding the specified ones.
        """
        self.engineer.add_all_features(data=self.data, exclude=exclude, ticker=self.ticker, start_date=self.start_date, end_date=self.end_date)

    def fix_missing_values(self, strategy: str = 'bfill') -> None:
        """
        Fixes missing values in the data using the specified strategy.

        Parameters
        ----------
        strategy : str, optional
            The strategy to use for fixing missing values (default is 'bfill').

        Notes
        -----
        This method uses the DataFiller instance to fix missing values in the data.
        """
        self.filler.fix_missing_values(data=self.data, strategy=strategy)

    def plot(self, features: List[str] = [], save_file: str = "plot.png", start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots the specified features.

        Parameters
        ----------
        features : list, optional
            A list of feature names to plot (default is an empty list).
        save_file : str, optional
            The name of the file to save the plot (default is "plot.png").
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).

        Notes
        -----
        This method uses the DataVisualizer instance to plot the specified features.
        """
        self.visualizer.plot(data=self.data, ticker=self.ticker, features=features, save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_all_features(self, exclude: List[str] = [], save_file: str = "plot.png", start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots all available features, excluding specified ones.

        Parameters
        ----------
        exclude : list, optional
            A list of feature names to exclude (default is an empty list).
        save_file : str, optional
            The name of the file to save the plot (default is "plot.png").
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).

        Notes
        -----
        This method uses the DataVisualizer instance to plot all available features, excluding the specified ones.
        """
        self.visualizer.plot_all_features(data=self.data, ticker=self.ticker, exclude=exclude, save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_non_volume_features(self, save_file: str = 'plot.png', start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots all non-volume features.

        Parameters
        ----------
        save_file : str, optional
            The name of the file to save the plot (default is 'plot.png').
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).

        Notes
        -----
        This method uses the DataVisualizer instance to plot all non-volume features.
        """
        self.visualizer.plot_non_volume_features(data=self.data, ticker=self.ticker, save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_volume_features(self, save_file: str = 'plot.png', start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots all volume features.

        Parameters
        ----------
        save_file : str, optional
            The name of the file to save the plot (default is 'plot.png').
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).

        Notes
        -----
        This method uses the DataVisualizer instance to plot all volume features.
        """
        self.visualizer.plot_volume_features(data=self.data, ticker=self.ticker, save_file=save_file, start_date=start_date, end_date=end_date)

    def to_csv(self, file_name: str = 'data.csv') -> None:
        """
        Saves the data to a CSV file.

        Parameters
        ----------
        file_name : str, optional
            The name of the CSV file to save (default is 'data.csv').

        Notes
        -----
        This method uses the DataFileManager instance to save the data to a CSV file.
        """
        self.filemanager.to_csv(self.data, file_name)

    def to_feather(self, file_name: str = 'data.feather') -> None:
        """
        Saves the data to a Feather file.

        Parameters
        ----------
        file_name : str, optional
            The name of the Feather file to save (default is 'data.feather').

        Notes
        -----
        This method uses the DataFileManager instance to save the data to a Feather file.
        """
        self.filemanager.to_feather(self.data, file_name)

    def read_csv(self, file_name: str = 'data.csv') -> None:
        """
        Loads the data from a CSV file.

        Parameters
        ----------
        file_name : str, optional
            The name of the CSV file to load (default is 'data.csv').

        Notes
        -----
        This method uses the DataFileManager instance to load the data from a CSV file.
        """
        self.data = self.filemanager.read_csv(file_name)

    def read_feather(self, file_name: str = 'data.feather') -> None:
        """
        Loads the data from a Feather file.

        Parameters
        ----------
        file_name : str, optional
            The name of the Feather file to load (default is 'data.feather').

        Notes
        -----
        This method uses the DataFileManager instance to load the data from a Feather file.
        """
        self.data = self.filemanager.read_feather(file_name)