import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt

class DataVisualizer:
    def plot(self, data: pd.DataFrame, ticker: str, features: List[str] = [], save_file: str = "plot.png", start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots the specified features.
        
        Parameters
        ----------
        features : list
            A list of feature names to plot.
        save_file : str, optional
            The name of the file to save the plot (default is "plot.png").
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).
        
        Notes
        -----
        If start_date and end_date are provided, the data is filtered to that date range before plotting.
        The plot is saved to the specified file.
        """
        if start_date is not None and end_date is not None:
            data = data.loc[start_date:end_date]
        
        data[features].plot(figsize=(14, 7), title=f'{ticker} {features}', grid=True)
        plt.savefig(save_file)

    def plot_all_features(self, data: pd.DataFrame, ticker: str, exclude: List[str] = [], save_file: str = "plot.png", start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots all available features, excluding specified ones.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to plot.
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
        All features except those specified in the exclude list are plotted.
        The plot is saved to the specified file.
        """
        features_to_plot = [feature for feature in data.columns if feature not in exclude]
        self.plot(data=data, ticker=ticker, features=features_to_plot, save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_non_volume_features(self, data: pd.DataFrame, ticker: str, save_file: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots all non-volume features.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to plot.
        save_file : str, optional
            The name of the file to save the plot (default is 'plot.png').
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).
        
        Notes
        -----
        All features except 'Volume', 'OBV', and 'ADL' are plotted.
        The plot is saved to the specified file.
        """
        self.plot_all_features(data=data, ticker=ticker, exclude=['Volume', 'OBV', 'ADL'], save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_volume_features(self, data: pd.DataFrame, ticker: str, save_file: str = 'plot.png', start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Plots all volume features.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to plot.
        save_file : str, optional
            The name of the file to save the plot (default is 'plot.png').
        start_date : str, optional
            The start date for the plot in 'YYYY-MM-DD' format (default is None).
        end_date : str, optional
            The end date for the plot in 'YYYY-MM-DD' format (default is None).
        
        Notes
        -----
        Only 'Volume', 'OBV', and 'ADL' features are plotted.
        The plot is saved to the specified file.
        """
        self.plot(data=data, ticker=ticker, features=['Volume', 'OBV', 'ADL'], save_file=save_file, start_date=start_date, end_date=end_date)
