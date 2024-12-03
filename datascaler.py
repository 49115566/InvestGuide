import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer

class DataScaler:
    """
    A class to handle scaling and unscaling of data features.

    Methods
    -------
    scale(data: pd.DataFrame, features: list, method: str = 'standard') -> None
        Scales the specified features in the data using the specified method.
    unscale(data: pd.DataFrame, features: list) -> None
        Unscales the specified features in the data.
    """

    def __init__(self):
        self.scalers = {}

    def scale(self, data: pd.DataFrame, features: list, method: str = 'standard') -> None:
        """
        Scales the specified features in the data using the specified method.

        Parameters
        ----------
        data : pd.DataFrame
            The data containing the features to scale.
        features : list
            A list of feature names to scale.
        method : str, optional
            The scaling method to use (default is 'standard').
            Options are 'standard', 'minmax', 'maxabs', 'robust', or 'quantile'.
        """
        scaler = None
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')

        if scaler:
            data[features] = scaler.fit_transform(data[features])
            self.scalers[method] = scaler

    def unscale(self, data: pd.DataFrame, features: list) -> None:
        """
        Unscales the specified features in the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data containing the features to unscale.
        features : list
            A list of feature names to unscale.
        """
        for method, scaler in self.scalers.items():
            data[features] = scaler.inverse_transform(data[features])