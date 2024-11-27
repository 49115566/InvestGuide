import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Any, Optional

class DataFiller:
    """
    A class used to fix missing values in data.

    Methods
    -------
    fix_missing_values(data: pd.DataFrame, strategy: str = 'bfill', fill_val: Optional[Any] = None) -> None
        Fixes missing values in the data using the specified strategy.
    """
    def fix_missing_values(self, data: pd.DataFrame, strategy: str = 'bfill', fill_val: Optional[Any] = None) -> None:
        """
        Fixes missing values in the data using the specified strategy.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to fix.
        strategy : str, optional
            The strategy to use for fixing missing values (default is 'bfill').
            Options are 'bfill', 'ffill', 'mean', 'median', 'most_frequent', 'constant', or 'drop'.
        fill_val : any, optional
            The value to use for filling missing values if provided.
        
        Notes
        -----
        The missing values are fixed using the specified strategy. If the strategy is 'bfill' or 'ffill', the missing values are filled using backward or forward fill, respectively. If the strategy is 'mean', 'median', 'most_frequent', or 'constant', the missing values are imputed using the specified strategy. If the strategy is 'drop', the rows with missing values are dropped.
        Priority goes to a fill value if provided.
        """
        if fill_val is not None:
            data.fillna(fill_val, inplace=True)
        elif strategy == 'bfill' or strategy == 'backfill':
            data.bfill(inplace=True)
        elif strategy == 'ffill' or strategy == 'pad':
            data.ffill(inplace=True)
        elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
            imputer = SimpleImputer(strategy=strategy)
            data[:] = imputer.fit_transform(data)
        else:
            data.dropna(inplace=True)