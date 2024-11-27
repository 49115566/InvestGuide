import pandas as pd

class DataFileManager:
    '''
    A class to manage data file operations, including saving and loading data in CSV and Feather formats.

    Methods
    -------
    to_csv(data: pd.DataFrame, file_name: str = 'data.csv') -> None
        Saves the data to a CSV file.
    to_feather(data: pd.DataFrame, file_name: str = 'data.feather') -> None
        Saves the data to a Feather file.
    read_csv(file_name: str = 'data.csv') -> pd.DataFrame
        Loads the data from a CSV file.
    read_feather(file_name: str = 'data.feather') -> pd.DataFrame
        Loads the data from a Feather file.
    '''
    def to_csv(self, data: pd.DataFrame, file_name: str = 'data.csv') -> None:
        """
        Saves the data to a CSV file.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to save.
        file_name : str, optional
            The name of the CSV file to save (default is 'data.csv').
        
        Notes
        -----
        The data is saved to the specified CSV file without the index.
        """
        data.reset_index().to_csv(file_name, index=False)

    def to_feather(self, data: pd.DataFrame, file_name: str = 'data.feather') -> None:
        """
        Saves the data to a Feather file.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to save.
        file_name : str, optional
            The name of the Feather file to save (default is 'data.feather').
        
        Notes
        -----
        The data is saved to the specified Feather file with the index reset.
        """
        data.reset_index().to_feather(file_name)

    def read_csv(self, file_name: str = 'data.csv') -> pd.DataFrame:
        """
        Loads the data from a CSV file.
        
        Parameters
        ----------
        file_name : str, optional
            The name of the CSV file to load (default is 'data.csv').
        
        Notes
        -----
        The data is loaded from the specified CSV file and the index is set to 'Date'.
        """
        data = pd.read_csv(file_name)
        data.set_index('Date', inplace=True)
        return data

    def read_feather(self, file_name: str = 'data.feather') -> pd.DataFrame:
        """
        Loads the data from a Feather file.
        
        Parameters
        ----------
        file_name : str, optional
            The name of the Feather file to load (default is 'data.feather').
        
        Notes
        -----
        The data is loaded from the specified Feather file and the index is set to 'Date'.
        """
        data = pd.read_feather(file_name)
        data.set_index('Date', inplace=True)
        return data