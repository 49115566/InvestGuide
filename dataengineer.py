from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime
from finnhub import Client
from typing import Callable, List, Optional

class DataEngineer:
    """
    A class used to represent a Data Engineer that adds various financial features to a dataset.

    Attributes
    ----------
    finnhub_client : Optional[Client]
        A client for accessing the Finnhub API.
    feature_functions : defaultdict[str, Callable]
        A dictionary mapping feature names to their corresponding methods.

    Methods
    -------
    add_custom_feature(data: pd.DataFrame, feature: Callable[[pd.DataFrame], pd.Series], FeatureCol: str) -> DataEngineer
        Adds a custom feature to the data.
    add_log_return(data: pd.DataFrame, LogReturnCol: str = 'LogReturn', CloseCol: str = 'Close') -> DataEngineer
        Adds the logarithmic return feature.
    add_sma(data: pd.DataFrame, window: int, SMACol: Optional[str] = None, CloseCol: str = 'Close') -> DataEngineer
        Adds the Simple Moving Average (SMA) feature.
    add_ema(data: pd.DataFrame, span: int, EMACol: Optional[str] = None, CloseCol: str = 'Close') -> DataEngineer
        Adds the Exponential Moving Average (EMA) feature.
    add_volatility(data: pd.DataFrame, VolatilityCol: str = 'Volatility', CloseCol: str = 'Close') -> DataEngineer
        Adds the volatility feature.
    add_momentum(data: pd.DataFrame, MomentumCol: str = 'Momentum', CloseCol: str = 'Close') -> DataEngineer
        Adds the momentum feature.
    add_rsi(data: pd.DataFrame, RSICOl: str = 'RSI', CloseCol: str = 'Close') -> DataEngineer
        Adds the Relative Strength Index (RSI) feature.
    add_macd(data: pd.DataFrame, MACDCol: str = 'MACD', CloseCol: str = 'Close') -> DataEngineer
        Adds the Moving Average Convergence Divergence (MACD) feature.
    add_macd_signal(data: pd.DataFrame, MACDSignalCol: str = 'MACD_Signal', CloseCol: str = 'Close') -> DataEngineer
        Adds the MACD signal line feature.
    add_bollinger_bands(data: pd.DataFrame, BollingerUpperCol: str = 'BollingerUpper', BollingerLowerCol: str = 'BollingerLower', CloseCol: str = 'Close') -> DataEngineer
        Adds the Bollinger Bands feature.
    add_atr(data: pd.DataFrame, ATRCol: str = 'ATR', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> DataEngineer
        Adds the Average True Range (ATR) feature.
    add_obv(data: pd.DataFrame, OBVCol: str = 'OBV', CloseCol: str = 'Close', VolumeCol: str = 'Volume') -> DataEngineer
        Adds the On-Balance Volume (OBV) feature.
    add_vroc(data: pd.DataFrame, VROCCol: str = 'VROC', VolumeCol: str = 'Volume', window: int = 14) -> DataEngineer
        Adds the Volume Rate of Change (VROC) feature.
    add_adl(data: pd.DataFrame, ADLCol: str = 'ADL', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close', VolumeCol: str = 'Volume') -> DataEngineer
        Adds the Accumulation/Distribution Line (ADL) feature.
    add_cmf(data: pd.DataFrame, CMFCol: str = 'CMF', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close', VolumeCol: str = 'Volume', window: int = 20) -> DataEngineer
        Adds the Chaikin Money Flow (CMF) feature.
    add_stochastic_oscillator(data: pd.DataFrame, StochasticOscillatorCol: str = 'StochasticOscillator', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> DataEngineer
        Adds the Stochastic Oscillator feature.
    add_williams_r(data: pd.DataFrame, WilliamsRCol: str = 'WilliamsR', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> DataEngineer
        Adds the Williams %R feature.
    add_cci(data: pd.DataFrame, CCICol: str = 'CCI', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> DataEngineer
        Adds the Commodity Channel Index (CCI) feature.
    add_ema_crossover(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, EMACrossoverCol: str = 'EMACrossover', CloseCol: str = 'Close') -> DataEngineer
        Adds the EMA crossover feature.
    add_atr_bands(data: pd.DataFrame, ATRUpperCol: str = 'ATRUpper', ATRLowerCol: str = 'ATRLower', ATRCol: str = 'ATR', CloseCol: str = 'Close') -> DataEngineer
        Adds the ATR bands feature.
    add_parabolic_sar(data: pd.DataFrame, PSARCol: str = 'PSAR', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> DataEngineer
        Adds the Parabolic SAR feature.
    add_ichimoku_cloud(data: pd.DataFrame, TenkanCol: str = 'Tenkan', KijunCol: str = 'Kijun', SenkouSpanACol: str = 'SenkouSpanA', SenkouSpanBCol: str = 'SenkouSpanB', CloseCol: str = 'Close') -> DataEngineer
        Adds the Ichimoku Cloud feature.
    add_mfi(data: pd.DataFrame, MFICol: str = 'MFI', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close', VolumeCol: str = 'Volume') -> DataEngineer
        Adds the Money Flow Index (MFI) feature.
    add_roc(data: pd.DataFrame, ROCCol: str = 'ROC', CloseCol: str = 'Close') -> DataEngineer
        Adds the Rate of Change (ROC) feature.
    add_pivot_points(data: pd.DataFrame, PivotCol: str = 'Pivot', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> DataEngineer
        Adds the Pivot Points feature.
    add_keltner_channels(data: pd.DataFrame, KeltnerUpperCol: str = 'KeltnerUpper', KeltnerLowerCol: str = 'KeltnerLower', EMACol: str = 'EMA20', ATRCol: str = 'ATR') -> DataEngineer
        Adds the Keltner Channels feature.
    add_donchian_channels(data: pd.DataFrame, DonchianUpperCol: str = 'DonchianUpper', DonchianLowerCol: str = 'DonchianLower', HighCol: str = 'High', LowCol: str = 'Low') -> DataEngineer
        Adds the Donchian Channels feature.
    add_lagged_returns(data: pd.DataFrame, lags: int = 5, LagCol: str = 'Lag_', CloseCol: str = 'Close') -> DataEngineer
        Adds lagged returns features.
    add_sentiment(data: pd.DataFrame, ticker: str, start_date: str, end_date: str, SentimentCol: str = 'mspr', api_key: Optional[str] = None) -> DataEngineer
        Adds sentiment analysis feature using Finnhub API.
    add_features(data: pd.DataFrame, features: List[str], ticker: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> DataEngineer
        Adds the specified features to the data.
    add_all_features(data: pd.DataFrame, exclude: List[str] = [], ticker: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> DataEngineer
        Adds all available features to the data, excluding specified ones.
    """
    def __init__(self):
        self.finnhub_client: Optional[Client] = None
        self.feature_functions: defaultdict[str, Callable] = defaultdict(lambda: None, {
            'LogReturn': self.add_log_return,
            'SMA20': lambda data: self.add_sma(data=data, window=20),
            'SMA50': lambda data: self.add_sma(data=data, window=50),
            'SMA100': lambda data: self.add_sma(data=data, window=100),
            'SMA200': lambda data: self.add_sma(data=data, window=200),
            'EMA20': lambda data: self.add_ema(data=data, span=20),
            'EMA50': lambda data: self.add_ema(data=data, span=50),
            'EMA100': lambda data: self.add_ema(data=data, span=100),
            'EMA200': lambda data: self.add_ema(data=data, span=200),
            'Volatility': self.add_volatility,
            'Momentum': self.add_momentum,
            'RSI': self.add_rsi,
            'MACD': self.add_macd,
            'MACD_Signal': self.add_macd_signal,
            'BollingerBands': self.add_bollinger_bands,
            'ATR': self.add_atr,
            'OBV': self.add_obv,
            'VROC': self.add_vroc,
            'ADL': self.add_adl,
            'StochasticOscillator': self.add_stochastic_oscillator,
            'WilliamsR': self.add_williams_r,
            'CMF': self.add_cmf,
            'CCI': self.add_cci,
            'EMACrossover': self.add_ema_crossover,
            'ATRUpper': lambda data: self.add_atr_bands(data=data, ATRUpperCol='ATRUpper', ATRLowerCol='ATRLower', ATRCol='ATR'),
            'PSAR': self.add_parabolic_sar,
            'IchimokuCloud': self.add_ichimoku_cloud,
            'MFI': self.add_mfi,
            'ROC': self.add_roc,
            'PivotPoints': self.add_pivot_points,
            'KeltnerChannels': self.add_keltner_channels,
            'DonchianChannels': self.add_donchian_channels,
            'Last5Lags': lambda data: self.add_lagged_returns(data=data, lags=5),
            'MSPR': self.add_sentiment
        })

    def add_custom_feature(self, data: pd.DataFrame, feature: Callable[[pd.DataFrame], pd.Series], FeatureCol: str) -> 'DataEngineer':
        """
        Adds a custom feature to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        feature : function
            A function that takes the data as input and returns the custom feature.
        FeatureCol : str
            The name of the column to store the custom feature.
        """
        try:
            data[FeatureCol] = feature(data)
        except Exception as e:
            print(f"Error adding custom feature '{FeatureCol}': {e}")
        return self

    def add_log_return(self, data: pd.DataFrame, LogReturnCol: str = 'LogReturn', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the logarithmic return feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        LogReturnCol : str, optional
            The name of the column to store the logarithmic return (default is 'LogReturn').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        data[LogReturnCol] = np.log(data[CloseCol] / data[CloseCol].shift(1))
        return self

    def add_sma(self, data: pd.DataFrame, window: int, SMACol: Optional[str] = None, CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Simple Moving Average (SMA) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        window : int
            The window size for the SMA.
        SMACol : str, optional
            The name of the column to store the SMA (default is None, which sets it to 'SMA{window}').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        SMACol = SMACol or f'SMA{window}'
        data[SMACol] = data[CloseCol].rolling(window=window).mean()
        return self

    def add_ema(self, data: pd.DataFrame, span: int, EMACol: Optional[str] = None, CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Exponential Moving Average (EMA) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        span : int
            The span for the EMA.
        EMACol : str, optional
            The name of the column to store the EMA (default is None, which sets it to 'EMA{span}').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        EMACol = EMACol or f'EMA{span}'
        data[EMACol] = data[CloseCol].ewm(span=span, adjust=False).mean()
        return self

    def add_volatility(self, data: pd.DataFrame, VolatilityCol: str = 'Volatility', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the volatility feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        VolatilityCol : str, optional
            The name of the column to store the volatility (default is 'Volatility').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        data[VolatilityCol] = data[CloseCol].rolling(window=20).std()
        return self

    def add_momentum(self, data: pd.DataFrame, MomentumCol: str = 'Momentum', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the momentum feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        MomentumCol : str, optional
            The name of the column to store the momentum (default is 'Momentum').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        data[MomentumCol] = data[CloseCol] - data[CloseCol].shift(4)
        return self

    def add_rsi(self, data: pd.DataFrame, RSICOl: str = 'RSI', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Relative Strength Index (RSI) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        RSICOl : str, optional
            The name of the column to store the RSI (default is 'RSI').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        delta = data[CloseCol].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        data[RSICOl] = 100 - (100 / (1 + rs))
        return self

    def add_macd(self, data: pd.DataFrame, MACDCol: str = 'MACD', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Moving Average Convergence Divergence (MACD) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        MACDCol : str, optional
            The name of the column to store the MACD (default is 'MACD').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        ema12 = data[CloseCol].ewm(span=12, adjust=False).mean()
        ema26 = data[CloseCol].ewm(span=26, adjust=False).mean()
        data[MACDCol] = ema12 - ema26
        return self

    def add_macd_signal(self, data: pd.DataFrame, MACDSignalCol: str = 'MACD_Signal', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the MACD signal line feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        MACDSignalCol : str, optional
            The name of the column to store the MACD signal line (default is 'MACD_Signal').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        ema12 = data[CloseCol].ewm(span=12, adjust=False).mean()
        ema26 = data[CloseCol].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        data[MACDSignalCol] = macd.ewm(span=9, adjust=False).mean()
        return self

    def add_bollinger_bands(self, data: pd.DataFrame, BollingerUpperCol: str = 'BollingerUpper', BollingerLowerCol: str = 'BollingerLower', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Bollinger Bands feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        BollingerUpperCol : str, optional
            The name of the column to store the upper Bollinger Band (default is 'BollingerUpper').
        BollingerLowerCol : str, optional
            The name of the column to store the lower Bollinger Band (default is 'BollingerLower').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        sma20 = data[CloseCol].rolling(window=20).mean()
        std20 = data[CloseCol].rolling(window=20).std()
        data[BollingerUpperCol] = sma20 + (std20 * 2)
        data[BollingerLowerCol] = sma20 - (std20 * 2)
        return self

    def add_atr(self, data: pd.DataFrame, ATRCol: str = 'ATR', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Average True Range (ATR) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        ATRCol : str, optional
            The name of the column to store the ATR (default is 'ATR').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        high_low = data[HighCol] - data[LowCol]
        high_close = np.abs(data[HighCol] - data[CloseCol].shift())
        low_close = np.abs(data[LowCol] - data[CloseCol].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        data[ATRCol] = tr.rolling(window=14).mean()
        return self

    def add_obv(self, data: pd.DataFrame, OBVCol: str = 'OBV', CloseCol: str = 'Close', VolumeCol: str = 'Volume') -> 'DataEngineer':
        """
        Adds the On-Balance Volume (OBV) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        OBVCol : str, optional
            The name of the column to store the OBV (default is 'OBV').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        """
        data[OBVCol] = (np.sign(data[CloseCol].diff()) * data[VolumeCol]).fillna(0).cumsum()
        return self

    def add_vroc(self, data: pd.DataFrame, VROCCol: str = 'VROC', VolumeCol: str = 'Volume', window: int = 14) -> 'DataEngineer':
        """
        Adds the Volume Rate of Change (VROC) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        VROCCol : str, optional
            The name of the column to store the VROC (default is 'VROC').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        window : int, optional
            The window size for the VROC calculation (default is 14).
        """
        data[VROCCol] = data[VolumeCol].pct_change(periods=window) * 100
        return self

    def add_adl(self, data: pd.DataFrame, ADLCol: str = 'ADL', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close', VolumeCol: str = 'Volume') -> 'DataEngineer':
        """
        Adds the Accumulation/Distribution Line (ADL) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        ADLCol : str, optional
            The name of the column to store the ADL (default is 'ADL').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        """
        mfm = ((data[CloseCol] - data[LowCol]) - (data[HighCol] - data[CloseCol])) / (data[HighCol] - data[LowCol])
        mfm = mfm.fillna(0)  # Handle division by zero
        data[ADLCol] = (mfm * data[VolumeCol]).cumsum()
        return self

    def add_cmf(self, data: pd.DataFrame, CMFCol: str = 'CMF', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close', VolumeCol: str = 'Volume', window: int = 20) -> 'DataEngineer':
        """
        Adds the Chaikin Money Flow (CMF) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        CMFCol : str, optional
            The name of the column to store the CMF (default is 'CMF').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        window : int, optional
            The window size for the CMF calculation (default is 20).
        
        Notes
        -----
        The CMF is calculated by summing the Money Flow Volume (MFV) over a specified window and dividing it by the sum of the volume over the same window.
        """
        mfm = ((data[CloseCol] - data[LowCol]) - (data[HighCol] - data[CloseCol])) / (data[HighCol] - data[LowCol])
        mfm = mfm.fillna(0)  # Handle division by zero
        mfv = mfm * data[VolumeCol]
        data[CMFCol] = mfv.rolling(window=window).sum() / data[VolumeCol].rolling(window=window).sum()
        return self

    def add_stochastic_oscillator(self, data: pd.DataFrame, StochasticOscillatorCol: str = 'StochasticOscillator', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Stochastic Oscillator feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        StochasticOscillatorCol : str, optional
            The name of the column to store the Stochastic Oscillator (default is 'StochasticOscillator').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The Stochastic Oscillator is calculated as the percentage of the current closing price relative to the range of prices over a specified period (default is 14).
        """
        low14 = data[LowCol].rolling(window=14).min()
        high14 = data[HighCol].rolling(window=14).max()
        data[StochasticOscillatorCol] = 100 * ((data[CloseCol] - low14) / (high14 - low14))
        return self

    def add_williams_r(self, data: pd.DataFrame, WilliamsRCol: str = 'WilliamsR', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Williams %R feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        WilliamsRCol : str, optional
            The name of the column to store the Williams %R (default is 'WilliamsR').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The Williams %R is calculated as the percentage of the difference between the highest high and the current closing price relative to the range of prices over a specified period (default is 14).
        """
        high14 = data[HighCol].rolling(window=14).max()
        low14 = data[LowCol].rolling(window=14).min()
        data[WilliamsRCol] = -100 * ((high14 - data[CloseCol]) / (high14 - low14))
        return self

    def add_cci(self, data: pd.DataFrame, CCICol: str = 'CCI', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Commodity Channel Index (CCI) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        CCICol : str, optional
            The name of the column to store the CCI (default is 'CCI').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The CCI is calculated as the difference between the typical price and its moving average, divided by the mean absolute deviation of the typical price over a specified period (default is 20).
        """
        tp = (data[HighCol] + data[LowCol] + data[CloseCol]) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
        data[CCICol] = (tp - sma_tp) / (0.015 * mad)
        return self

    def add_ema_crossover(self, data: pd.DataFrame, short_window: int = 12, long_window: int = 26, EMACrossoverCol: str = 'EMACrossover', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the EMA crossover feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        short_window : int, optional
            The short window size for the EMA (default is 12).
        long_window : int, optional
            The long window size for the EMA (default is 26).
        EMACrossoverCol : str, optional
            The name of the column to store the EMA crossover (default is 'EMACrossover').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The EMA crossover is calculated as the difference between the short-term EMA and the long-term EMA.
        """
        short_ema = data[CloseCol].ewm(span=short_window, adjust=False).mean()
        long_ema = data[CloseCol].ewm(span=long_window, adjust=False).mean()
        data[EMACrossoverCol] = short_ema - long_ema
        return self

    def add_atr_bands(self, data: pd.DataFrame, ATRUpperCol: str = 'ATRUpper', ATRLowerCol: str = 'ATRLower', ATRCol: str = 'ATR', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the ATR bands feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        ATRUpperCol : str, optional
            The name of the column to store the upper ATR band (default is 'ATRUpper').
        ATRLowerCol : str, optional
            The name of the column to store the lower ATR band (default is 'ATRLower').
        ATRCol : str, optional
            The name of the column containing the ATR (default is 'ATR').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The ATR bands are calculated as the closing price plus/minus twice the ATR.
        """
        self.add_atr(data=data, ATRCol=ATRCol)
        data[ATRUpperCol] = data[CloseCol] + (data[ATRCol] * 2)
        data[ATRLowerCol] = data[CloseCol] - (data[ATRCol] * 2)
        return self

    def add_parabolic_sar(self, data: pd.DataFrame, PSARCol: str = 'PSAR', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Parabolic SAR feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        PSARCol : str, optional
            The name of the column to store the Parabolic SAR (default is 'PSAR').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The Parabolic SAR is calculated using the high and low prices, with an acceleration factor that increases when the trend continues in the same direction.
        """
        af = 0.02
        max_af = 0.2
        uptrend = True
        psar = data[CloseCol].iloc[0]
        ep = data[HighCol].iloc[0]
        psar_list = [psar]
        for i in range(1, len(data)):
            if uptrend:
                psar = psar + af * (ep - psar)
                if data[LowCol].iloc[i] < psar:
                    uptrend = False
                    psar = ep
                    ep = data[LowCol].iloc[i]
                    af = 0.02
            else:
                psar = psar + af * (ep - psar)
                if data[HighCol].iloc[i] > psar:
                    uptrend = True
                    psar = ep
                    ep = data[HighCol].iloc[i]
                    af = 0.02
            if uptrend:
                if data[HighCol].iloc[i] > ep:
                    ep = data[HighCol].iloc[i]
                    af = min(af + 0.02, max_af)
            else:
                if data[LowCol].iloc[i] < ep:
                    ep = data[LowCol].iloc[i]
                    af = min(af + 0.02, max_af)
            psar_list.append(psar)
        data[PSARCol] = psar_list
        return self

    def add_ichimoku_cloud(self, data: pd.DataFrame, TenkanCol: str = 'Tenkan', KijunCol: str = 'Kijun', SenkouSpanACol: str = 'SenkouSpanA', SenkouSpanBCol: str = 'SenkouSpanB', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Ichimoku Cloud feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        TenkanCol : str, optional
            The name of the column to store the Tenkan-sen (default is 'Tenkan').
        KijunCol : str, optional
            The name of the column to store the Kijun-sen (default is 'Kijun').
        SenkouSpanACol : str, optional
            The name of the column to store the Senkou Span A (default is 'SenkouSpanA').
        SenkouSpanBCol : str, optional
            The name of the column to store the Senkou Span B (default is 'SenkouSpanB').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The Ichimoku Cloud is calculated using the high and low prices over different periods to create various lines (Tenkan-sen, Kijun-sen, Senkou Span A, and Senkou Span B).
        """
        high9 = data[CloseCol].rolling(window=9).max()
        low9 = data[CloseCol].rolling(window=9).min()
        data[TenkanCol] = (high9 + low9) / 2
        high26 = data[CloseCol].rolling(window=26).max()
        low26 = data[CloseCol].rolling(window=26).min()
        data[KijunCol] = (high26 + low26) / 2
        data[SenkouSpanACol] = ((data[TenkanCol] + data[KijunCol]) / 2).shift(26)
        high52 = data[CloseCol].rolling(window=52).max()
        low52 = data[CloseCol].rolling(window=52).min()
        data[SenkouSpanBCol] = ((high52 + low52) / 2).shift(26)
        return self

    def add_mfi(self, data: pd.DataFrame, MFICol: str = 'MFI', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close', VolumeCol: str = 'Volume') -> 'DataEngineer':
        """
        Adds the Money Flow Index (MFI) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        MFICol : str, optional
            The name of the column to store the MFI (default is 'MFI').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        
        Notes
        -----
        The MFI is calculated using the typical price and volume to determine the money flow, which is then used to calculate the MFI over a specified period (default is 14).
        """
        tp = (data[HighCol] + data[LowCol] + data[CloseCol]) / 3
        mf = tp * data[VolumeCol]
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum().replace(0, 0.0001)  # Handle division by zero
        data[MFICol] = 100 - (100 / (1 + pos_mf / neg_mf))
        return self

    def add_roc(self, data: pd.DataFrame, ROCCol: str = 'ROC', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Rate of Change (ROC) feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        ROCCol : str, optional
            The name of the column to store the ROC (default is 'ROC').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The ROC is calculated as the percentage change in the closing price over a specified period (default is 12).
        """
        data[ROCCol] = data[CloseCol].pct_change(periods=12) * 100
        return self

    def add_pivot_points(self, data: pd.DataFrame, PivotCol: str = 'Pivot', HighCol: str = 'High', LowCol: str = 'Low', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds the Pivot Points feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        PivotCol : str, optional
            The name of the column to store the Pivot Points (default is 'Pivot').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The Pivot Points are calculated as the average of the high, low, and closing prices.
        """
        data[PivotCol] = (data[HighCol] + data[LowCol] + data[CloseCol]) / 3
        return self

    def add_keltner_channels(self, data: pd.DataFrame, KeltnerUpperCol: str = 'KeltnerUpper', KeltnerLowerCol: str = 'KeltnerLower', EMACol: str = 'EMA20', ATRCol: str = 'ATR') -> 'DataEngineer':
        """
        Adds the Keltner Channels feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        KeltnerUpperCol : str, optional
            The name of the column to store the upper Keltner Channel (default is 'KeltnerUpper').
        KeltnerLowerCol : str, optional
            The name of the column to store the lower Keltner Channel (default is 'KeltnerLower').
        EMACol : str, optional
            The name of the column containing the EMA (default is 'EMA20').
        ATRCol : str, optional
            The name of the column containing the ATR (default is 'ATR').
        
        Notes
        -----
        The Keltner Channels are calculated using the EMA and ATR. The upper channel is the EMA plus twice the ATR, and the lower channel is the EMA minus twice the ATR.
        """
        self.add_ema(data=data, span=20, EMACol=EMACol)
        self.add_atr(data=data, ATRCol=ATRCol)
        data[KeltnerUpperCol] = data[EMACol] + (data[ATRCol] * 2)
        data[KeltnerLowerCol] = data[EMACol] - (data[ATRCol] * 2)
        return self

    def add_donchian_channels(self, data: pd.DataFrame, DonchianUpperCol: str = 'DonchianUpper', DonchianLowerCol: str = 'DonchianLower', HighCol: str = 'High', LowCol: str = 'Low') -> 'DataEngineer':
        """
        Adds the Donchian Channels feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        DonchianUpperCol : str, optional
            The name of the column to store the upper Donchian Channel (default is 'DonchianUpper').
        DonchianLowerCol : str, optional
            The name of the column to store the lower Donchian Channel (default is 'DonchianLower').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        
        Notes
        -----
        The Donchian Channels are calculated as the highest high and the lowest low over a specified period (default is 20).
        """
        data[DonchianUpperCol] = data[HighCol].rolling(window=20).max()
        data[DonchianLowerCol] = data[LowCol].rolling(window=20).min()
        return self

    def add_lagged_returns(self, data: pd.DataFrame, lags: int = 5, LagCol: str = 'Lag_', CloseCol: str = 'Close') -> 'DataEngineer':
        """
        Adds lagged returns features.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        lags : int, optional
            The number of lagged returns to add (default is 5).
        LagCol : str, optional
            The prefix for the lagged return columns (default is 'Lag_').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        Lagged returns are calculated as the closing price shifted by a specified number of periods.
        """
        for i in range(1, lags + 1):
            data[f'{LagCol}{i}'] = data[CloseCol].shift(i)
        return self

    def add_sentiment(self, data: pd.DataFrame, ticker: str, start_date: str, end_date: str, SentimentCol: str = 'mspr', api_key: Optional[str] = None) -> 'DataEngineer':
        """
        Adds sentiment analysis feature using Finnhub API.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the feature will be added.
        ticker : str
            The ticker symbol for the stock.
        start_date : str
            The start date for the sentiment analysis.
        end_date : str
            The end date for the sentiment analysis.
        SentimentCol : str, optional
            The name of the column to store the sentiment score (default is 'mspr').
        api_key : str, optional
            The API key for Finnhub (default is None).
        
        Notes
        -----
        The sentiment score is fetched from Finnhub API and added to the data.
        """
        if self.finnhub_client is None:
            if api_key is None:
                api_key = input("Enter Finnhub API key for sentiment analysis: ")
            self.finnhub_client = Client(api_key=api_key)
        json_df = pd.DataFrame(self.finnhub_client.stock_insider_sentiment(ticker, start_date, end_date)['data'])
        json_df['date'] = json_df.apply(lambda row: pd.Timestamp(datetime(row['year'], row['month'], 1), tz='UTC'), axis=1)
        data[SentimentCol] = json_df.set_index(['date']).reindex(data.index).ffill()[SentimentCol]
        return self

    def add_features(self, data: pd.DataFrame, features: List[str], ticker: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> 'DataEngineer':
        """
        Adds the specified features to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the features will be added.
        features : list
            A list of feature names to add to the data.
        ticker : str, optional
            The ticker symbol for the stock (default is None).
        start_date : str, optional
            The start date for the sentiment analysis (default is None).
        end_date : str, optional
            The end date for the sentiment analysis (default is None).
        
        Notes
        -----
        The features are added by calling the corresponding methods from the feature_functions dictionary.
        """
        for feature in features:
            if feature != "MSPR":
                self.feature_functions[feature](data)
            else:
                self.feature_functions[feature](data, ticker, start_date, end_date)
        return self

    def add_all_features(self, data: pd.DataFrame, exclude: List[str] = [], ticker: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> 'DataEngineer':
        """
        Adds all available features to the data, excluding specified ones.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to which the features will be added.
        exclude : list, optional
            A list of feature names to exclude (default is an empty list).
        ticker : str, optional
            The ticker symbol for the stock (default is None).
        start_date : str, optional
            The start date for the sentiment analysis (default is None).
        end_date : str, optional
            The end date for the sentiment analysis (default is None).
        
        Notes
        -----
        All features are added except those specified in the exclude list.
        """
        features_to_add = [feature for feature in self.feature_functions.keys() if feature not in exclude]
        self.add_features(data, features_to_add, ticker, start_date, end_date)
        return self