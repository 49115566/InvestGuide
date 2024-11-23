import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from sklearn.impute import SimpleImputer
from finnhub import Client

class MarketData:
    """
    A class used to perform feature engineering on financial data.
    Attributes
    ----------
    ticker : str
        The stock ticker symbol.
    start_date : str
        The start date for the data in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data in 'YYYY-MM-DD' format.
    data : pandas.DataFrame, optional
        The financial data (default is None).
    finnhub_client : finnhub.Client, optional
        The Finnhub client for sentiment analysis (default is None).
    feature_functions : defaultdict
        A dictionary mapping feature names to their corresponding methods.
    Methods
    -------
    fetch_data():
        Loads the financial data for the given ticker and date range.
    fetch_data_with_features(features):
        Loads the financial data and adds the specified features.
    fetch_data_with_all_features(exclude=[]):
        Loads the financial data and adds all features but those specified.
    add_custom_feature(feature, FeatureCol):
        Adds a custom feature to the data.
    add_log_return(LogReturnCol='LogReturn', CloseCol='Close'):
        Adds the logarithmic return feature.
    add_sma(window, SMACol=None, CloseCol='Close'):
        Adds the Simple Moving Average (SMA) feature.
    add_ema(span, EMACol=None, CloseCol='Close'):
        Adds the Exponential Moving Average (EMA) feature.
    add_volatility(VolatilityCol='Volatility', CloseCol='Close'):
        Adds the volatility feature.
    add_momentum(MomentumCol='Momentum', CloseCol='Close'):
        Adds the momentum feature.
    add_rsi(RSICOl='RSI', CloseCol='Close'):
        Adds the Relative Strength Index (RSI) feature.
    add_macd(MACDCol='MACD', CloseCol='Close'):
        Adds the Moving Average Convergence Divergence (MACD) feature.
    add_macd_signal(MACDSignalCol='MACD_Signal', CloseCol='Close'):
        Adds the MACD signal line feature.
    add_bollinger_bands(BollingerUpperCol='BollingerUpper', BollingerLowerCol='BollingerLower', CloseCol='Close'):
        Adds the Bollinger Bands feature.
    add_atr(ATRCol='ATR', HighCol='High', LowCol='Low', CloseCol='Close'):
        Adds the Average True Range (ATR) feature.
    add_obv(OBVCol='OBV', CloseCol='Close', VolumeCol='Volume'):
        Adds the On-Balance Volume (OBV) feature.
    add_vroc(VROCCol='VROC', VolumeCol='Volume', window=14):
        Adds the Volume Rate of Change (VROC) feature.
    add_adl(ADLCol='ADL', HighCol='High', LowCol='Low', CloseCol='Close', VolumeCol='Volume'):
        Adds the Accumulation/Distribution Line (ADL) feature.
    add_cmf(CMFCol='CMF', HighCol='High', LowCol='Low', CloseCol='Close', VolumeCol='Volume', window=20):
        Adds the Chaikin Money Flow (CMF) feature.
    add_stochastic_oscillator(StochasticOscillatorCol='StochasticOscillator', HighCol='High', LowCol='Low', CloseCol='Close'):
        Adds the Stochastic Oscillator feature.
    add_williams_r(WilliamsRCol='WilliamsR', HighCol='High', LowCol='Low', CloseCol='Close'):
        Adds the Williams %R feature.
    add_cci(CCICol='CCI', HighCol='High', LowCol='Low', CloseCol='Close'):
        Adds the Commodity Channel Index (CCI) feature.
    add_ema_crossover(short_window=12, long_window=26, EMACrossoverCol='EMACrossover', CloseCol='Close'):
        Adds the EMA crossover feature.
    add_atr_bands(ATRUpperCol='ATRUpper', ATRLowerCol='ATRLower', ATRCol='ATR', CloseCol='Close'):
        Adds the ATR bands feature.
    add_parabolic_sar(PSARCol='PSAR', HighCol='High', LowCol='Low', CloseCol='Close'):
        Adds the Parabolic SAR feature.
    add_ichimoku_cloud(TenkanCol='Tenkan', KijunCol='Kijun', SenkouSpanACol='SenkouSpanA', SenkouSpanBCol='SenkouSpanB', CloseCol='Close'):
        Adds the Ichimoku Cloud feature.
    add_mfi(MFICol='MFI', HighCol='High', LowCol='Low', CloseCol='Close', VolumeCol='Volume'):
        Adds the Money Flow Index (MFI) feature.
    add_roc(ROCCol='ROC', CloseCol='Close'):
        Adds the Rate of Change (ROC) feature.
    add_pivot_points(PivotCol='Pivot', HighCol='High', LowCol='Low', CloseCol='Close'):
        Adds the Pivot Points feature.
    add_keltner_channels(KeltnerUpperCol='KeltnerUpper', KeltnerLowerCol='KeltnerLower', EMACol='EMA20', ATRCol='ATR'):
        Adds the Keltner Channels feature.
    add_donchian_channels(DonchianUpperCol='DonchianUpper', DonchianLowerCol='DonchianLower', HighCol='High', LowCol='Low'):
        Adds the Donchian Channels feature.
    add_lagged_returns(lags=5, LagCol='Lag_', CloseCol='Close'):
        Adds lagged returns features.
    add_sentiment(SentimentCol='mspr', api_key=None):
        Adds sentiment analysis feature using Finnhub API.
    add_features(features):
        Adds the specified features to the data.
    add_all_features(exclude=[]):
        Adds all available features to the data, excluding specified ones.
    fix_missing_values(strategy='bfill'):
        Fixes missing values in the data using the specified strategy.
    plot(features=[], save_file="plot.png", start_date=None, end_date=None):
        Plots the specified features.
    plot_all_features(exclude=[], save_file="plot.png", start_date=None, end_date=None):
        Plots all available features, excluding specified ones.
    plot_non_volume_features(save_file='plot.png', start_date=None, end_date=None):
        Plots all non-volume features.
    plot_volume_features(save_file='plot.png', start_date=None, end_date=None):
        Plots all volume features, excluding specified ones.
    to_csv(file_name='data.csv'):
        Saves the data to a CSV file.
    to_feather(file_name='data.feather'):
        Saves the data to a Feather file.
    read_csv(file_name='data.csv'):
        Loads the data from a CSV file.
    read_feather(file_name='data.feather'):
        Loads the data from a Feather file.
    """
    def __init__(self, ticker, start_date, end_date, fetch_data=True, data=None):
        """
        Initializes the MarketData class with the given parameters.
        
        Parameters
        ----------
        ticker : str
            The stock ticker symbol.
        start_date : str
            The start date for the data in 'YYYY-MM-DD' format.
        end_date : str
            The end date for the data in 'YYYY-MM-DD' format.
        fetch_data : bool, optional
            Whether to fetch data immediately upon initialization (default is True).
        data : pandas.DataFrame, optional
            The financial data (default is None).
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        if fetch_data:
            self.fetch_data()
        else:
            self.data = data

        self.finnhub_client = None
        self.feature_functions = defaultdict(lambda: None, {
            'LogReturn': self.add_log_return,
            'SMA20': lambda: self.add_sma(20),
            'SMA50': lambda: self.add_sma(50),
            'SMA100': lambda: self.add_sma(100),
            'SMA200': lambda: self.add_sma(200),
            'EMA20': lambda: self.add_ema(20),
            'EMA50': lambda: self.add_ema(50),
            'EMA100': lambda: self.add_ema(100),
            'EMA200': lambda: self.add_ema(200),
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
            'ATRUpper': lambda: self.add_atr_bands(ATRUpperCol='ATRUpper', ATRLowerCol='ATRLower', ATRCol='ATR'),
            'PSAR': self.add_parabolic_sar,
            'IchimokuCloud': self.add_ichimoku_cloud,
            'MFI': self.add_mfi,
            'ROC': self.add_roc,
            'PivotPoints': self.add_pivot_points,
            'KeltnerChannels': self.add_keltner_channels,
            'DonchianChannels': self.add_donchian_channels,
            'Last5Lags': lambda: self.add_lagged_returns(5),
            'MSPR': self.add_sentiment
        })

    def fetch_data(self):
        """
        Fetches the financial data for the given ticker and date range.
        
        Raises
        ------
        ValueError
            If no data is found for the given ticker and date range.
        """
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker} between {self.start_date} and {self.end_date}")
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.droplevel(1)

    def fetch_data_with_features(self, features):
        """
        Fetches the financial data and adds the specified features.
        
        Parameters
        ----------
        features : list
            A list of feature names to add to the data.
        """
        self.fetch_data()
        self.add_features(features)

    def fetch_data_with_all_features(self, exclude=[]):
        """
        Fetches the financial data and adds all features but those specified.
        
        Parameters
        ----------
        exclude : list, optional
            A list of feature names to exclude from the data (default is an empty list).
        """
        self.fetch_data()
        self.add_all_features(exclude=exclude)

    def add_custom_feature(self, feature, FeatureCol):
        """
        Adds a custom feature to the data.
        
        Parameters
        ----------
        feature : function
            A function that takes the data as input and returns the custom feature.
        FeatureCol : str
            The name of the column to store the custom feature.
        """
        try:
            self.data[FeatureCol] = feature(self.data)
        except Exception as e:
            print(f"Error adding custom feature '{FeatureCol}': {e}")

    def add_log_return(self, LogReturnCol='LogReturn', CloseCol='Close'):
        """
        Adds the logarithmic return feature.
        
        Parameters
        ----------
        LogReturnCol : str, optional
            The name of the column to store the logarithmic return (default is 'LogReturn').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        self.data[LogReturnCol] = np.log(self.data[CloseCol] / self.data[CloseCol].shift(1))

    def add_sma(self, window, SMACol=None, CloseCol='Close'):
        """
        Adds the Simple Moving Average (SMA) feature.
        
        Parameters
        ----------
        window : int
            The window size for the SMA.
        SMACol : str, optional
            The name of the column to store the SMA (default is None, which sets it to 'SMA{window}').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        SMACol = SMACol or f'SMA{window}'
        self.data[SMACol] = self.data[CloseCol].rolling(window=window).mean()

    def add_ema(self, span, EMACol=None, CloseCol='Close'):
        """
        Adds the Exponential Moving Average (EMA) feature.
        
        Parameters
        ----------
        span : int
            The span for the EMA.
        EMACol : str, optional
            The name of the column to store the EMA (default is None, which sets it to 'EMA{span}').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        EMACol = EMACol or f'EMA{span}'
        self.data[EMACol] = self.data[CloseCol].ewm(span=span, adjust=False).mean()

    def add_volatility(self, VolatilityCol='Volatility', CloseCol='Close'):
        """
        Adds the volatility feature.
        
        Parameters
        ----------
        VolatilityCol : str, optional
            The name of the column to store the volatility (default is 'Volatility').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        self.data[VolatilityCol] = self.data[CloseCol].rolling(window=20).std()

    def add_momentum(self, MomentumCol='Momentum', CloseCol='Close'):
        """
        Adds the momentum feature.
        
        Parameters
        ----------
        MomentumCol : str, optional
            The name of the column to store the momentum (default is 'Momentum').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        self.data[MomentumCol] = self.data[CloseCol] - self.data[CloseCol].shift(4)

    def add_rsi(self, RSICOl='RSI', CloseCol='Close'):
        """
        Adds the Relative Strength Index (RSI) feature.
        
        Parameters
        ----------
        RSICOl : str, optional
            The name of the column to store the RSI (default is 'RSI').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        delta = self.data[CloseCol].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        self.data[RSICOl] = 100 - (100 / (1 + rs))

    def add_macd(self, MACDCol='MACD', CloseCol='Close'):
        """
        Adds the Moving Average Convergence Divergence (MACD) feature.
        
        Parameters
        ----------
        MACDCol : str, optional
            The name of the column to store the MACD (default is 'MACD').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        ema12 = self.data[CloseCol].ewm(span=12, adjust=False).mean()
        ema26 = self.data[CloseCol].ewm(span=26, adjust=False).mean()
        self.data[MACDCol] = ema12 - ema26

    def add_macd_signal(self, MACDSignalCol='MACD_Signal', CloseCol='Close'):
        """
        Adds the MACD signal line feature.
        
        Parameters
        ----------
        MACDSignalCol : str, optional
            The name of the column to store the MACD signal line (default is 'MACD_Signal').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        ema12 = self.data[CloseCol].ewm(span=12, adjust=False).mean()
        ema26 = self.data[CloseCol].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        self.data[MACDSignalCol] = macd.ewm(span=9, adjust=False).mean()

    def add_bollinger_bands(self, BollingerUpperCol='BollingerUpper', BollingerLowerCol='BollingerLower', CloseCol='Close'):
        """
        Adds the Bollinger Bands feature.
        
        Parameters
        ----------
        BollingerUpperCol : str, optional
            The name of the column to store the upper Bollinger Band (default is 'BollingerUpper').
        BollingerLowerCol : str, optional
            The name of the column to store the lower Bollinger Band (default is 'BollingerLower').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        sma20 = self.data[CloseCol].rolling(window=20).mean()
        std20 = self.data[CloseCol].rolling(window=20).std()
        self.data[BollingerUpperCol] = sma20 + (std20 * 2)
        self.data[BollingerLowerCol] = sma20 - (std20 * 2)

    def add_atr(self, ATRCol='ATR', HighCol='High', LowCol='Low', CloseCol='Close'):
        """
        Adds the Average True Range (ATR) feature.
        
        Parameters
        ----------
        ATRCol : str, optional
            The name of the column to store the ATR (default is 'ATR').
        HighCol : str, optional
            The name of the column containing the high prices (default is 'High').
        LowCol : str, optional
            The name of the column containing the low prices (default is 'Low').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        """
        high_low = self.data[HighCol] - self.data[LowCol]
        high_close = np.abs(self.data[HighCol] - self.data[CloseCol].shift())
        low_close = np.abs(self.data[LowCol] - self.data[CloseCol].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data[ATRCol] = tr.rolling(window=14).mean()

    def add_obv(self, OBVCol='OBV', CloseCol='Close', VolumeCol='Volume'):
        """
        Adds the On-Balance Volume (OBV) feature.
        
        Parameters
        ----------
        OBVCol : str, optional
            The name of the column to store the OBV (default is 'OBV').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        """
        self.data[OBVCol] = (np.sign(self.data[CloseCol].diff()) * self.data[VolumeCol]).fillna(0).cumsum()

    def add_vroc(self, VROCCol='VROC', VolumeCol='Volume', window=14):
        """
        Adds the Volume Rate of Change (VROC) feature.
        
        Parameters
        ----------
        VROCCol : str, optional
            The name of the column to store the VROC (default is 'VROC').
        VolumeCol : str, optional
            The name of the column containing the volume (default is 'Volume').
        window : int, optional
            The window size for the VROC calculation (default is 14).
        """
        self.data[VROCCol] = self.data[VolumeCol].pct_change(periods=window) * 100

    def add_adl(self, ADLCol='ADL', HighCol='High', LowCol='Low', CloseCol='Close', VolumeCol='Volume'):
        """
        Adds the Accumulation/Distribution Line (ADL) feature.
        
        Parameters
        ----------
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
        mfm = ((self.data[CloseCol] - self.data[LowCol]) - (self.data[HighCol] - self.data[CloseCol])) / (self.data[HighCol] - self.data[LowCol])
        mfm = mfm.fillna(0)  # Handle division by zero
        self.data[ADLCol] = (mfm * self.data[VolumeCol]).cumsum()

    def add_cmf(self, CMFCol='CMF', HighCol='High', LowCol='Low', CloseCol='Close', VolumeCol='Volume', window=20):
        """
        Adds the Chaikin Money Flow (CMF) feature.
        
        Parameters
        ----------
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
        mfm = ((self.data[CloseCol] - self.data[LowCol]) - (self.data[HighCol] - self.data[CloseCol])) / (self.data[HighCol] - self.data[LowCol])
        mfm = mfm.fillna(0)  # Handle division by zero
        mfv = mfm * self.data[VolumeCol]
        self.data[CMFCol] = mfv.rolling(window=window).sum() / self.data[VolumeCol].rolling(window=window).sum()

    def add_stochastic_oscillator(self, StochasticOscillatorCol='StochasticOscillator', HighCol='High', LowCol='Low', CloseCol='Close'):
        """
        Adds the Stochastic Oscillator feature.
        
        Parameters
        ----------
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
        low14 = self.data[LowCol].rolling(window=14).min()
        high14 = self.data[HighCol].rolling(window=14).max()
        self.data[StochasticOscillatorCol] = 100 * ((self.data[CloseCol] - low14) / (high14 - low14))

    def add_williams_r(self, WilliamsRCol='WilliamsR', HighCol='High', LowCol='Low', CloseCol='Close'):
        """
        Adds the Williams %R feature.
        
        Parameters
        ----------
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
        high14 = self.data[HighCol].rolling(window=14).max()
        low14 = self.data[LowCol].rolling(window=14).min()
        self.data[WilliamsRCol] = -100 * ((high14 - self.data[CloseCol]) / (high14 - low14))

    def add_cci(self, CCICol='CCI', HighCol='High', LowCol='Low', CloseCol='Close'):
        """
        Adds the Commodity Channel Index (CCI) feature.
        
        Parameters
        ----------
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
        tp = (self.data[HighCol] + self.data[LowCol] + self.data[CloseCol]) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
        self.data[CCICol] = (tp - sma_tp) / (0.015 * mad)

    def add_ema_crossover(self, short_window=12, long_window=26, EMACrossoverCol='EMACrossover', CloseCol='Close'):
        """
        Adds the EMA crossover feature.
        
        Parameters
        ----------
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
        short_ema = self.data[CloseCol].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data[CloseCol].ewm(span=long_window, adjust=False).mean()
        self.data[EMACrossoverCol] = short_ema - long_ema

    def add_atr_bands(self, ATRUpperCol='ATRUpper', ATRLowerCol='ATRLower', ATRCol='ATR', CloseCol='Close'):
        """
        Adds the ATR bands feature.
        
        Parameters
        ----------
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
        self.add_atr(ATRCol=ATRCol)
        self.data[ATRUpperCol] = self.data[CloseCol] + (self.data[ATRCol] * 2)
        self.data[ATRLowerCol] = self.data[CloseCol] - (self.data[ATRCol] * 2)

    def add_parabolic_sar(self, PSARCol='PSAR', HighCol='High', LowCol='Low', CloseCol='Close'):
        """
        Adds the Parabolic SAR feature.
        
        Parameters
        ----------
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
        psar = self.data[CloseCol].iloc[0]
        ep = self.data[HighCol].iloc[0]
        psar_list = [psar]
        for i in range(1, len(self.data)):
            if uptrend:
                psar = psar + af * (ep - psar)
                if self.data[LowCol].iloc[i] < psar:
                    uptrend = False
                    psar = ep
                    ep = self.data[LowCol].iloc[i]
                    af = 0.02
            else:
                psar = psar + af * (ep - psar)
                if self.data[HighCol].iloc[i] > psar:
                    uptrend = True
                    psar = ep
                    ep = self.data[HighCol].iloc[i]
                    af = 0.02
            if uptrend:
                if self.data[HighCol].iloc[i] > ep:
                    ep = self.data[HighCol].iloc[i]
                    af = min(af + 0.02, max_af)
            else:
                if self.data[LowCol].iloc[i] < ep:
                    ep = self.data[LowCol].iloc[i]
                    af = min(af + 0.02, max_af)
            psar_list.append(psar)
        self.data[PSARCol] = psar_list

    def add_ichimoku_cloud(self, TenkanCol='Tenkan', KijunCol='Kijun', SenkouSpanACol='SenkouSpanA', SenkouSpanBCol='SenkouSpanB', CloseCol='Close'):
        """
        Adds the Ichimoku Cloud feature.
        
        Parameters
        ----------
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
        high9 = self.data[CloseCol].rolling(window=9).max()
        low9 = self.data[CloseCol].rolling(window=9).min()
        self.data[TenkanCol] = (high9 + low9) / 2
        high26 = self.data[CloseCol].rolling(window=26).max()
        low26 = self.data[CloseCol].rolling(window=26).min()
        self.data[KijunCol] = (high26 + low26) / 2
        self.data[SenkouSpanACol] = ((self.data[TenkanCol] + self.data[KijunCol]) / 2).shift(26)
        high52 = self.data[CloseCol].rolling(window=52).max()
        low52 = self.data[CloseCol].rolling(window=52).min()
        self.data[SenkouSpanBCol] = ((high52 + low52) / 2).shift(26)

    def add_mfi(self, MFICol='MFI', HighCol='High', LowCol='Low', CloseCol='Close', VolumeCol='Volume'):
        """
        Adds the Money Flow Index (MFI) feature.
        
        Parameters
        ----------
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
        tp = (self.data[HighCol] + self.data[LowCol] + self.data[CloseCol]) / 3
        mf = tp * self.data[VolumeCol]
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum().replace(0, 0.0001)  # Handle division by zero
        self.data[MFICol] = 100 - (100 / (1 + pos_mf / neg_mf))

    def add_roc(self, ROCCol='ROC', CloseCol='Close'):
        """
        Adds the Rate of Change (ROC) feature.
        
        Parameters
        ----------
        ROCCol : str, optional
            The name of the column to store the ROC (default is 'ROC').
        CloseCol : str, optional
            The name of the column containing the closing prices (default is 'Close').
        
        Notes
        -----
        The ROC is calculated as the percentage change in the closing price over a specified period (default is 12).
        """
        self.data[ROCCol] = self.data[CloseCol].pct_change(periods=12) * 100

    def add_pivot_points(self, PivotCol='Pivot', HighCol='High', LowCol='Low', CloseCol='Close'):
        """
        Adds the Pivot Points feature.
        
        Parameters
        ----------
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
        self.data[PivotCol] = (self.data[HighCol] + self.data[LowCol] + self.data[CloseCol]) / 3

    def add_keltner_channels(self, KeltnerUpperCol='KeltnerUpper', KeltnerLowerCol='KeltnerLower', EMACol='EMA20', ATRCol='ATR'):
        """
        Adds the Keltner Channels feature.
        
        Parameters
        ----------
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
        self.add_ema(20, EMACol=EMACol)
        self.add_atr(ATRCol=ATRCol)
        self.data[KeltnerUpperCol] = self.data[EMACol] + (self.data[ATRCol] * 2)
        self.data[KeltnerLowerCol] = self.data[EMACol] - (self.data[ATRCol] * 2)

    def add_donchian_channels(self, DonchianUpperCol='DonchianUpper', DonchianLowerCol='DonchianLower', HighCol='High', LowCol='Low'):
        """
        Adds the Donchian Channels feature.
        
        Parameters
        ----------
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
        self.data[DonchianUpperCol] = self.data[HighCol].rolling(window=20).max()
        self.data[DonchianLowerCol] = self.data[LowCol].rolling(window=20).min()

    def add_lagged_returns(self, lags=5, LagCol='Lag_', CloseCol='Close'):
        """
        Adds lagged returns features.
        
        Parameters
        ----------
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
            self.data[f'{LagCol}{i}'] = self.data[CloseCol].shift(i)

    def add_sentiment(self, SentimentCol='mspr', api_key=None):
        """
        Adds sentiment analysis feature using Finnhub API.
        
        Parameters
        ----------
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
        json_df = pd.DataFrame(self.finnhub_client.stock_insider_sentiment('AAPL', self.start_date, self.end_date)['data'])
        json_df['date'] = json_df.apply(lambda row: pd.Timestamp(datetime(row['year'], row['month'], 1), tz='UTC'), axis=1)
        self.data[SentimentCol] = json_df.set_index(['date']).reindex(self.data.index).ffill()[SentimentCol]

    def add_features(self, features):
        """
        Adds the specified features to the data.
        
        Parameters
        ----------
        features : list
            A list of feature names to add to the data.
        
        Notes
        -----
        The features are added by calling the corresponding methods from the feature_functions dictionary.
        """
        for feature in features:
            self.feature_functions[feature]()

    def add_all_features(self, exclude=[]):
        """
        Adds all available features to the data, excluding specified ones.
        
        Parameters
        ----------
        exclude : list, optional
            A list of feature names to exclude (default is an empty list).
        
        Notes
        -----
        All features are added except those specified in the exclude list.
        """
        features_to_add = [feature for feature in self.feature_functions.keys() if feature not in exclude]
        self.add_features(features_to_add)

    def fix_missing_values(self, strategy='bfill'):
        """
        Fixes missing values in the data using the specified strategy.
        
        Parameters
        ----------
        strategy : str, optional
            The strategy to use for fixing missing values (default is 'bfill').
            Options are 'bfill', 'ffill', 'mean', 'median', 'most_frequent', 'constant', or 'drop'.
        
        Notes
        -----
        The missing values are fixed using the specified strategy. If the strategy is 'bfill' or 'ffill', the missing values are filled using backward or forward fill, respectively. If the strategy is 'mean', 'median', 'most_frequent', or 'constant', the missing values are imputed using the specified strategy. If the strategy is 'drop', the rows with missing values are dropped.
        """
        if strategy == 'bfill' or strategy == 'backfill':
            self.data.bfill(inplace=True)
        elif strategy == 'ffill' or strategy == 'pad':
            self.data.ffill(inplace=True)
        elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
            imputer = SimpleImputer(strategy=strategy)
            self.data[:] = imputer.fit_transform(self.data)
        else:
            self.data.dropna(inplace=True)

    def plot(self, features=[], save_file="plot.png", start_date=None, end_date=None):
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
            data = self.data.loc[start_date:end_date]
        else:
            data = self.data
        data[features].plot(figsize=(14, 7), title=f'{self.ticker} {features}', grid=True)
        plt.savefig(save_file)

    def plot_all_features(self, exclude=[], save_file="plot.png", start_date=None, end_date=None):
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
        All features except those specified in the exclude list are plotted.
        The plot is saved to the specified file.
        """
        features_to_plot = [feature for feature in self.data.columns if feature not in exclude]
        self.plot(features_to_plot, save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_non_volume_features(self, save_file='plot.png', start_date=None, end_date=None):
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
        All features except 'Volume', 'OBV', and 'ADL' are plotted.
        The plot is saved to the specified file.
        """
        self.plot_all_features(exclude=['Volume', 'OBV', 'ADL'], save_file=save_file, start_date=start_date, end_date=end_date)

    def plot_volume_features(self, save_file='plot.png', start_date=None, end_date=None):
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
        Only 'Volume', 'OBV', and 'ADL' features are plotted.
        The plot is saved to the specified file.
        """
        self.plot(features=['Volume', 'OBV', 'ADL'], save_file=save_file, start_date=start_date, end_date=end_date)

    def to_csv(self, file_name='data.csv'):
        """
        Saves the data to a CSV file.
        
        Parameters
        ----------
        file_name : str, optional
            The name of the CSV file to save (default is 'data.csv').
        
        Notes
        -----
        The data is saved to the specified CSV file with the index included.
        """
        self.data.to_csv(file_name, index=True)

    def to_feather(self, file_name='data.feather'):
        """
        Saves the data to a Feather file.
        
        Parameters
        ----------
        file_name : str, optional
            The name of the Feather file to save (default is 'data.feather').
        
        Notes
        -----
        The data is saved to the specified Feather file with the index reset.
        """
        self.data.reset_index().to_feather(file_name)

    def read_csv(self, file_name='data.csv'):
        """
        Loads the data from a CSV file.
        
        Parameters
        ----------
        file_name : str, optional
            The name of the CSV file to load (default is 'data.csv').
        
        Notes
        -----
        The data is loaded from the specified CSV file with the index parsed as dates.
        """
        self.data = pd.read_csv(file_name, index_col=0, parse_dates=True)

    def read_feather(self, file_name='data.feather'):
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
        self.data = pd.read_feather(file_name)
        self.data.set_index('Date', inplace=True)