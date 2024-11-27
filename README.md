# Market Data Management

This repository contains a set of Python classes designed to manage market data fetching, feature engineering, missing value handling, visualization, and file operations. Below is a brief overview of each class and its functionality.
Additionally, it has a `MarketPredictor` class that is yet to be completed. Eventually, I plan to make it predict the change in closing prices from day to day.

## Installation

To use the `MarketData` class, you need to install the following dependencies:

```bash
pip install yfinance numpy pandas matplotlib scikit-learn finnhub-python
```

## Classes Overview

### 1. MarketData
A class to manage market data fetching, feature engineering, missing value handling, visualization, and file operations.

#### Attributes
- `ticker`: The ticker symbol of the stock.
- `start_date`: The start date for fetching data in 'YYYY-MM-DD' format.
- `end_date`: The end date for fetching data in 'YYYY-MM-DD' format.
- `fetcher`: An instance of `DataFetcher` to fetch the data.
- `data`: The market data.
- `engineer`: An instance of `DataEngineer` for feature engineering.
- `filler`: An instance of `DataFiller` for handling missing values.
- `visualizer`: An instance of `DataVisualizer` for plotting data.
- `filemanager`: An instance of `DataFileManager` for file operations.

#### Methods
- `fetch_data()`: Fetches the market data for the specified ticker and date range.
- `fetch_data_with_features(features)`: Fetches the market data and adds the specified features.
- `fetch_data_with_all_features(exclude)`: Fetches the market data and adds all available features, excluding specified ones.
- `add_features(features)`: Adds the specified features to the data.
- `add_all_features(exclude)`: Adds all available features to the data, excluding specified ones.
- `fix_missing_values(strategy)`: Fixes missing values in the data using the specified strategy.
- `plot(features, save_file, start_date, end_date)`: Plots the specified features.
- `plot_all_features(exclude, save_file, start_date, end_date)`: Plots all available features, excluding specified ones.
- `plot_non_volume_features(save_file, start_date, end_date)`: Plots all non-volume features.
- `plot_volume_features(save_file, start_date, end_date)`: Plots all volume features.
- `to_csv(file_name)`: Saves the data to a CSV file.
- `to_feather(file_name)`: Saves the data to a Feather file.
- `read_csv(file_name)`: Loads the data from a CSV file.
- `read_feather(file_name)`: Loads the data from a Feather file.

### 2. DataFetcher
A class used to fetch historical stock data for a given ticker symbol.

#### Methods
- `fetch(ticker, start_date, end_date)`: Fetches historical stock data for the specified ticker symbol between the given start and end dates.

### 3. DataEngineer
A class used to represent a Data Engineer that adds various financial features to a dataset.

#### Methods
- `add_custom_feature(data, feature, FeatureCol)`: Adds a custom feature to the data.
- `add_log_return(data, LogReturnCol, CloseCol)`: Adds the logarithmic return feature.
- `add_sma(data, window, SMACol, CloseCol)`: Adds the Simple Moving Average (SMA) feature.
- `add_ema(data, span, EMACol, CloseCol)`: Adds the Exponential Moving Average (EMA) feature.
- `add_volatility(data, VolatilityCol, CloseCol)`: Adds the volatility feature.
- `add_momentum(data, MomentumCol, CloseCol)`: Adds the momentum feature.
- `add_rsi(data, RSICOl, CloseCol)`: Adds the Relative Strength Index (RSI) feature.
- `add_macd(data, MACDCol, CloseCol)`: Adds the Moving Average Convergence Divergence (MACD) feature.
- `add_macd_signal(data, MACDSignalCol, CloseCol)`: Adds the MACD signal line feature.
- `add_bollinger_bands(data, BollingerUpperCol, BollingerLowerCol, CloseCol)`: Adds the Bollinger Bands feature.
- `add_atr(data, ATRCol, HighCol, LowCol, CloseCol)`: Adds the Average True Range (ATR) feature.
- `add_obv(data, OBVCol, CloseCol, VolumeCol)`: Adds the On-Balance Volume (OBV) feature.
- `add_vroc(data, VROCCol, VolumeCol, window)`: Adds the Volume Rate of Change (VROC) feature.
- `add_adl(data, ADLCol, HighCol, LowCol, CloseCol, VolumeCol)`: Adds the Accumulation/Distribution Line (ADL) feature.
- `add_cmf(data, CMFCol, HighCol, LowCol, CloseCol, VolumeCol, window)`: Adds the Chaikin Money Flow (CMF) feature.
- `add_stochastic_oscillator(data, StochasticOscillatorCol, HighCol, LowCol, CloseCol)`: Adds the Stochastic Oscillator feature.
- `add_williams_r(data, WilliamsRCol, HighCol, LowCol, CloseCol)`: Adds the Williams %R feature.
- `add_cci(data, CCICol, HighCol, LowCol, CloseCol)`: Adds the Commodity Channel Index (CCI) feature.
- `add_ema_crossover(data, short_window, long_window, EMACrossoverCol, CloseCol)`: Adds the EMA crossover feature.
- `add_atr_bands(data, ATRUpperCol, ATRLowerCol, ATRCol, CloseCol)`: Adds the ATR bands feature.
- `add_parabolic_sar(data, PSARCol, HighCol, LowCol, CloseCol)`: Adds the Parabolic SAR feature.
- `add_ichimoku_cloud(data, TenkanCol, KijunCol, SenkouSpanACol, SenkouSpanBCol, CloseCol)`: Adds the Ichimoku Cloud feature.
- `add_mfi(data, MFICol, HighCol, LowCol, CloseCol, VolumeCol)`: Adds the Money Flow Index (MFI) feature.
- `add_roc(data, ROCCol, CloseCol)`: Adds the Rate of Change (ROC) feature.
- `add_pivot_points(data, PivotCol, HighCol, LowCol, CloseCol)`: Adds the Pivot Points feature.
- `add_keltner_channels(data, KeltnerUpperCol, KeltnerLowerCol, EMACol, ATRCol)`: Adds the Keltner Channels feature.
- `add_donchian_channels(data, DonchianUpperCol, DonchianLowerCol, HighCol, LowCol)`: Adds the Donchian Channels feature.
- `add_lagged_returns(data, lags, LagCol, CloseCol)`: Adds lagged returns features.
- `add_sentiment(data, ticker, start_date, end_date, SentimentCol, api_key)`: Adds sentiment analysis feature using Finnhub API.
- `add_features(data, features, ticker, start_date, end_date)`: Adds the specified features to the data.
- `add_all_features(data, exclude, ticker, start_date, end_date)`: Adds all available features to the data, excluding specified ones.

### 4. DataFiller
A class used to fix missing values in data.

#### Methods
- `fix_missing_values(data, strategy, fill_val)`: Fixes missing values in the data using the specified strategy.

### 5. DataVisualizer
A class used to visualize data by plotting various features.

#### Methods
- `plot(data, ticker, features, save_file, start_date, end_date)`: Plots the specified features.
- `plot_all_features(data, ticker, exclude, save_file, start_date, end_date)`: Plots all available features, excluding specified ones.
- `plot_non_volume_features(data, ticker, save_file, start_date, end_date)`: Plots all non-volume features.
- `plot_volume_features(data, ticker, save_file, start_date, end_date)`: Plots all volume features.

### 6. DataFileManager
A class to manage data file operations, including saving and loading data in CSV and Feather formats.

#### Methods
- `to_csv(data, file_name)`: Saves the data to a CSV file.
- `to_feather(data, file_name)`: Saves the data to a Feather file.
- `read_csv(file_name)`: Loads the data from a CSV file.
- `read_feather(file_name)`: Loads the data from a Feather file.

## Usage
To use these classes, you need to create instances of them and call their methods as required. Below is an example of how to use the `MarketData` class to fetch data, add features, fix missing values, and plot the data.

```python
from marketdata import MarketData

# Initialize MarketData
market_data = MarketData(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01')

# Fetch data
market_data.fetch_data()

# Add features
market_data.add_features(['SMA', 'EMA'])

# Fix missing values
market_data.fix_missing_values(strategy='mean')

# Plot data
market_data.plot(features=['Close', 'SMA', 'EMA'])
```

## License

There is no official license on this. Feel free to use it however you would like!
Please don't try to hold me liable for any of this. Use it at your own risk.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, feel free to contact me!
One way to do this is to connect over LinkedIn and send a message. My profile is in my links.