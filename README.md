# MarketPredictor

`MarketPredictor` is yet to be completed

# MarketData

`MarketData` is a Python class designed to perform feature engineering on financial data. It provides a comprehensive set of methods to fetch financial data, add various technical indicators, and handle missing values.

## Features

- Fetch financial data using Yahoo Finance
- Add a wide range of technical indicators:
    - Logarithmic Return
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Volatility
    - Momentum
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands
    - Average True Range (ATR)
    - On-Balance Volume (OBV)
    - Volume Rate of Change (VROC)
    - Accumulation/Distribution Line (ADL)
    - Chaikin Money Flow (CMF)
    - Stochastic Oscillator
    - Williams %R
    - Commodity Channel Index (CCI)
    - EMA Crossover
    - ATR Bands
    - Parabolic SAR
    - Ichimoku Cloud
    - Money Flow Index (MFI)
    - Rate of Change (ROC)
    - Pivot Points
    - Keltner Channels
    - Donchian Channels
    - Lagged Returns
    - Sentiment Analysis using Finnhub API
    - And probably more
- Handle missing values with various strategies
- Plot features and save plots as images
- Save and load data in CSV and Feather formats

## Installation

To use the `MarketData` class, you need to install the following dependencies:

```bash
pip install yfinance numpy pandas matplotlib scikit-learn finnhub-python
```

## Usage

### Initialization

```python
from marketdata import MarketData

# Initialize MarketData with a ticker symbol and date range.
# By default, this will grab some data, but you can tell it not to or give it data directly.
md = MarketData(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01')
```

### Fetch Data

```python
# Fetch financial data. The data fetched will be Adj Close, Close, High, Low, Open, and Volume.
md.fetch_data()
```

### Add Features

```python
# Add specific features
md.add_features(features=['LogReturn', 'SMA20', 'RSI'])

# Add all possible features except specified ones
md.add_all_features(exclude=['Volume'])

# Add a column via function
md.add_custom_feature(feature=lambda data: np.log(data['Close'] / data['Close'].shift(2)), feature_column='LogReturnDoubleShift')

# Add a specific feature
md.add_log_return()
```

### Handle Missing Values

```python
# Fix missing values using backfill strategy
# Can use any strategy from the SimpleImputer class, as well as bfill and ffill
md.fix_missing_values(strategy='bfill')
```

### Plot Features

```python
# Plot specific features
md.plot(features=['Close', 'SMA20', 'RSI'])

# Plot all features except specified ones
md.plot_all_features(exclude=['Volume'])
```

### Save and Load Data

```python
# Save data to CSV
md.to_csv('data.csv')

# Load data from CSV
md.read_csv('data.csv')

# Save data to Feather
md.to_feather('data.feather')

# Load data from Feather
md.read_feather('data.feather')
```

## License

There is no official license on this. Feel free to use it however you would like!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact the project maintainer.
