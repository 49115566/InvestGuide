from datavisualizer import DataVisualizer
from marketdata import MarketData
from datafilemanager import DataFileManager

features = MarketData('AAPL', '2019-01-01', '2020-01-01')
features.add_all_features(exclude=['MSPR'])
features.fix_missing_values()

print(features.data)
"""
visualizer = DataVisualizer()
features.plot_non_volume_features(save_file='NonVolume.png')
features.plot_volume_features(save_file='Volume.png')
visualizer.plot_all_features(data=features.data, ticker=features.ticker, save_file='All.png')
"""

print("Saving data as CSV and Feather...")
features.to_csv('AAPL.csv')
print("Data saved as CSV.")
features.to_feather('AAPL.feather')
print("Data saved as Feather.")

print("Loading data from CSV and Feather...")
csv = MarketData('AAPL', '2019-01-01', '2020-01-01')
csv.read_csv('AAPL.csv')
feather = MarketData('AAPL', '2019-01-01', '2020-01-01')
feather.read_feather('AAPL.feather')

print("Comparing data...")
print(csv.data)
print(feather.data)

"""
from datafetcher import DataFetcher
from feature_engineering import FeatureEngineering
import pandas as pd

fetcher = DataFetcher()
ticker = 'AAPL'
start_date = '2019-01-01'
end_date = '2020-01-01'

#data = pd.DataFrame()

data = fetcher.fetch(ticker=ticker, start_date=start_date, end_date=end_date)

modifier = FeatureEngineering()

modifier.add_all_features(data=data, ticker=ticker, start_date=start_date, end_date=end_date)

print(data.info())
"""