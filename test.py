from marketdata import MarketData

features = MarketData('AAPL', '2019-01-01', '2020-01-01')
features.add_all_features(exclude=['MSPR'])
features.fix_missing_values()

features.plot_non_volume_features(save_file='NonVolume.png')
features.plot_volume_features(save_file='Volume.png')
features.plot_all_features(save_file='All.png')
"""
print("Saving data as CSV and Feather...")
features.save_data('AAPL.csv')
print("Data saved as CSV.")
features.save_as_feather('AAPL.feather')
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