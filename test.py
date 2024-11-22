from featureengineering import FeatureEngineering
import pandas as pd

features = FeatureEngineering('AAPL', '2019-01-01', '2020-01-01')
features.add_all_features(exclude=['MSPR'])
features.fix_missing_values()

features.plot_non_volume_features('NonVolume.png')
features.plot_volume_features('Volume.png')
features.plot_all_features('All.png')

print("Saving data as CSV and Feather...")
features.save_data('AAPL.csv')
print("Data saved as CSV.")
features.save_as_feather('AAPL.feather')
print("Data saved as Feather.")

print("Loading data from CSV and Feather...")
csv = FeatureEngineering('AAPL', '2019-01-01', '2020-01-01')
csv.load_data_from_file('AAPL.csv')
feather = FeatureEngineering('AAPL', '2019-01-01', '2020-01-01')
feather.load_data_from_feather('AAPL.feather')

print("Comparing data...")
print(csv.data.equals(feather.data))