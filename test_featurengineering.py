import unittest
from featureengineering import FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngineering(ticker='AAPL', start_date='2022-01-01', end_date='2022-12-31')

    def test_add_log_return(self):
        self.fe.add_log_return()
        self.assertIn('LogReturn', self.fe.data.columns)
        self.assertFalse(self.fe.data['LogReturn'].isnull().all())

    def test_add_sma(self):
        self.fe.add_sma(20)
        self.assertIn('SMA20', self.fe.data.columns)
        self.assertFalse(self.fe.data['SMA20'].isnull().all())

    def test_add_ema(self):
        self.fe.add_ema(20)
        self.assertIn('EMA20', self.fe.data.columns)
        self.assertFalse(self.fe.data['EMA20'].isnull().all())

    def test_add_volatility(self):
        self.fe.add_volatility()
        self.assertIn('Volatility', self.fe.data.columns)
        self.assertFalse(self.fe.data['Volatility'].isnull().all())

    def test_add_momentum(self):
        self.fe.add_momentum()
        self.assertIn('Momentum', self.fe.data.columns)
        self.assertFalse(self.fe.data['Momentum'].isnull().all())

    def test_add_rsi(self):
        self.fe.add_rsi()
        self.assertIn('RSI', self.fe.data.columns)
        self.assertFalse(self.fe.data['RSI'].isnull().all())

    def test_add_macd(self):
        self.fe.add_macd()
        self.assertIn('MACD', self.fe.data.columns)
        self.assertFalse(self.fe.data['MACD'].isnull().all())

    def test_add_macd_signal(self):
        self.fe.add_macd_signal()
        self.assertIn('MACD_Signal', self.fe.data.columns)
        self.assertFalse(self.fe.data['MACD_Signal'].isnull().all())

    def test_add_bollinger_bands(self):
        self.fe.add_bollinger_bands()
        self.assertIn('BollingerUpper', self.fe.data.columns)
        self.assertIn('BollingerLower', self.fe.data.columns)
        self.assertFalse(self.fe.data['BollingerUpper'].isnull().all())
        self.assertFalse(self.fe.data['BollingerLower'].isnull().all())

    def test_add_atr(self):
        self.fe.add_atr()
        self.assertIn('ATR', self.fe.data.columns)
        self.assertFalse(self.fe.data['ATR'].isnull().all())

    def test_add_obv(self):
        self.fe.add_obv()
        self.assertIn('OBV', self.fe.data.columns)
        self.assertFalse(self.fe.data['OBV'].isnull().all())

    def test_add_stochastic_oscillator(self):
        self.fe.add_stochastic_oscillator()
        self.assertIn('StochasticOscillator', self.fe.data.columns)
        self.assertFalse(self.fe.data['StochasticOscillator'].isnull().all())

    def test_add_williams_r(self):
        self.fe.add_williams_r()
        self.assertIn('WilliamsR', self.fe.data.columns)
        self.assertFalse(self.fe.data['WilliamsR'].isnull().all())

    def test_add_cmf(self):
        self.fe.add_cmf()
        self.assertIn('CMF', self.fe.data.columns)
        self.assertFalse(self.fe.data['CMF'].isnull().all())

    def test_add_cci(self):
        self.fe.add_cci()
        self.assertIn('CCI', self.fe.data.columns)
        self.assertFalse(self.fe.data['CCI'].isnull().all())

    def test_add_ema_crossover(self):
        self.fe.add_ema_crossover()
        self.assertIn('EMACrossover', self.fe.data.columns)
        self.assertFalse(self.fe.data['EMACrossover'].isnull().all())

    def test_add_atr_bands(self):
        self.fe.add_atr_bands()
        self.assertIn('ATRUpper', self.fe.data.columns)
        self.assertIn('ATRLower', self.fe.data.columns)
        self.assertFalse(self.fe.data['ATRUpper'].isnull().all())
        self.assertFalse(self.fe.data['ATRLower'].isnull().all())

    def test_add_parabolic_sar(self):
        self.fe.add_parabolic_sar()
        self.assertIn('PSAR', self.fe.data.columns)
        self.assertFalse(self.fe.data['PSAR'].isnull().all())

    def test_add_ichimoku_cloud(self):
        self.fe.add_ichimoku_cloud()
        self.assertIn('Tenkan', self.fe.data.columns)
        self.assertIn('Kijun', self.fe.data.columns)
        self.assertIn('SenkouSpanA', self.fe.data.columns)
        self.assertIn('SenkouSpanB', self.fe.data.columns)
        self.assertFalse(self.fe.data['Tenkan'].isnull().all())
        self.assertFalse(self.fe.data['Kijun'].isnull().all())
        self.assertFalse(self.fe.data['SenkouSpanA'].isnull().all())
        self.assertFalse(self.fe.data['SenkouSpanB'].isnull().all())

    def test_add_mfi(self):
        self.fe.add_mfi()
        self.assertIn('MFI', self.fe.data.columns)
        self.assertFalse(self.fe.data['MFI'].isnull().all())

    def test_add_roc(self):
        self.fe.add_roc()
        self.assertIn('ROC', self.fe.data.columns)
        self.assertFalse(self.fe.data['ROC'].isnull().all())

    def test_add_pivot_points(self):
        self.fe.add_pivot_points()
        self.assertIn('Pivot', self.fe.data.columns)
        self.assertFalse(self.fe.data['Pivot'].isnull().all())

    def test_add_keltner_channels(self):
        self.fe.add_keltner_channels()
        self.assertIn('KeltnerUpper', self.fe.data.columns)
        self.assertIn('KeltnerLower', self.fe.data.columns)
        self.assertFalse(self.fe.data['KeltnerUpper'].isnull().all())
        self.assertFalse(self.fe.data['KeltnerLower'].isnull().all())

    def test_add_donchian_channels(self):
        self.fe.add_donchian_channels()
        self.assertIn('DonchianUpper', self.fe.data.columns)
        self.assertIn('DonchianLower', self.fe.data.columns)
        self.assertFalse(self.fe.data['DonchianUpper'].isnull().all())
        self.assertFalse(self.fe.data['DonchianLower'].isnull().all())

    def test_add_lagged_returns(self):
        self.fe.add_lagged_returns()
        for i in range(1, 6):
            self.assertIn(f'Lag_{i}', self.fe.data.columns)
            self.assertFalse(self.fe.data[f'Lag_{i}'].isnull().all())

    def test_add_all_features(self):
        self.fe.add_all_features()
        for col in self.fe.data.columns:
            self.assertFalse(self.fe.data[col].isnull().all())

if __name__ == '__main__':
    unittest.main()