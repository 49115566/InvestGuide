from marketpredictor import MarketPredictor

# Create a MarketPredictor object
predictor = MarketPredictor('AAPL', '2010-01-01', '2020-01-01', [], 30)
predictor.preprocess_data()
predictor.create_default_model()
predictor.train_model()
print(predictor.evaluate_model())
predictor.save_model('model.keras')
print("Tomorrow's prediction:")
print(predictor.predict_tomorrow())