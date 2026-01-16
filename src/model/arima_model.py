from pmdarima import auto_arima

def arima_forecast(prices):
    split = int(len(prices)*0.8)
    model = auto_arima(prices[:split], seasonal=False, suppress_warnings=True)
    return model.predict(n_periods=1)[0]