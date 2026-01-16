import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

def lstm_forecast(prices, look_back=60, epochs=50):
    """
    Forecast menggunakan LSTM dengan auto-save model
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1))
    
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back,1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    # ✅ TAMBAHAN: Save model
    os.makedirs('output/models', exist_ok=True)
    model_path = f'output/models/lstm_model_{len(prices)}.h5'
    model.save(model_path)
    
    last_seq = scaled[-look_back:].reshape(1, look_back, 1)
    pred = model.predict(last_seq, verbose=0)
    forecast = scaler.inverse_transform(pred)[0][0]
    
    return float(forecast)