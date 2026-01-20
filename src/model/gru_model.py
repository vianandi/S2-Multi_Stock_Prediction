import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def gru_forecast(prices, look_back=60, epochs=50):
    """
    Forecast menggunakan GRU (Gated Recurrent Unit).
    Seringkali performanya setara LSTM tapi training lebih cepat.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1))
    
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Perbedaan utama ada di layer ini: GRU menggantikan LSTM
    model = keras.Sequential([
        keras.layers.GRU(50, return_sequences=True, input_shape=(look_back,1)),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(50, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    last_seq = scaled[-look_back:].reshape(1, look_back, 1)
    pred = model.predict(last_seq, verbose=0)
    forecast = scaler.inverse_transform(pred)[0][0]
    
    return float(forecast)