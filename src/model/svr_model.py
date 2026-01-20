import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def svr_forecast(prices, look_back=30):
    """
    Forecast menggunakan Support Vector Regression (SVR).
    Bagus untuk menangkap tren non-linear tanpa kompleksitas Deep Learning.
    """
    # SVR butuh data preparation manual (Sliding Window)
    X, y = [], []
    for i in range(look_back, len(prices)):
        X.append(prices[i-look_back:i])
        y.append(prices[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Scaling penting untuk SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Train model (Kernel RBF paling umum untuk saham)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_scaled, y_scaled)
    
    # Predict next day
    last_window = prices[-look_back:].reshape(1, -1)
    last_window_scaled = scaler_X.transform(last_window)
    
    pred_scaled = model.predict(last_window_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    return float(pred)