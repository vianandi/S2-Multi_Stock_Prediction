"""
STACKED ENSEMBLE MODEL
======================
True ensemble dengan meta-learning approach.
Mirip kombinasi model NLP (IndoBERT-Lite + IndoBERTweet).

Meta-model belajar cara optimal menggabungkan prediksi ARIMA + LSTM
berdasarkan historical performance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor

try:
    from sklearn.ensemble import RandomForestRegressor
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False


class StackedEnsemble:
    """
    Meta-learning ensemble yang menggabungkan ARIMA + LSTM
    
    Cara kerja:
    1. Gunakan historical data untuk train meta-model
    2. Meta-model belajar bobot optimal untuk setiap base model
    3. Untuk prediksi baru, gabungkan output ARIMA + LSTM menggunakan meta-model
    
    Parameters:
    -----------
    meta_model : str
        Tipe meta-model: 'xgboost', 'random_forest', 'gradient_boosting'
    test_size : float
        Proporsi data untuk validation (default: 0.2)
    """
    
    def __init__(self, meta_model='xgboost', test_size=0.2, random_state=42):
        self.meta_model_type = meta_model
        self.meta_model = None
        self.test_size = test_size
        self.random_state = random_state
        self.is_trained = False
        self.feature_importance = None
        self.validation_score = None
        
    def prepare_training_data(self, prices, arima_predictions, lstm_predictions):
        """
        Prepare data untuk training meta-model
        
        Parameters:
        -----------
        prices : array
            Actual prices (target values)
        arima_predictions : array
            Historical ARIMA predictions
        lstm_predictions : array
            Historical LSTM predictions
            
        Returns:
        --------
        X : array
            Features (stacked predictions)
        y : array
            Target (actual prices)
        """
        # Stack predictions sebagai features
        X = np.column_stack([arima_predictions, lstm_predictions])
        y = prices
        
        return X, y
    
    def train(self, prices, arima_predictions, lstm_predictions, verbose=True):
        """
        Train meta-model menggunakan historical data
        
        Parameters:
        -----------
        prices : array
            Actual prices untuk training
        arima_predictions : array
            Historical ARIMA predictions
        lstm_predictions : array
            Historical LSTM predictions
        verbose : bool
            Print training progress
        """
        # Prepare data
        X, y = self.prepare_training_data(prices, arima_predictions, lstm_predictions)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Initialize meta-model
        if self.meta_model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.meta_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            )
        elif self.meta_model_type == 'random_forest' and RF_AVAILABLE:
            self.meta_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_state
            )
        elif self.meta_model_type == 'gradient_boosting':
            self.meta_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        else:
            # Fallback to Gradient Boosting
            self.meta_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
            self.meta_model_type = 'gradient_boosting'
        
        # Train meta-model
        self.meta_model.fit(X_train, y_train)
        
        # Calculate validation metrics
        y_pred = self.meta_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        self.validation_score = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        # Get feature importance (weights)
        if hasattr(self.meta_model, 'feature_importances_'):
            importance = self.meta_model.feature_importances_
            self.feature_importance = {
                'arima_weight': importance[0],
                'lstm_weight': importance[1]
            }
        else:
            # For models without feature_importances_, use coefficients or equal weights
            self.feature_importance = {
                'arima_weight': 0.5,
                'lstm_weight': 0.5
            }
        
        self.is_trained = True
        
        if verbose:
            print(f"✅ Stacked Ensemble trained successfully!")
            print(f"   Meta-model: {self.meta_model_type}")
            print(f"   Validation R²: {r2:.4f}")
            print(f"   Validation MAE: {mae:.2f}")
            print(f"   ARIMA weight: {self.feature_importance['arima_weight']:.2%}")
            print(f"   LSTM weight: {self.feature_importance['lstm_weight']:.2%}")
    
    def predict(self, arima_pred, lstm_pred):
        """
        Predict menggunakan trained meta-model
        
        Parameters:
        -----------
        arima_pred : float
            ARIMA prediction
        lstm_pred : float
            LSTM prediction
            
        Returns:
        --------
        prediction : float
            Final ensemble prediction
        """
        if not self.is_trained:
            # Fallback to simple average if not trained
            return (arima_pred + lstm_pred) / 2
        
        # Prepare features
        X = np.array([[arima_pred, lstm_pred]])
        
        # Predict
        try:
            prediction = self.meta_model.predict(X)[0]
            
            # Sanity check: prediction should be within reasonable range of inputs
            min_input = min(arima_pred, lstm_pred)
            max_input = max(arima_pred, lstm_pred)
            
            # If prediction is way off, fallback to weighted average
            if prediction < min_input * 0.5 or prediction > max_input * 2.0:
                # Use learned weights if available
                if self.feature_importance:
                    w_arima = self.feature_importance['arima_weight']
                    w_lstm = self.feature_importance['lstm_weight']
                    prediction = w_arima * arima_pred + w_lstm * lstm_pred
                else:
                    prediction = (arima_pred + lstm_pred) / 2
        except:
            # Fallback to simple average on any error
            prediction = (arima_pred + lstm_pred) / 2
        
        return prediction
    
    def get_weights(self):
        """
        Get learned weights for ARIMA and LSTM
        
        Returns:
        --------
        weights : dict
            Dictionary dengan arima_weight dan lstm_weight
        """
        if self.feature_importance:
            return self.feature_importance
        else:
            return {'arima_weight': 0.5, 'lstm_weight': 0.5}
    
    def get_validation_metrics(self):
        """
        Get validation metrics
        
        Returns:
        --------
        metrics : dict
            Dictionary dengan MAE, RMSE, R2
        """
        return self.validation_score if self.validation_score else {}


def train_stacked_ensemble_from_history(prices, lookback=60, forecast_horizon=1):
    """
    Train stacked ensemble menggunakan walk-forward validation
    pada historical data
    
    Parameters:
    -----------
    prices : array
        Historical prices
    lookback : int
        Window size untuk training base models
    forecast_horizon : int
        Number of steps ahead to predict
        
    Returns:
    --------
    ensemble : StackedEnsemble
        Trained stacked ensemble model
    arima_preds : list
        Historical ARIMA predictions
    lstm_preds : list
        Historical LSTM predictions
    actuals : list
        Actual prices
    """
    from src.model.arima_model import arima_forecast
    from src.model.lstm_model import lstm_forecast
    
    if len(prices) < lookback + 20:
        raise ValueError(f"Not enough data. Need at least {lookback + 20} points, got {len(prices)}")
    
    arima_preds = []
    lstm_preds = []
    actuals = []
    
    # Walk-forward validation
    print("🔄 Generating training data using walk-forward validation...")
    for i in range(lookback, len(prices) - forecast_horizon):
        train_data = prices[:i]
        actual = prices[i + forecast_horizon - 1]
        
        try:
            # Get predictions from base models
            arima_pred = arima_forecast(train_data)
            lstm_pred = lstm_forecast(train_data)
            
            arima_preds.append(arima_pred)
            lstm_preds.append(lstm_pred)
            actuals.append(actual)
        except:
            continue
    
    # Train stacked ensemble
    ensemble = StackedEnsemble(meta_model='xgboost')
    ensemble.train(
        np.array(actuals),
        np.array(arima_preds),
        np.array(lstm_preds),
        verbose=True
    )
    
    return ensemble, arima_preds, lstm_preds, actuals


# Quick test function
if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    # Simulate prices
    prices = np.cumsum(np.random.randn(100)) + 100
    
    # Simulate ARIMA predictions (slightly off)
    arima_preds = prices + np.random.randn(100) * 2
    
    # Simulate LSTM predictions (slightly off, different pattern)
    lstm_preds = prices + np.random.randn(100) * 3
    
    # Train ensemble
    ensemble = StackedEnsemble(meta_model='xgboost')
    ensemble.train(prices, arima_preds, lstm_preds)
    
    # Test prediction
    test_arima = 105.0
    test_lstm = 103.0
    prediction = ensemble.predict(test_arima, test_lstm)
    
    print(f"\nTest Prediction:")
    print(f"  ARIMA: {test_arima:.2f}")
    print(f"  LSTM:  {test_lstm:.2f}")
    print(f"  Stacked Ensemble: {prediction:.2f}")
