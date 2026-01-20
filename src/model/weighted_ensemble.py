"""
WEIGHTED ENSEMBLE MODEL
=======================
Dynamic weight optimization untuk ensemble.
Menggunakan optimization algorithms untuk menemukan bobot optimal
yang meminimalkan error antara prediksi ensemble dan actual.

Berbeda dengan Stacked Ensemble:
- Stacked: Train meta-model untuk belajar kombinasi non-linear
- Weighted: Optimize weights (linear combination) untuk minimize error
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error


class WeightedEnsemble:
    """
    Ensemble dengan optimized weights untuk ARIMA + LSTM
    
    Formula: prediction = w1 * arima + w2 * lstm
    Constraint: w1 + w2 = 1, w1 >= 0, w2 >= 0
    
    Parameters:
    -----------
    optimization_metric : str
        Metric untuk optimize: 'mse', 'mae', 'mape'
    """
    
    def __init__(self, optimization_metric='mse'):
        self.optimization_metric = optimization_metric
        self.weights = None
        self.is_trained = False
        self.optimization_result = None
        
    def _objective_function(self, weights, arima_preds, lstm_preds, actuals):
        """
        Objective function untuk minimize
        
        Parameters:
        -----------
        weights : array
            [w_arima, w_lstm]
        arima_preds : array
            ARIMA predictions
        lstm_preds : array
            LSTM predictions
        actuals : array
            Actual values
            
        Returns:
        --------
        error : float
            Error metric to minimize
        """
        w_arima, w_lstm = weights
        
        # Ensemble prediction
        predictions = w_arima * arima_preds + w_lstm * lstm_preds
        
        # Calculate error based on metric
        if self.optimization_metric == 'mse':
            error = mean_squared_error(actuals, predictions)
        elif self.optimization_metric == 'mae':
            error = mean_absolute_error(actuals, predictions)
        elif self.optimization_metric == 'mape':
            error = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        else:
            error = mean_squared_error(actuals, predictions)
        
        return error
    
    def train(self, actuals, arima_predictions, lstm_predictions, verbose=True):
        """
        Optimize weights menggunakan scipy.optimize
        
        Parameters:
        -----------
        actuals : array
            Actual prices
        arima_predictions : array
            ARIMA predictions
        lstm_predictions : array
            LSTM predictions
        verbose : bool
            Print optimization results
        """
        # Convert to numpy arrays
        actuals = np.array(actuals)
        arima_preds = np.array(arima_predictions)
        lstm_preds = np.array(lstm_predictions)
        
        # Initial guess: equal weights
        initial_weights = [0.5, 0.5]
        
        # Constraints: w1 + w2 = 1
        constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
        
        # Bounds: 0.1 <= w1, w2 <= 0.9 (prevent extreme weights)
        bounds = [(0.1, 0.9), (0.1, 0.9)]
        
        # Optimize
        result = minimize(
            self._objective_function,
            initial_weights,
            args=(arima_preds, lstm_preds, actuals),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        self.weights = result.x
        self.optimization_result = result
        self.is_trained = True
        
        if verbose:
            optimal_error = result.fun
            print(f"✅ Weighted Ensemble optimized successfully!")
            print(f"   Optimization metric: {self.optimization_metric.upper()}")
            print(f"   ARIMA weight: {self.weights[0]:.4f} ({self.weights[0]*100:.2f}%)")
            print(f"   LSTM weight:  {self.weights[1]:.4f} ({self.weights[1]*100:.2f}%)")
            print(f"   Optimized {self.optimization_metric.upper()}: {optimal_error:.4f}")
            print(f"   Convergence: {'Success' if result.success else 'Failed'}")
    
    def predict(self, arima_pred, lstm_pred):
        """
        Predict menggunakan optimized weights
        
        Parameters:
        -----------
        arima_pred : float
            ARIMA prediction
        lstm_pred : float
            LSTM prediction
            
        Returns:
        --------
        prediction : float
            Weighted ensemble prediction
        """
        if not self.is_trained:
            # Fallback to equal weights
            return (arima_pred + lstm_pred) / 2
        
        w_arima, w_lstm = self.weights
        prediction = w_arima * arima_pred + w_lstm * lstm_pred
        
        return prediction
    
    def get_weights(self):
        """
        Get optimized weights
        
        Returns:
        --------
        weights : dict
            Dictionary dengan arima_weight dan lstm_weight
        """
        if self.weights is not None:
            return {
                'arima_weight': self.weights[0],
                'lstm_weight': self.weights[1]
            }
        else:
            return {'arima_weight': 0.5, 'lstm_weight': 0.5}
    
    def evaluate(self, actuals, arima_predictions, lstm_predictions):
        """
        Evaluate ensemble performance
        
        Parameters:
        -----------
        actuals : array
            Actual values
        arima_predictions : array
            ARIMA predictions
        lstm_predictions : array
            LSTM predictions
            
        Returns:
        --------
        metrics : dict
            Dictionary dengan MAE, RMSE, MAPE
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        actuals = np.array(actuals)
        arima_preds = np.array(arima_predictions)
        lstm_preds = np.array(lstm_predictions)
        
        # Ensemble predictions
        ensemble_preds = self.weights[0] * arima_preds + self.weights[1] * lstm_preds
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, ensemble_preds)
        rmse = np.sqrt(mean_squared_error(actuals, ensemble_preds))
        mape = np.mean(np.abs((actuals - ensemble_preds) / actuals)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }


def grid_search_weights(actuals, arima_predictions, lstm_predictions, step=0.1):
    """
    Grid search untuk menemukan optimal weights
    (Alternative method tanpa scipy.optimize)
    
    Parameters:
    -----------
    actuals : array
        Actual values
    arima_predictions : array
        ARIMA predictions
    lstm_predictions : array
        LSTM predictions
    step : float
        Step size for grid search (default: 0.1)
        
    Returns:
    --------
    best_weights : tuple
        (w_arima, w_lstm) yang memberikan error terendah
    best_error : float
        MSE terendah
    """
    actuals = np.array(actuals)
    arima_preds = np.array(arima_predictions)
    lstm_preds = np.array(lstm_predictions)
    
    best_error = float('inf')
    best_weights = (0.5, 0.5)
    
    # Grid search
    for w_arima in np.arange(0, 1 + step, step):
        w_lstm = 1 - w_arima
        
        # Ensemble prediction
        ensemble_pred = w_arima * arima_preds + w_lstm * lstm_preds
        
        # Calculate MSE
        mse = mean_squared_error(actuals, ensemble_pred)
        
        if mse < best_error:
            best_error = mse
            best_weights = (w_arima, w_lstm)
    
    return best_weights, best_error


def compare_weighting_methods(actuals, arima_predictions, lstm_predictions):
    """
    Compare different weighting methods
    
    Returns:
    --------
    results : dict
        Comparison of different methods
    """
    actuals = np.array(actuals)
    arima_preds = np.array(arima_predictions)
    lstm_preds = np.array(lstm_predictions)
    
    results = {}
    
    # 1. Equal weights (baseline)
    equal_pred = 0.5 * arima_preds + 0.5 * lstm_preds
    results['equal'] = {
        'weights': (0.5, 0.5),
        'mse': mean_squared_error(actuals, equal_pred),
        'mae': mean_absolute_error(actuals, equal_pred)
    }
    
    # 2. Optimized weights (MSE)
    ensemble_mse = WeightedEnsemble(optimization_metric='mse')
    ensemble_mse.train(actuals, arima_preds, lstm_preds, verbose=False)
    weights_mse = ensemble_mse.get_weights()
    pred_mse = ensemble_mse.weights[0] * arima_preds + ensemble_mse.weights[1] * lstm_preds
    results['optimized_mse'] = {
        'weights': (weights_mse['arima_weight'], weights_mse['lstm_weight']),
        'mse': mean_squared_error(actuals, pred_mse),
        'mae': mean_absolute_error(actuals, pred_mse)
    }
    
    # 3. Optimized weights (MAE)
    ensemble_mae = WeightedEnsemble(optimization_metric='mae')
    ensemble_mae.train(actuals, arima_preds, lstm_preds, verbose=False)
    weights_mae = ensemble_mae.get_weights()
    pred_mae = ensemble_mae.weights[0] * arima_preds + ensemble_mae.weights[1] * lstm_preds
    results['optimized_mae'] = {
        'weights': (weights_mae['arima_weight'], weights_mae['lstm_weight']),
        'mse': mean_squared_error(actuals, pred_mae),
        'mae': mean_absolute_error(actuals, pred_mae)
    }
    
    # 4. Grid search
    grid_weights, grid_mse = grid_search_weights(actuals, arima_preds, lstm_preds)
    pred_grid = grid_weights[0] * arima_preds + grid_weights[1] * lstm_preds
    results['grid_search'] = {
        'weights': grid_weights,
        'mse': mean_squared_error(actuals, pred_grid),
        'mae': mean_absolute_error(actuals, pred_grid)
    }
    
    return results


# Quick test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate data
    actuals = np.cumsum(np.random.randn(100)) + 100
    arima_preds = actuals + np.random.randn(100) * 2
    lstm_preds = actuals + np.random.randn(100) * 3
    
    # Test weighted ensemble
    ensemble = WeightedEnsemble(optimization_metric='mse')
    ensemble.train(actuals, arima_preds, lstm_preds)
    
    # Test prediction
    test_arima = 105.0
    test_lstm = 103.0
    prediction = ensemble.predict(test_arima, test_lstm)
    
    print(f"\nTest Prediction:")
    print(f"  ARIMA: {test_arima:.2f}")
    print(f"  LSTM:  {test_lstm:.2f}")
    print(f"  Weighted Ensemble: {prediction:.2f}")
    
    # Compare methods
    print("\n" + "="*60)
    print("COMPARING WEIGHTING METHODS")
    print("="*60)
    results = compare_weighting_methods(actuals, arima_preds, lstm_preds)
    
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Weights: ARIMA={metrics['weights'][0]:.3f}, LSTM={metrics['weights'][1]:.3f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
