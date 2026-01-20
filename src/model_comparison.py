import pandas as pd
import numpy as np
import time
from src.model.arima_model import arima_forecast
from src.model.lstm_model import lstm_forecast
from src.model.ensemble import ensemble
from src.model.stacked_ensemble import StackedEnsemble
from src.model.weighted_ensemble import WeightedEnsemble

def compare_single_stock(prices, stock_code, stacked_model=None, weighted_model=None):
    """
    Compare 5 models (ARIMA, LSTM, Simple Ensemble, Stacked Ensemble, Weighted Ensemble) untuk 1 saham
    
    NOTE: Prediksi adalah untuk NEXT TRADING DAY (T+1)
          Jika data terakhir = Jumat, prediksi = Senin
    
    Parameters:
    -----------
    prices : array
        Historical prices
    stock_code : str
        Stock code
    stacked_model : StackedEnsemble
        Pre-trained stacked ensemble model (optional)
    weighted_model : WeightedEnsemble
        Pre-trained weighted ensemble model (optional)
    
    Returns:
        dict dengan hasil prediksi dan execution time
    """
    results = {
        'stock': stock_code,
        'current_price': prices[-1],
        'data_points': len(prices)
    }
    
    # Test ARIMA
    print(f"  🔹 Testing ARIMA...")
    start = time.time()
    try:
        arima_pred = arima_forecast(prices)
        arima_time = time.time() - start
        results['arima_forecast'] = arima_pred
        results['arima_change_pct'] = (arima_pred - prices[-1]) / prices[-1] * 100
        results['arima_time'] = arima_time
        print(f"     ✅ ARIMA: {arima_pred:.2f} ({arima_time:.2f}s)")
    except Exception as e:
        print(f"     ❌ ARIMA failed: {e}")
        results['arima_forecast'] = None
        results['arima_change_pct'] = None
        results['arima_time'] = None
    
    # Test LSTM
    print(f"  🔹 Testing LSTM...")
    start = time.time()
    try:
        lstm_pred = lstm_forecast(prices)
        lstm_time = time.time() - start
        results['lstm_forecast'] = lstm_pred
        results['lstm_change_pct'] = (lstm_pred - prices[-1]) / prices[-1] * 100
        results['lstm_time'] = lstm_time
        print(f"     ✅ LSTM: {lstm_pred:.2f} ({lstm_time:.2f}s)")
    except Exception as e:
        print(f"     ❌ LSTM failed: {e}")
        results['lstm_forecast'] = None
        results['lstm_change_pct'] = None
        results['lstm_time'] = None
    
    # Test Simple Ensemble (Average)
    print(f"  🔹 Testing Simple Ensemble...")
    start = time.time()
    try:
        if results['arima_forecast'] is not None and results['lstm_forecast'] is not None:
            ensemble_pred = ensemble(results['arima_forecast'], results['lstm_forecast'])
            ensemble_time = time.time() - start
            results['ensemble_forecast'] = ensemble_pred
            results['ensemble_change_pct'] = (ensemble_pred - prices[-1]) / prices[-1] * 100
            results['ensemble_time'] = ensemble_time
            print(f"     ✅ Simple Ensemble: {ensemble_pred:.2f} ({ensemble_time:.2f}s)")
        else:
            results['ensemble_forecast'] = None
            results['ensemble_change_pct'] = None
            results['ensemble_time'] = None
    except Exception as e:
        print(f"     ❌ Simple Ensemble failed: {e}")
        results['ensemble_forecast'] = None
        results['ensemble_change_pct'] = None
        results['ensemble_time'] = None
    
    # Test Stacked Ensemble (Meta-learning)
    print(f"  🔹 Testing Stacked Ensemble...")
    start = time.time()
    try:
        if stacked_model and stacked_model.is_trained:
            if results['arima_forecast'] is not None and results['lstm_forecast'] is not None:
                stacked_pred = stacked_model.predict(results['arima_forecast'], results['lstm_forecast'])
                stacked_time = time.time() - start
                results['stacked_forecast'] = stacked_pred
                results['stacked_change_pct'] = (stacked_pred - prices[-1]) / prices[-1] * 100
                results['stacked_time'] = stacked_time
                print(f"     ✅ Stacked Ensemble: {stacked_pred:.2f} ({stacked_time:.2f}s)")
            else:
                results['stacked_forecast'] = None
                results['stacked_change_pct'] = None
                results['stacked_time'] = None
        else:
            results['stacked_forecast'] = None
            results['stacked_change_pct'] = None
            results['stacked_time'] = None
            print(f"     ⚠️  Stacked Ensemble: Not trained")
    except Exception as e:
        print(f"     ❌ Stacked Ensemble failed: {e}")
        results['stacked_forecast'] = None
        results['stacked_change_pct'] = None
        results['stacked_time'] = None
    
    # Test Weighted Ensemble (Optimized weights)
    print(f"  🔹 Testing Weighted Ensemble...")
    start = time.time()
    try:
        if weighted_model and weighted_model.is_trained:
            if results['arima_forecast'] is not None and results['lstm_forecast'] is not None:
                weighted_pred = weighted_model.predict(results['arima_forecast'], results['lstm_forecast'])
                weighted_time = time.time() - start
                results['weighted_forecast'] = weighted_pred
                results['weighted_change_pct'] = (weighted_pred - prices[-1]) / prices[-1] * 100
                results['weighted_time'] = weighted_time
                weights = weighted_model.get_weights()
                print(f"     ✅ Weighted Ensemble: {weighted_pred:.2f} ({weighted_time:.2f}s)")
                print(f"        Weights: ARIMA={weights['arima_weight']:.2%}, LSTM={weights['lstm_weight']:.2%}")
            else:
                results['weighted_forecast'] = None
                results['weighted_change_pct'] = None
                results['weighted_time'] = None
        else:
            results['weighted_forecast'] = None
            results['weighted_change_pct'] = None
            results['weighted_time'] = None
            print(f"     ⚠️  Weighted Ensemble: Not trained")
    except Exception as e:
        print(f"     ❌ Weighted Ensemble failed: {e}")
        results['weighted_forecast'] = None
        results['weighted_change_pct'] = None
        results['weighted_time'] = None
    
    return results

def compare_models_bulk(stock_data_dict, use_advanced_ensemble=True):
    """
    Compare models untuk multiple stocks
    
    Parameters:
    -----------
    stock_data_dict : dict
        Dictionary {stock_code: prices_array}
    use_advanced_ensemble : bool
        Whether to train and use stacked/weighted ensemble (default: True)
    
    Returns:
    --------
    DataFrame dengan comparison results
    """
    all_results = []
    stacked_model = None
    weighted_model = None
    
    print("\n" + "="*80)
    if use_advanced_ensemble:
        print("MODEL COMPARISON: ARIMA vs LSTM vs Simple vs Stacked vs Weighted Ensemble")
    else:
        print("MODEL COMPARISON: ARIMA vs LSTM vs Simple Ensemble")
    print("="*80)
    
    # Train advanced ensemble models using stock with most data
    if use_advanced_ensemble and len(stock_data_dict) > 0:
        print("\n🔧 Training Advanced Ensemble Models...")
        print("-" * 80)
        
        # Get stock with most data for training ensemble (best generalization)
        best_stock_code = max(stock_data_dict.items(), key=lambda x: len(x[1]))[0]
        training_prices = stock_data_dict[best_stock_code]
        
        if len(training_prices) >= 150:  # Need enough data for training
            print(f"\n📚 Using {best_stock_code} ({len(training_prices)} days) for ensemble training...")
            
            # Generate training data (walk-forward on historical data)
            arima_preds_train = []
            lstm_preds_train = []
            actuals_train = []
            
            lookback = 100
            target_samples = 50
            step_size = max(1, (len(training_prices) - lookback - 10) // target_samples)
            print(f"   Generating training samples: lookback={lookback}, step={step_size}, target={target_samples}")
            
            # Generate samples across full dataset for better generalization
            sample_count = 0
            for i in range(lookback, len(training_prices) - 10, step_size):
                if sample_count >= target_samples:
                    break
                    
                train_window = training_prices[:i]
                actual = training_prices[i]
                
                try:
                    arima_p = arima_forecast(train_window)
                    lstm_p = lstm_forecast(train_window)
                    
                    arima_preds_train.append(arima_p)
                    lstm_preds_train.append(lstm_p)
                    actuals_train.append(actual)
                    sample_count += 1
                    
                    # Progress tracking
                    if sample_count % 10 == 0:
                        print(f"   → Generated {sample_count}/{target_samples} samples...")
                except:
                    continue
            
            if len(actuals_train) >= 10:  # Need minimum samples
                # Train Stacked Ensemble
                print(f"\n🤖 Training Stacked Ensemble (XGBoost)...")
                stacked_model = StackedEnsemble(meta_model='xgboost')
                stacked_model.train(
                    np.array(actuals_train),
                    np.array(arima_preds_train),
                    np.array(lstm_preds_train),
                    verbose=True
                )
                
                # Train Weighted Ensemble
                print(f"\n⚖️  Training Weighted Ensemble (Optimized Weights)...")
                weighted_model = WeightedEnsemble(optimization_metric='mse')
                weighted_model.train(
                    np.array(actuals_train),
                    np.array(arima_preds_train),
                    np.array(lstm_preds_train),
                    verbose=True
                )
            else:
                print(f"   ⚠️  Insufficient training samples ({len(actuals_train)}), skipping advanced ensemble")
        else:
            print(f"   ⚠️  Insufficient data for training ({len(training_prices)} days), skipping advanced ensemble")
        
        print("-" * 80)
    
    # Compare models for all stocks
    total = len(stock_data_dict)
    for i, (code, prices) in enumerate(stock_data_dict.items(), 1):
        print(f"\n[{i}/{total}] Comparing models for {code}...")
        result = compare_single_stock(prices, code, stacked_model, weighted_model)
        all_results.append(result)
    
    df_comparison = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Average execution time
    print("\n⏱️  Average Execution Time:")
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        col = f'{model}_time'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            print(f"  {model.capitalize():12s}: {df_comparison[col].mean():.2f}s")
    
    # Average predicted change
    print("\n📈 Average Predicted Change:")
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        col = f'{model}_change_pct'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            print(f"  {model.capitalize():12s}: {df_comparison[col].mean():.2f}%")
    
    # Model agreement (how similar are predictions)
    if (df_comparison['arima_forecast'].notna().all() and 
        df_comparison['lstm_forecast'].notna().all()):
        
        # Calculate correlation between predictions
        corr_arima_lstm = df_comparison['arima_change_pct'].corr(df_comparison['lstm_change_pct'])
        print(f"\n🔗 Prediction Correlation:")
        print(f"  ARIMA vs LSTM: {corr_arima_lstm:.3f}")
        
        # Calculate average difference
        avg_diff = abs(df_comparison['arima_change_pct'] - df_comparison['lstm_change_pct']).mean()
        print(f"\n📊 Average Prediction Difference:")
        print(f"  |ARIMA - LSTM|: {avg_diff:.2f}%")
    
    # Display learned weights
    if weighted_model and weighted_model.is_trained:
        weights = weighted_model.get_weights()
        print(f"\n⚖️  Learned Ensemble Weights:")
        print(f"  ARIMA:  {weights['arima_weight']:.2%}")
        print(f"  LSTM:   {weights['lstm_weight']:.2%}")
    
    return df_comparison

def analyze_model_performance(df_comparison):
    """
    Analisis performa model berdasarkan berbagai kriteria
    """
    analysis = {
        'total_stocks': len(df_comparison),
        'successful_predictions': {}
    }
    
    # Count successful predictions
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        col = f'{model}_forecast'
        if col in df_comparison.columns:
            analysis['successful_predictions'][model] = df_comparison[col].notna().sum()
    
    # Speed ranking
    speed_data = []
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        col = f'{model}_time'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            speed_data.append((model.upper(), df_comparison[col].mean()))
    
    speed_data.sort(key=lambda x: x[1])
    analysis['speed_ranking'] = speed_data
    
    # Prediction distribution
    analysis['prediction_stats'] = {}
    for model in ['arima', 'lstm', 'ensemble', 'stacked', 'weighted']:
        col = f'{model}_change_pct'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            analysis['prediction_stats'][model] = {
                'mean': df_comparison[col].mean(),
                'std': df_comparison[col].std(),
                'min': df_comparison[col].min(),
                'max': df_comparison[col].max(),
                'bullish_count': (df_comparison[col] > 0).sum(),
                'bearish_count': (df_comparison[col] < 0).sum(),
            }
    
    return analysis

def generate_comparison_report(df_comparison, analysis, output_path):
    """
    Generate detailed text report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("ARIMA vs LSTM vs ENSEMBLE\n")
        f.write("="*80 + "\n\n")
        
        from datetime import datetime
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Stocks Analyzed: {analysis['total_stocks']}\n")
        f.write(f"\n⚠️  FORECAST HORIZON: Next Trading Day (T+1)\n")
        f.write(f"    All predictions are for the next available trading day.\n")
        f.write(f"    Example: Last data = Friday 6 Jan 2023 → Forecast = Monday 9 Jan 2023\n")
        f.write(f"    (Automatically skips weekends and market holidays)\n\n")
        
        # Successful predictions
        f.write("="*80 + "\n")
        f.write("SUCCESSFUL PREDICTIONS\n")
        f.write("="*80 + "\n")
        for model, count in analysis['successful_predictions'].items():
            pct = count / analysis['total_stocks'] * 100
            f.write(f"{model.upper():12s}: {count}/{analysis['total_stocks']} ({pct:.1f}%)\n")
        f.write("\n")
        
        # Speed ranking
        f.write("="*80 + "\n")
        f.write("EXECUTION SPEED RANKING (Fastest to Slowest)\n")
        f.write("="*80 + "\n")
        for rank, (model, time_val) in enumerate(analysis['speed_ranking'], 1):
            f.write(f"{rank}. {model:12s}: {time_val:.2f}s average\n")
        f.write("\n")
        
        # Prediction statistics
        f.write("="*80 + "\n")
        f.write("PREDICTION STATISTICS\n")
        f.write("="*80 + "\n\n")
        for model, stats in analysis['prediction_stats'].items():
            f.write(f"{model.upper()}:\n")
            f.write(f"  Average Change:  {stats['mean']:+.2f}%\n")
            f.write(f"  Std Deviation:   {stats['std']:.2f}%\n")
            f.write(f"  Min/Max:         {stats['min']:+.2f}% / {stats['max']:+.2f}%\n")
            f.write(f"  Bullish Signals: {stats['bullish_count']} stocks\n")
            f.write(f"  Bearish Signals: {stats['bearish_count']} stocks\n")
            f.write("\n")
        
        # Detailed comparison table
        f.write("="*80 + "\n")
        f.write("DETAILED COMPARISON TABLE\n")
        f.write("="*80 + "\n")
        f.write(df_comparison.to_string(index=False))
        f.write("\n\n")
        
        # Model recommendations
        f.write("="*80 + "\n")
        f.write("MODEL RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        fastest_model = analysis['speed_ranking'][0][0] if analysis['speed_ranking'] else 'N/A'
        f.write(f"🏃 Fastest Model: {fastest_model}\n")
        f.write(f"   → Best for: Real-time trading, quick decisions\n\n")
        
        f.write(f"🤖 LSTM:\n")
        f.write(f"   → Best for: Complex patterns, high-volatility stocks\n")
        f.write(f"   → Requires: More data, longer computation time\n\n")
        
        f.write(f"⚖️  Ensemble:\n")
        f.write(f"   → Best for: Balanced approach, risk mitigation\n")
        f.write(f"   → Combines: Strengths of both models\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Comparison report saved: {output_path}")
