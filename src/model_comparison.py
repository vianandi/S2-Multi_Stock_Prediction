import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import model-model single
from src.model.arima_model import arima_forecast
from src.model.lstm_model import lstm_forecast
# from src.model.gru_model import gru_forecast       # Original GRU
from src.model.gru_model_optimized import gru_forecast_quick_optimized as gru_forecast  # ✨ OPTIMIZED GRU
from src.model.svr_model import svr_forecast       # Model Baru
from src.model.xgboost_model import xgboost_forecast # Model Baru

def compare_single_stock(prices, stock_code):
    """
    Compare 5 Single Models (ARIMA, LSTM, GRU, SVR, XGBoost) untuk 1 saham.
    
    NOTE: Prediksi adalah untuk NEXT TRADING DAY (T+1)
    """
    results = {
        'stock': stock_code,
        'current_price': prices[-1],
        'data_points': len(prices)
    }
    
    # Daftar model yang akan diuji
    # Format: (nama_key, fungsi_model)
    models = [
        ('arima', arima_forecast),
        ('lstm', lstm_forecast),
        ('gru', gru_forecast),
        ('svr', svr_forecast),
        ('xgb', xgboost_forecast)
    ]
    
    for name, func in models:
        print(f"  🔹 Testing {name.upper()}...")
        start = time.time()
        try:
            # Jalankan forecasting
            # ✨ Special handling for optimized GRU
            if name == 'gru':
                pred = func(prices, look_back=30, epochs=50)  # Optimized parameters
            else:
                pred = func(prices)
            exec_time = time.time() - start
            
            # Simpan hasil
            results[f'{name}_forecast'] = pred
            results[f'{name}_change_pct'] = (pred - prices[-1]) / prices[-1] * 100
            results[f'{name}_time'] = exec_time
            
            print(f"     ✅ {name.upper()}: {pred:.2f} ({exec_time:.2f}s)")
            
        except Exception as e:
            print(f"     ❌ {name.upper()} failed: {e}")
            results[f'{name}_forecast'] = None
            results[f'{name}_change_pct'] = None
            results[f'{name}_time'] = None
    
    return results

def compare_models_bulk(stock_data_dict):
    """
    Looping perbandingan model untuk semua saham dalam dictionary.
    """
    all_results = []
    
    print("\n" + "="*80)
    print("MODEL COMPARISON: ARIMA vs LSTM vs GRU vs SVR vs XGBoost")
    print("="*80)
    
    total = len(stock_data_dict)
    for i, (code, prices) in enumerate(stock_data_dict.items(), 1):
        print(f"\n[{i}/{total}] Comparing models for {code}...")
        result = compare_single_stock(prices, code)
        all_results.append(result)
    
    df_comparison = pd.DataFrame(all_results)
    
    # --- SUMMARY STATISTICS ---
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    model_list = ['arima', 'lstm', 'gru', 'svr', 'xgb']
    
    # 1. Average execution time
    print("\n⏱️  Average Execution Time:")
    for model in model_list:
        col = f'{model}_time'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            print(f"  {model.upper():12s}: {df_comparison[col].mean():.2f}s")
    
    # 2. Average predicted change
    print("\n📈 Average Predicted Change:")
    for model in model_list:
        col = f'{model}_change_pct'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            print(f"  {model.upper():12s}: {df_comparison[col].mean():.2f}%")
            
    return df_comparison

def analyze_model_performance(df_comparison):
    """
    Analisis performa model: Success Rate, Speed, dan Distribusi Prediksi
    """
    model_list = ['arima', 'lstm', 'gru', 'svr', 'xgb']
    
    analysis = {
        'total_stocks': len(df_comparison),
        'successful_predictions': {},
        'speed_ranking': [],
        'prediction_stats': {}
    }
    
    # 1. Count successful predictions
    for model in model_list:
        col = f'{model}_forecast'
        if col in df_comparison.columns:
            analysis['successful_predictions'][model] = df_comparison[col].notna().sum()
    
    # 2. Speed ranking
    for model in model_list:
        col = f'{model}_time'
        if col in df_comparison.columns and df_comparison[col].notna().any():
            analysis['speed_ranking'].append((model.upper(), df_comparison[col].mean()))
    
    analysis['speed_ranking'].sort(key=lambda x: x[1])
    
    # 3. Prediction distribution stats
    for model in model_list:
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
        f.write("ARIMA vs LSTM vs GRU vs SVR vs XGBoost\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Stocks Analyzed: {analysis['total_stocks']}\n")
        f.write(f"\n⚠️  FORECAST HORIZON: Next Trading Day (T+1)\n\n")
        
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
        
        # Model recommendations (Updated)
        f.write("="*80 + "\n")
        f.write("MODEL CHARACTERISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"🤖 LSTM/GRU:\n")
        f.write(f"   → Deep Learning models. Best for capturing complex, non-linear patterns.\n")
        f.write(f"   → GRU is generally faster than LSTM.\n\n")
        
        f.write(f"🌲 XGBoost:\n")
        f.write(f"   → Tree-based ensemble. Very popular in competitions. Fast and accurate.\n\n")
        
        f.write(f"📉 ARIMA:\n")
        f.write(f"   → Statistical model. Best for linear trends and simpler patterns.\n\n")
        
        f.write(f"📐 SVR:\n")
        f.write(f"   → Geometric model. Good for smaller datasets where Deep Learning might overfit.\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Comparison report saved: {output_path}")