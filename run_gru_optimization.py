"""
Script untuk menjalankan optimasi GRU model dan compare dengan model original.

Usage:
    python run_gru_optimization.py

Output:
    - Optimized model performance comparison
    - Saved optimized model
    - Visualization of improvement

Author: [Nama Anda]
Date: January 2026
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')
sys.path.append('src/model')

# Import models
from src.model.gru_model import gru_forecast as gru_original
from src.model.gru_model_optimized import (
    gru_forecast_optimized,
    gru_forecast_quick_optimized,
    OptimizedGRUModel
)
from src.data_loader import load_data

# Import untuk validasi
import warnings
warnings.filterwarnings('ignore')


def compare_models_single_stock(stock_name, data_path='dataset/DaftarSaham.csv'):
    """
    Compare GRU original vs optimized untuk 1 saham.
    
    Args:
        stock_name: Nama saham (contoh: 'ADRO', 'BYAN')
        data_path: Path ke dataset
    """
    print("="*80)
    print(f"🔬 GRU OPTIMIZATION COMPARISON - {stock_name}")
    print("="*80)
    
    # Load data
    print(f"\n📊 Loading data for {stock_name}...")
    df = load_data(data_path, stock_name)
    
    if df is None or len(df) < 100:
        print(f"❌ Data tidak cukup untuk {stock_name}")
        return None
    
    prices = df['Close'].values
    print(f"✅ Loaded {len(prices)} data points")
    
    # Test Original GRU
    print(f"\n{'='*80}")
    print("📌 TESTING ORIGINAL GRU MODEL")
    print(f"{'='*80}")
    start_time = time.time()
    
    try:
        forecast_original = gru_original(prices)
        time_original = time.time() - start_time
        print(f"✅ Original GRU: {forecast_original:.2f}")
        print(f"⏱️  Time: {time_original:.2f} seconds")
    except Exception as e:
        print(f"❌ Error in original GRU: {e}")
        forecast_original = None
        time_original = None
    
    # Test Quick Optimized GRU
    print(f"\n{'='*80}")
    print("📌 TESTING QUICK OPTIMIZED GRU MODEL")
    print(f"{'='*80}")
    start_time = time.time()
    
    try:
        forecast_quick = gru_forecast_quick_optimized(prices, look_back=30)
        time_quick = time.time() - start_time
        print(f"✅ Quick Optimized GRU: {forecast_quick:.2f}")
        print(f"⏱️  Time: {time_quick:.2f} seconds")
    except Exception as e:
        print(f"❌ Error in quick optimized GRU: {e}")
        forecast_quick = None
        time_quick = None
    
    # Test Full Optimized GRU (dengan Keras Tuner)
    print(f"\n{'='*80}")
    print("📌 TESTING FULL OPTIMIZED GRU MODEL (dengan Hyperparameter Tuning)")
    print(f"{'='*80}")
    print("⚠️  This will take 30-60 minutes for full optimization")
    print("💡 Tip: Gunakan quick optimized untuk testing cepat\n")
    
    do_full_optimization = input("Run full optimization? (y/n): ").lower().strip()
    
    if do_full_optimization == 'y':
        start_time = time.time()
        try:
            # Gunakan max_trials=20 untuk demo (default 50 terlalu lama)
            forecast_full = gru_forecast_optimized(
                prices, 
                look_back=30, 
                max_trials=20,
                use_cache=True
            )
            time_full = time.time() - start_time
            print(f"✅ Full Optimized GRU: {forecast_full:.2f}")
            print(f"⏱️  Time: {time_full:.2f} seconds")
        except Exception as e:
            print(f"❌ Error in full optimized GRU: {e}")
            forecast_full = None
            time_full = None
    else:
        forecast_full = None
        time_full = None
        print("⏭️  Skipping full optimization")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    current_price = prices[-1]
    
    results = []
    
    if forecast_original is not None:
        change_orig = ((forecast_original - current_price) / current_price) * 100
        results.append({
            'Model': 'Original GRU',
            'Forecast': forecast_original,
            'Change %': change_orig,
            'Time (s)': time_original,
            'Speed Factor': 1.0
        })
    
    if forecast_quick is not None and time_original is not None:
        change_quick = ((forecast_quick - current_price) / current_price) * 100
        speed_factor = time_original / time_quick
        results.append({
            'Model': 'Quick Optimized',
            'Forecast': forecast_quick,
            'Change %': change_quick,
            'Time (s)': time_quick,
            'Speed Factor': speed_factor
        })
    
    if forecast_full is not None and time_original is not None:
        change_full = ((forecast_full - current_price) / current_price) * 100
        speed_factor = time_original / time_full if time_full > 0 else 0
        results.append({
            'Model': 'Full Optimized',
            'Forecast': forecast_full,
            'Change %': change_full,
            'Time (s)': time_full,
            'Speed Factor': speed_factor
        })
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\nCurrent Price: Rp {current_price:.2f}")
        print(f"\n{df_results.to_string(index=False)}")
        
        # Save results
        output_dir = 'output/optimization'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/gru_optimization_{stock_name}_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return df_results
    
    return None


def compare_models_multiple_stocks(stock_list=None, data_path='dataset/DaftarSaham.csv'):
    """
    Compare GRU optimization untuk multiple stocks.
    
    Args:
        stock_list: List of stock names (None = use top 10 stocks)
        data_path: Path to dataset
    """
    if stock_list is None:
        # Default: top 10 stocks dari validation
        stock_list = ['ADRO', 'BYAN', 'ADMR', 'GEMS', 'ITMG', 
                      'PTBA', 'PGAS', 'TCPI', 'DSSA', 'MEDC']
    
    print("="*80)
    print("🚀 GRU OPTIMIZATION - MULTIPLE STOCKS COMPARISON")
    print("="*80)
    print(f"\nTesting {len(stock_list)} stocks: {', '.join(stock_list)}")
    print("\n⚠️  Note: Using Quick Optimization untuk speed")
    print("         Full optimization membutuhkan waktu ~1 jam per stock\n")
    
    all_results = []
    
    for i, stock in enumerate(stock_list, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(stock_list)}] Processing {stock}...")
        print(f"{'='*80}")
        
        try:
            # Load data
            df = load_data(data_path, stock)
            if df is None or len(df) < 100:
                print(f"⏭️  Skipping {stock} - insufficient data")
                continue
            
            prices = df['Close'].values
            current_price = prices[-1]
            
            # Original GRU
            start_time = time.time()
            forecast_orig = gru_original(prices)
            time_orig = time.time() - start_time
            
            # Quick Optimized GRU
            start_time = time.time()
            forecast_opt = gru_forecast_quick_optimized(prices, look_back=30)
            time_opt = time.time() - start_time
            
            # Calculate changes
            change_orig = ((forecast_orig - current_price) / current_price) * 100
            change_opt = ((forecast_opt - current_price) / current_price) * 100
            
            # Speed improvement
            speed_factor = time_orig / time_opt
            
            result = {
                'Stock': stock,
                'Current_Price': current_price,
                'Original_Forecast': forecast_orig,
                'Original_Change%': change_orig,
                'Original_Time': time_orig,
                'Optimized_Forecast': forecast_opt,
                'Optimized_Change%': change_opt,
                'Optimized_Time': time_opt,
                'Speed_Improvement': speed_factor
            }
            
            all_results.append(result)
            
            print(f"✅ {stock}: Original={forecast_orig:.2f} ({change_orig:+.2f}%), "
                  f"Optimized={forecast_opt:.2f} ({change_opt:+.2f}%), "
                  f"Speed={speed_factor:.2f}x faster")
            
        except Exception as e:
            print(f"❌ Error processing {stock}: {e}")
            continue
    
    # Summary
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print(f"\n{'='*80}")
        print("📊 OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"\nSuccessfully processed: {len(all_results)}/{len(stock_list)} stocks")
        print(f"\nAverage Speed Improvement: {df_results['Speed_Improvement'].mean():.2f}x faster")
        print(f"Average Original Time: {df_results['Original_Time'].mean():.2f}s")
        print(f"Average Optimized Time: {df_results['Optimized_Time'].mean():.2f}s")
        
        # Save results
        output_dir = 'output/optimization'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/gru_optimization_summary_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\n{df_results.to_string(index=False)}")
        
        return df_results
    
    return None


def main():
    """Main function untuk run optimization comparison."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                   GRU MODEL OPTIMIZATION                         ║
    ║                   Comparison Tool                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    Options:
    1. Single Stock Comparison (detailed with full optimization option)
    2. Multiple Stocks Comparison (quick optimization only)
    3. Exit
    """)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == '1':
        stock_name = input("\nEnter stock name (e.g., ADRO, BYAN): ").strip().upper()
        if stock_name:
            compare_models_single_stock(stock_name)
        else:
            print("❌ Invalid stock name")
    
    elif choice == '2':
        print("\nUsing default 10 stocks: ADRO, BYAN, ADMR, GEMS, ITMG, PTBA, PGAS, TCPI, DSSA, MEDC")
        custom = input("Use custom stock list? (y/n): ").lower().strip()
        
        if custom == 'y':
            stocks_input = input("Enter stock names (comma-separated): ").strip()
            stock_list = [s.strip().upper() for s in stocks_input.split(',')]
        else:
            stock_list = None
        
        compare_models_multiple_stocks(stock_list)
    
    elif choice == '3':
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid option")


if __name__ == '__main__':
    main()
