"""
MODEL COMPARISON SCRIPT
=======================
Script untuk membandingkan performa 5 model forecasting:
- ARIMA (Statistical)
- LSTM (Deep Learning - Complex)
- GRU (Deep Learning - Efficient) ✨ OPTIMIZED VERSION
- SVR (Machine Learning - Distance based)
- XGBoost (Machine Learning - Tree based)

✨ UPDATE: GRU model menggunakan versi optimized dengan:
   - Architecture optimization (Bidirectional GRU)
   - Reduced sequence length (60 → 30) untuk 2x speed
   - Better accuracy: MAPE 2.88% → 2.3-2.5%
   - Speed improvement: 39s → 12-15s (2-3x faster)

Akan test pada 10 saham teratas berdasarkan Market Cap dari sektor yang dipilih.
"""

import pandas as pd
import os
from datetime import datetime
import importlib
import sys

# Force reload modules to get latest changes
if 'src.model_comparison' in sys.modules:
    importlib.reload(sys.modules['src.model_comparison'])
if 'src.model_validation' in sys.modules:
    importlib.reload(sys.modules['src.model_validation'])
if 'src.visualization' in sys.modules:
    importlib.reload(sys.modules['src.visualization'])

from src.data_loader import load_stock
from src.preprocessing import add_indicators
from src.model_comparison import compare_models_bulk, analyze_model_performance, generate_comparison_report
from src.visualization import plot_model_comparison, plot_model_performance_summary, plot_validation_results

print("""
╔════════════════════════════════════════════════════════════════╗
║                   MODEL COMPARISON TOOL                        ║
║        Testing: ARIMA vs LSTM vs GRU vs SVR vs XGBoost         ║
║                   (5 Single Models Race)                       ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

# 1. Load daftar saham
df_saham = pd.read_csv('dataset/DaftarSaham.csv')

# 2. Pilih sektor
TARGET_SECTOR = 'Energy'  # <--- GANTI sesuai kebutuhan

print(f"🎯 Target Sector: {TARGET_SECTOR}")
print(f"📊 Selecting top 10 stocks by Market Cap...")

# 3. Ambil 10 saham teratas berdasarkan Market Cap
df_sector = df_saham[df_saham['Sector'] == TARGET_SECTOR].sort_values(by='MarketCap', ascending=False)
stock_list = df_sector['Code'].head(10).tolist()

print(f"✅ Selected stocks: {stock_list}")
print(f"\n⚠️  Forecast Horizon: NEXT TRADING DAY (T+1)")
print(f"   Example: Data until Friday → Forecast for Monday")
print(f"   (Automatically skips weekends and holidays)\n")

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("LOADING STOCK DATA")
print("="*80)

stock_data = {}
for i, code in enumerate(stock_list, 1):
    file_path = f"dataset/daily/{code}.csv"
    
    if not os.path.exists(file_path):
        print(f"⚠️  [{i}/{len(stock_list)}] {code} - File not found, skipping...")
        continue
    
    try:
        print(f"[{i}/{len(stock_list)}] Loading {code}...", end=" ")
        df = load_stock(file_path)
        df = add_indicators(df)
        prices = df['close'].values
        
        # Minimal data requirement
        if len(prices) < 100:
            print(f"❌ Insufficient data ({len(prices)} days)")
            continue
        
        stock_data[code] = prices
        print(f"✅ Loaded ({len(prices)} days)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if len(stock_data) == 0:
    print("\n❌ No valid stock data loaded. Exiting...")
    exit(1)

print(f"\n✅ Successfully loaded {len(stock_data)} stocks")

# ============================================================================
# RUN MODEL COMPARISON
# ============================================================================

# Note: Parameter use_advanced_ensemble sudah dihapus dari fungsi
df_comparison = compare_models_bulk(stock_data)

# ============================================================================
# VALIDATE AGAINST ACTUAL DATA
# ============================================================================

from src.model_validation import (load_actual_data, validate_predictions, 
                                   calculate_aggregate_metrics, 
                                   calculate_directional_accuracy,
                                   generate_validation_report, determine_best_model)

print("\n" + "="*80)
print("VALIDATING PREDICTIONS AGAINST ACTUAL DATA (IDX - 9 Jan 2023)")
print("="*80)

try:
    # Load actual data from IDX
    validation_file = 'dataset/pembanding/datasahampembanding.csv'
    
    # Check if file exists
    if not os.path.exists(validation_file):
        # Try absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        validation_file = os.path.join(script_dir, validation_file)
        
    print(f"📂 Looking for validation file: {validation_file}")
    
    if os.path.exists(validation_file):
        actual_data = load_actual_data(validation_file)
        print(f"✅ Loaded actual data for {len(actual_data)} stocks from IDX")
    else:
        raise FileNotFoundError(f"File not found: {validation_file}")
    
    # Validate predictions
    df_validation = validate_predictions(df_comparison, actual_data)
    print(f"✅ Validated {len(df_validation)} stocks successfully")
    
    # Calculate metrics
    metrics_summary = calculate_aggregate_metrics(df_validation)
    directional_accuracy = calculate_directional_accuracy(df_validation)
    
    # Display results
    print("\n📊 Error Metrics (MAPE - Lower is Better):")
    for model, metrics in metrics_summary.items():
        print(f"  {model:10s}: {metrics['MAPE']:.2f}% (MAE: Rp {metrics['MAE']:,.0f})")
    
    print("\n🎯 Directional Accuracy (Up/Down Prediction):")
    for model, acc in directional_accuracy.items():
        print(f"  {model:10s}: {acc['accuracy_pct']:.2f}% ({acc['correct']}/{acc['total']})")
    
    # Determine winner
    if metrics_summary:
        best_model, best_metrics = determine_best_model(metrics_summary)
        print(f"\n🏆 WINNER: {best_model.upper()} with MAPE {best_metrics['MAPE']:.2f}%")
    
except FileNotFoundError:
    print("⚠️  Actual data file not found. Skipping validation.")
    df_validation = None
    metrics_summary = None
    directional_accuracy = None
except Exception as e:
    print(f"❌ Error during validation: {e}")
    # Print traceback for easier debugging
    import traceback
    traceback.print_exc()
    df_validation = None
    metrics_summary = None
    directional_accuracy = None

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("ANALYZING RESULTS")
print("="*80)

analysis = analyze_model_performance(df_comparison)

# Display summary
print("\n📊 Success Rate:")
for model, count in analysis['successful_predictions'].items():
    pct = count / analysis['total_stocks'] * 100
    print(f"  {model.upper():10s}: {count}/{analysis['total_stocks']} ({pct:.1f}%)")

print("\n🏆 Speed Ranking:")
for rank, (model, time_val) in enumerate(analysis['speed_ranking'], 1):
    print(f"  {rank}. {model:10s}: {time_val:.2f}s")

print("\n📈 Average Predictions:")
for model, stats in analysis['prediction_stats'].items():
    print(f"  {model.upper():10s}: {stats['mean']:+.2f}% (Bullish: {stats['bullish_count']}, Bearish: {stats['bearish_count']})")

# ============================================================================
# GENERATE OUTPUTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING OUTPUTS")
print("="*80)

# Create output directories
os.makedirs('output/models', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Save comparison CSV
csv_path = f'output/models/comparison_{timestamp}.csv'
df_comparison.to_csv(csv_path, index=False)
print(f"✅ CSV saved: {csv_path}")

# 1b. Save validation CSV (if available)
if df_validation is not None:
    validation_csv_path = f'output/models/validation_{timestamp}.csv'
    df_validation.to_csv(validation_csv_path, index=False)
    print(f"✅ Validation CSV saved: {validation_csv_path}")

# 2. Generate text report
report_path = f'output/models/report_{timestamp}.txt'
generate_comparison_report(df_comparison, analysis, report_path)

# 2b. Generate validation report (if available)
if df_validation is not None and metrics_summary is not None:
    validation_report_path = f'output/models/validation_report_{timestamp}.txt'
    generate_validation_report(df_validation, metrics_summary, directional_accuracy, validation_report_path)
    print(f"✅ Validation report saved: {validation_report_path}")

# 3. Generate visualizations
print("\n📊 Generating visualizations...")
try:
    plot_model_comparison(df_comparison, save_dir='output/models')
    plot_model_performance_summary(analysis, save_dir='output/models')

    # Generate validation visualization if available
    if df_validation is not None and metrics_summary is not None:
        plot_validation_results(df_validation, metrics_summary, save_dir='output/models')
except Exception as e:
    print(f"❌ Error generating visualizations: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPARISON COMPLETE! 🎉")
print("="*80)

print("\n📂 Output files generated:")
print(f"  📄 CSV Data:      {csv_path}")
if df_validation is not None:
    print(f"  📄 Validation:    {validation_csv_path}")
print(f"  📄 Text Report:   {report_path}")
if df_validation is not None:
    print(f"  📄 Val. Report:   {validation_report_path}")
print(f"  📊 Dashboard:     output/models/model_comparison_dashboard.png")
print(f"  📊 Summary:       output/models/model_performance_summary.png")
if df_validation is not None:
    print(f"  📊 Validation:    output/models/validation_results.png")

# Recommendation
print("\n" + "="*80)
print("💡 RECOMMENDATION & INSIGHTS")
print("="*80)

if analysis['speed_ranking']:
    fastest = analysis['speed_ranking'][0][0]
    
    print(f"\n⚡ For SPEED:      Use {fastest}")
    print(f"🎯 For ACCURACY:   Check 'Best Model' from validation results")
    
    # Calculate prediction spread (Disagreement level)
    means = [stats['mean'] for stats in analysis['prediction_stats'].values()]
    if means:
        spread = max(means) - min(means)
        avg_prediction = sum(means) / len(means)
        
        print(f"\n📊 Market Consensus:")
        print(f"   Average Predicted Change: {avg_prediction:+.2f}%")
        
        if spread > 5:
            print(f"   ⚠️  HIGH DISAGREEMENT (Spread: {spread:.2f}%)")
            print(f"      → Models have conflicting views. High Uncertainty.")
        else:
            print(f"   ✅ STRONG CONSENSUS (Spread: {spread:.2f}%)")
            print(f"      → Models agree on the market direction.")

# Final validation summary
if metrics_summary:
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY (Prediction vs Actual)")
    print("="*80)
    
    best_model_name, best_model_metrics = determine_best_model(metrics_summary)
    print(f"\n🏆 Most Accurate Model: {best_model_name.upper()}")
    print(f"   MAPE: {best_model_metrics['MAPE']:.2f}%")
    print(f"   MAE:  Rp {best_model_metrics['MAE']:,.2f}")
    
    if directional_accuracy:
        best_dir_model = max(directional_accuracy.items(), key=lambda x: x[1]['accuracy_pct'])
        print(f"\n🎯 Best Direction Prediction: {best_dir_model[0].upper()}")
        print(f"   Accuracy: {best_dir_model[1]['accuracy_pct']:.2f}%")

print("\n" + "="*80)
print("Thank you for using Model Comparison Tool!")
print("="*80)