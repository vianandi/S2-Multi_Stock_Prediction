from src.data_loader import load_stock
from src.preprocessing import add_indicators
from src.decision import make_decision
from src.profit_simulation import simulate
from src.visualization import (plot_stock, plot_technical_indicators, 
                                plot_risk_metrics, plot_portfolio_allocation,
                                plot_correlation_matrix)
# ✅ UPDATE: Import 5 Model Single (with Optimized GRU)
from src.model.arima_model import arima_forecast
from src.model.lstm_model import lstm_forecast
# from src.model.gru_model import gru_forecast  # Original GRU
from src.model.gru_model_optimized import gru_forecast_quick_optimized as gru_forecast  # ✨ OPTIMIZED GRU
from src.model.svr_model import svr_forecast
from src.model.xgboost_model import xgboost_forecast

from src.risk_management import (calculate_all_risk_metrics, risk_assessment,
                                  calculate_stop_loss_take_profit)
from src.portfolio_optimization import optimize_portfolio

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ✅ Buat folder output
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

# ==============================================================================
# KONFIGURASI PEMILIHAN SEKTOR OTOMATIS
# ==============================================================================

# 1. Load daftar seluruh saham
df_saham = pd.read_csv('dataset/DaftarSaham.csv')

# 2. Tentukan Sektor yang ingin dianalisa
TARGET_SECTOR = 'Energy'  # <--- GANTI INI sesuai sektor yang diinginkan

# 3. Filter saham berdasarkan sektor
# Ambil 10 saham teratas berdasarkan Market Cap agar waktu proses wajar
df_sector = df_saham[df_saham['Sector'] == TARGET_SECTOR].sort_values(by='MarketCap', ascending=False)
stock_list = df_sector['Code'].head(10).tolist()

print("="*60)
print(f"DECISION SUPPORT SYSTEM - SECTOR: {TARGET_SECTOR.upper()}")
print("="*60)
print(f"Total saham ditemukan: {len(df_sector)}")
print(f"Top 10 Analyzed: {stock_list}")
print(f"\n⚠️  Forecast Horizon: NEXT TRADING DAY (T+1)")
print(f"   (Prediksi untuk hari bursa berikutnya, skip weekend/holidays)")
print("-" * 60)

results = []
stock_data_for_portfolio = {}  # Store price data for portfolio optimization

# ==============================================================================
# PROSES ANALISA
# ==============================================================================

for i, code in enumerate(stock_list, 1):
    # Cek apakah file data harian tersedia
    file_path = f"dataset/daily/{code}.csv"
    
    if not os.path.exists(file_path):
        print(f"⚠️  [{i}/{len(stock_list)}] Data {code} tidak ditemukan ({file_path}), dilewati.")
        continue

    try:
        print(f"[{i}/{len(stock_list)}] Processing {code}...")
        df = load_stock(file_path)
        
        df = add_indicators(df)
        prices = df['close'].values 
        
        # Cek jika data terlalu sedikit
        if len(prices) < 100:
            print(f"❌ Data {code} terlalu sedikit ({len(prices)}), dilewati.")
            continue

        # Store data for portfolio optimization
        stock_data_for_portfolio[code] = prices
        
        # ✅ RUN 5 MODELS (with Optimized GRU)
        try:
            pred_arima = arima_forecast(prices)
            pred_lstm = lstm_forecast(prices)
            pred_gru = gru_forecast(prices, look_back=30, epochs=50)  # ✨ Optimized: 2-3x faster
            pred_svr = svr_forecast(prices)
            pred_xgb = xgboost_forecast(prices)
            
            # ✅ CALCULATE CONSENSUS (AVERAGE)
            # Kita menggunakan rata-rata dari 5 model untuk keputusan yang lebih stabil
            forecast_consensus = np.mean([pred_arima, pred_lstm, pred_gru, pred_svr, pred_xgb])
            
        except Exception as e:
            print(f"❌ Error during modeling for {code}: {e}")
            continue
            
        # Decision making based on Consensus
        decision, change = make_decision(prices[-1], forecast_consensus)
        
        # ✅ RISK MANAGEMENT: Calculate risk metrics
        risk_metrics = calculate_all_risk_metrics(prices)
        risk_level = risk_assessment(risk_metrics)
        
        # ✅ Calculate Stop Loss & Take Profit (using ATR)
        atr = df['ATR'].iloc[-1]
        sl_tp = calculate_stop_loss_take_profit(prices[-1], atr)
        
        # Get last data date
        last_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
        
        results.append({
            "Stock": code,
            "Last_Date": last_date,
            "Current": prices[-1],
            "Forecast": forecast_consensus,
            "Change%": change,
            "Decision": decision,
            # Individual Model Forecasts (for report)
            "ARIMA": pred_arima,
            "LSTM": pred_lstm,
            "GRU": pred_gru,
            "SVR": pred_svr,
            "XGB": pred_xgb,
            # Risk metrics
            "Volatility": risk_metrics['volatility'],
            "Sharpe_Ratio": risk_metrics['sharpe_ratio'],
            "Max_Drawdown": risk_metrics['max_drawdown'],
            "Risk_Level": risk_level,
            # Stop Loss & Take Profit
            "Stop_Loss": sl_tp['stop_loss'],
            "Take_Profit": sl_tp['take_profit'],
            "Risk_Reward_Ratio": sl_tp['risk_reward_ratio']
        })
        
        # Visualization (Plot Consensus)
        plot_stock(df, forecast_consensus, code)
        plot_technical_indicators(df, code)
        
        print(f"✅ {code} completed! Consensus: {decision} ({change:+.2f}%) | Sharpe: {risk_metrics['sharpe_ratio']:.2f}\n")
        
    except Exception as err:
        print(f"❌ Error pada {code}: {err}\n")
        import traceback
        traceback.print_exc()

# ✅ Create DataFrame for analysis
df_results = pd.DataFrame(results)

if df_results.empty:
    print("❌ No results generated. Exiting.")
    exit()

# ✅ Simulasi profit
profit, roi = simulate(df_results.to_dict('records'))

# ✅ PORTFOLIO OPTIMIZATION
print("\n" + "="*60)
print("PORTFOLIO OPTIMIZATION")
print("="*60)

portfolio_results = {}
if len(stock_data_for_portfolio) >= 2:
    # Optimize portfolio with different methods
    portfolio_methods = {
        'max_sharpe': 'Maximum Sharpe Ratio',
        'min_variance': 'Minimum Variance',
        'risk_parity': 'Risk Parity'
    }
    
    for method, name in portfolio_methods.items():
        print(f"\n🔍 Optimizing: {name}")
        try:
            portfolio = optimize_portfolio(stock_data_for_portfolio, method=method)
            portfolio_results[method] = portfolio
            
            print(f"✅ Expected Return: {portfolio['expected_return']*100:.2f}%")
            print(f"✅ Volatility: {portfolio['volatility']*100:.2f}%")
            print(f"✅ Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")
            print(f"\nTop 5 Allocations:")
            for alloc in portfolio['allocation'][:5]:
                print(f"  {alloc['stock']}: {alloc['weight_pct']:.2f}%")
            
            # Plot allocation
            plot_portfolio_allocation(portfolio)
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
    
    # Plot correlation matrix
    print("\n📊 Generating correlation matrix...")
    plot_correlation_matrix(stock_data_for_portfolio)
else:
    print("⚠️ Minimum 2 saham diperlukan untuk portfolio optimization")

# ✅ RISK METRICS VISUALIZATION
print("\n📊 Generating risk metrics visualization...")
plot_risk_metrics(df_results)

# ✅ SAVE RESULTS
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'output/reports/analysis_{timestamp}.csv'
df_results.to_csv(csv_path, index=False)

# ✅ SAVE SUMMARY REPORT
summary_path = f'output/reports/summary_{timestamp}.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write(f"{TARGET_SECTOR.upper()} SECTOR DECISION SUPPORT SYSTEM\n")
    f.write("WITH 5-MODEL CONSENSUS & PORTFOLIO OPTIMIZATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Stocks Analyzed: {len(stock_list)}\n")
    f.write(f"Successful Analysis: {len(results)}\n")
    f.write(f"\nMODELS USED: ARIMA, LSTM, GRU, SVR, XGBoost\n")
    f.write(f"DECISION BASIS: Consensus Mean of all 5 models\n\n")
    
    f.write("="*80 + "\n")
    f.write("STOCK ANALYSIS RESULTS\n")
    f.write("="*80 + "\n")
    # Select columns for report
    report_cols = ['Stock', 'Current', 'Forecast', 'Change%', 'Decision', 'Risk_Level', 'Sharpe_Ratio']
    f.write(df_results[report_cols].to_string(index=False))
    f.write("\n\n")
    
    # Profit simulation
    f.write("="*80 + "\n")
    f.write("PROFIT SIMULATION\n")
    f.write("="*80 + "\n")
    f.write(f"Total Profit: Rp {profit:,.2f}\n")
    f.write(f"ROI: {roi:.2f}%\n\n")
    
    # Risk summary
    f.write("="*80 + "\n")
    f.write("RISK SUMMARY\n")
    f.write("="*80 + "\n")
    risk_summary = df_results.groupby('Risk_Level').size()
    for level, count in risk_summary.items():
        f.write(f"{level} Risk: {count} stocks ({count/len(results)*100:.1f}%)\n")
    f.write(f"\nAverage Sharpe Ratio: {df_results['Sharpe_Ratio'].mean():.2f}\n")
    
    # Top performers
    f.write("="*80 + "\n")
    f.write("TOP 5 RECOMMENDATIONS (by Sharpe Ratio)\n")
    f.write("="*80 + "\n")
    top_5 = df_results.nlargest(5, 'Sharpe_Ratio')[['Stock', 'Decision', 'Change%', 'Sharpe_Ratio', 'Risk_Level']]
    f.write(top_5.to_string(index=False))
    f.write("\n\n")
    
    # Portfolio optimization results
    if portfolio_results:
        f.write("="*80 + "\n")
        f.write("PORTFOLIO OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        for method, portfolio in portfolio_results.items():
            f.write(f"\n{portfolio_methods[method]}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Expected Return: {portfolio['expected_return']*100:.2f}%\n")
            f.write(f"Volatility: {portfolio['volatility']*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}\n")
            f.write(f"\nAllocations (Top 5):\n")
            for alloc in portfolio['allocation'][:5]:
                f.write(f"  {alloc['stock']}: {alloc['weight_pct']:.2f}%\n")
            f.write("\n")
    
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

# ✅ Print ke console
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(df_results[['Stock', 'Decision', 'Change%', 'Sharpe_Ratio', 
                    'Risk_Level', 'Stop_Loss', 'Take_Profit']].to_string(index=False))
print("="*80)
print(f"\n💰 Total Profit: Rp {profit:,.2f}")
print(f"📈 ROI: {roi:.2f}%")
print("\n📊 Risk Summary:")
risk_summary = df_results.groupby('Risk_Level').size()
for level, count in risk_summary.items():
    print(f"  {level} Risk: {count} stocks")
print(f"\n⭐ Average Sharpe Ratio: {df_results['Sharpe_Ratio'].mean():.2f}")

if portfolio_results:
    print("\n" + "="*80)
    print("PORTFOLIO RECOMMENDATIONS")
    print("="*80)
    for method, portfolio in portfolio_results.items():
        print(f"\n{portfolio_methods[method]}:")
        print(f"  Expected Return: {portfolio['expected_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")

print("\n" + "="*80)
print(f"✅ CSV Report saved: {csv_path}")
print(f"✅ Summary Report saved: {summary_path}")
print(f"✅ All plots saved in: output/plots/")
print("="*80)