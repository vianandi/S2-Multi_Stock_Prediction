from src.data_loader import load_stock
from src.preprocessing import add_indicators
from src.decision import make_decision
from src.profit_simulation import simulate
from src.visualization import plot_stock
from src.model.arima_model import arima_forecast
from src.model.lstm_model import lstm_forecast
from src.model.ensemble import ensemble

import pandas as pd
import os
from datetime import datetime

# ✅ TAMBAHAN: Buat folder output
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

# PILIH OPSI SAHAM
# energy = ["ADRO","PTBA","ITMG","MEDC"]  # 4 saham
energy = ["ADRO","BYAN","ADMR","GEMS","ITMG","PTBA","PGAS","TCPI","DSSA","MEDC"]  # 10 saham
# energy = ["ADRO","BYAN","ADMR","GEMS","ITMG","PTBA","PGAS","TCPI","DSSA","MEDC","AKRA","MCOL","BUMI","INDY","BSSR"]  # 15 saham

results = []

print("="*60)
print("ENERGY SECTOR MULTI-STOCK DECISION SUPPORT SYSTEM")
print("="*60)
print(f"Analyzing {len(energy)} stocks...")
print()

for i, code in enumerate(energy, 1):
    print(f"[{i}/{len(energy)}] Processing {code}...")
    
    df = load_stock(f"dataset/daily/{code}.csv")
    df = add_indicators(df)
    prices = df['close'].values

    a = arima_forecast(prices)
    l = lstm_forecast(prices)
    e = ensemble(a, l)

    decision, change = make_decision(prices[-1], e)

    results.append({
        "Stock": code,
        "Current": prices[-1],
        "Forecast": e,
        "Change%": change,
        "Decision": decision
    })

    plot_stock(df, e, code)
    print(f"✅ {code} completed!\n")

# ✅ TAMBAHAN: Simulasi profit
profit, roi = simulate(results)

# ✅ TAMBAHAN: Save results ke CSV
df_results = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'output/reports/analysis_{timestamp}.csv'
df_results.to_csv(csv_path, index=False)

# ✅ TAMBAHAN: Save summary report
summary_path = f'output/reports/summary_{timestamp}.txt'
with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("ENERGY SECTOR MULTI-STOCK DECISION SUPPORT SYSTEM\n")
    f.write("="*60 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Stocks Analyzed: {len(energy)}\n\n")
    f.write("RESULTS:\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n" + "="*60 + "\n")
    f.write(f"Total Profit: Rp {profit:,.2f}\n")
    f.write(f"ROI: {roi:.2f}%\n")
    f.write("="*60 + "\n")

# ✅ Print ke console
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(df_results.to_string(index=False))
print("="*60)
print(f"Total Profit: Rp {profit:,.2f}")
print(f"ROI: {roi:.2f}%")
print("="*60)
print(f"\n✅ CSV Report saved: {csv_path}")
print(f"✅ Summary Report saved: {summary_path}")
print(f"✅ All plots saved in: output/plots/")
print("="*60)