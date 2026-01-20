import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Standard colors for models
COLORS = {
    'arima': '#FF6B6B',   # Red
    'lstm': '#4ECDC4',    # Teal
    'gru': '#FFE66D',     # Yellow
    'svr': '#1A535C',     # Dark Blue
    'xgb': '#FF9F1C',     # Orange
    'ensemble': '#45B7D1',# Legacy Blue
    'actual': '#2D3436'   # Dark Gray
}

def get_color(model_name):
    return COLORS.get(model_name.lower(), '#95A5A6') # Return gray if unknown

def plot_stock(df, forecast, stock_code):
    """
    Visualisasi harga historis + forecast dengan auto-save
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(df['close'].values[-100:], label='Historical Close', color='blue')
    
    # Plot forecast point
    plt.axhline(y=forecast, color='red', linestyle='--', label=f'Forecast: {forecast:.2f}')
    
    plt.title(f'{stock_code} - Stock Price Analysis')
    plt.xlabel('Days')
    plt.ylabel('Price (IDR)')
    plt.legend()
    plt.grid(True)
    
    # Auto-save plot
    os.makedirs('output/plots', exist_ok=True)
    save_path = f'output/plots/{stock_code}_forecast.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {save_path}")
    plt.close()

def plot_technical_indicators(df, stock_code):
    """
    Plot technical indicators: Price, MACD, RSI, Bollinger Bands
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 1. Price with Moving Averages and Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1.5)
    ax1.plot(df.index, df['MA20'], label='MA20', color='blue', alpha=0.7)
    ax1.plot(df.index, df['MA50'], label='MA50', color='red', alpha=0.7)
    ax1.fill_between(df.index, df['BB_High'], df['BB_Low'], alpha=0.2, color='gray', label='Bollinger Bands')
    ax1.set_title(f'{stock_code} - Price & Moving Averages')
    ax1.set_ylabel('Price (IDR)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. MACD
    ax2 = axes[1]
    ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax2.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
    ax2.bar(df.index, df['MACD_Diff'], label='Histogram', color='gray', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_title('MACD (Moving Average Convergence Divergence)')
    ax2.set_ylabel('MACD')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. RSI
    ax3 = axes[2]
    ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1.5)
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_title('RSI (Relative Strength Index)')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Volume with OBV
    ax4 = axes[3]
    ax4_twin = ax4.twinx()
    ax4.bar(df.index, df['volume'], label='Volume', color='lightblue', alpha=0.5)
    ax4_twin.plot(df.index, df['OBV'], label='OBV', color='orange', linewidth=2)
    ax4.set_title('Volume & On-Balance Volume (OBV)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Volume')
    ax4_twin.set_ylabel('OBV')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'output/plots/{stock_code}_technical_indicators.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Technical indicators plot saved: {save_path}")
    plt.close()

def plot_risk_metrics(results_df):
    """
    Visualisasi risk metrics untuk semua saham
    """
    if 'Volatility' not in results_df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Risk-Return Scatter
    ax1 = axes[0, 0]
    scatter = ax1.scatter(results_df['Volatility'], results_df['Change%'], 
                         c=results_df['Sharpe_Ratio'], cmap='RdYlGn', 
                         s=100, alpha=0.6, edgecolors='black')
    for idx, row in results_df.iterrows():
        ax1.annotate(row['Stock'], (row['Volatility'], row['Change%']), 
                    fontsize=8, alpha=0.7)
    ax1.set_xlabel('Volatility (Annual)')
    ax1.set_ylabel('Expected Return (%)')
    ax1.set_title('Risk-Return Profile')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
    
    # 2. Sharpe Ratio Bar Chart
    ax2 = axes[0, 1]
    colors = ['green' if x > 1 else 'orange' if x > 0 else 'red' 
              for x in results_df['Sharpe_Ratio']]
    results_df_sorted = results_df.sort_values('Sharpe_Ratio', ascending=True)
    ax2.barh(results_df_sorted['Stock'], results_df_sorted['Sharpe_Ratio'], color=colors)
    ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Sharpe=1')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Maximum Drawdown
    ax3 = axes[1, 0]
    results_df_sorted_dd = results_df.sort_values('Max_Drawdown', ascending=True)
    colors_dd = ['red' if x < -0.3 else 'orange' if x < -0.2 else 'green' 
                 for x in results_df_sorted_dd['Max_Drawdown']]
    ax3.barh(results_df_sorted_dd['Stock'], results_df_sorted_dd['Max_Drawdown'] * 100, 
             color=colors_dd)
    ax3.axvline(x=-20, color='black', linestyle='--', alpha=0.5, label='20% threshold')
    ax3.set_xlabel('Max Drawdown (%)')
    ax3.set_title('Maximum Drawdown Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Risk Level Distribution
    ax4 = axes[1, 1]
    risk_counts = results_df['Risk_Level'].value_counts()
    colors_risk = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    ax4.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            colors=[colors_risk.get(x, 'gray') for x in risk_counts.index],
            startangle=90)
    ax4.set_title('Risk Level Distribution')
    
    plt.tight_layout()
    save_path = 'output/plots/risk_metrics_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Risk metrics plot saved: {save_path}")
    plt.close()

def plot_portfolio_allocation(portfolio_result):
    """
    Visualisasi portfolio allocation
    """
    allocation = portfolio_result['allocation']
    if not allocation:
        return
    
    stocks = [a['stock'] for a in allocation]
    weights = [a['weight_pct'] for a in allocation]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    ax1.pie(weights, labels=stocks, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f"Portfolio Allocation ({portfolio_result['method']})")
    
    # Bar chart
    ax2.barh(stocks, weights, color='steelblue')
    ax2.set_xlabel('Allocation (%)')
    ax2.set_title('Portfolio Weight Distribution')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = f"output/plots/portfolio_allocation_{portfolio_result['method']}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Portfolio allocation plot saved: {save_path}")
    plt.close()

def plot_correlation_matrix(stock_data_dict):
    """
    Plot correlation heatmap antar saham
    """
    from src.risk_management import calculate_returns
    
    # Prepare returns data
    returns_dict = {}
    for code, prices in stock_data_dict.items():
        returns_dict[code] = calculate_returns(prices)
    
    # Find minimum length
    min_len = min(len(r) for r in returns_dict.values())
    
    # Create DataFrame with equal length
    returns_df = pd.DataFrame({
        code: returns[-min_len:] 
        for code, returns in returns_dict.items()
    })
    
    # Calculate correlation
    corr_matrix = returns_df.corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Stock Returns Correlation Matrix')
    plt.tight_layout()
    
    save_path = 'output/plots/correlation_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation matrix plot saved: {save_path}")
    plt.close()

def plot_model_comparison(df_comparison, save_dir='output/plots'):
    """
    Visualisasi perbandingan Model yang ada di df_comparison
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Detect available models from columns
    available_models = []
    for col in df_comparison.columns:
        if col.endswith('_forecast'):
            model_name = col.replace('_forecast', '')
            available_models.append(model_name)
    
    if not available_models:
        print("⚠️ No model forecast data found")
        return

    # Filter columns to check validity (at least one forecast must be present)
    check_cols = [f'{m}_forecast' for m in available_models]
    df_valid = df_comparison.dropna(subset=check_cols, how='all')
    
    if len(df_valid) == 0:
        print("⚠️ No valid data for comparison visualization")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # 1. Percentage Change Comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df_valid))
    width = 0.8 / len(available_models)
    
    for i, model in enumerate(available_models):
        offset = width * i - (width * len(available_models) / 2) + (width/2)
        col_name = f'{model}_change_pct'
        if col_name in df_valid.columns:
            ax1.bar(x + offset, df_valid[col_name], width, 
                   label=model.upper(), alpha=0.8, color=get_color(model))
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Stock')
    ax1.set_ylabel('Predicted Change (%)')
    ax1.set_title('Predicted Price Change (%) by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_valid['stock'], rotation=45, ha='right')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Execution Time Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    avg_times = {}
    for model in available_models:
        col_name = f'{model}_time'
        if col_name in df_valid.columns:
            avg_times[model.upper()] = df_valid[col_name].mean()
            
    if avg_times:
        colors = [get_color(m) for m in avg_times.keys()]
        bars = ax2.bar(avg_times.keys(), avg_times.values(), color=colors, alpha=0.8)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Average Execution Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Prediction Distribution (Boxplot)
    ax3 = fig.add_subplot(gs[1, 1])
    data_to_plot = []
    labels = []
    colors_box = []
    
    for model in available_models:
        col_name = f'{model}_change_pct'
        if col_name in df_valid.columns:
            data_to_plot.append(df_valid[col_name].dropna())
            labels.append(model.upper())
            colors_box.append(get_color(model))
    
    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Change (%)')
    ax3.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MODEL COMPARISON DASHBOARD', fontsize=16, fontweight='bold', y=0.995)
    
    save_path = f'{save_dir}/model_comparison_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Model comparison dashboard saved: {save_path}")
    plt.close()

def plot_model_performance_summary(analysis, save_dir='output/plots'):
    """
    Summary visualization dari analysis results (Auto-adaptive to available models)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Success Rate
    ax1 = axes[0, 0]
    models = list(analysis['successful_predictions'].keys())
    success_counts = list(analysis['successful_predictions'].values())
    total = analysis['total_stocks']
    success_rates = [count/total*100 for count in success_counts]
    
    colors_success = [get_color(m) for m in models]
    
    bars1 = ax1.bar([m.upper() for m in models], success_rates, color=colors_success, alpha=0.7)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Model Success Rate', fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Speed Comparison
    ax2 = axes[0, 1]
    if analysis['speed_ranking']:
        speed_models = [item[0] for item in analysis['speed_ranking']]
        speed_times = [item[1] for item in analysis['speed_ranking']]
        
        colors_speed = [get_color(m) for m in speed_models]
        
        bars2 = ax2.barh(speed_models, speed_times, color=colors_speed, alpha=0.7)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Execution Speed (Lower is Better)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}s', ha='left', va='center', fontweight='bold')
    
    # 3. Average Prediction
    ax3 = axes[1, 0]
    if analysis['prediction_stats']:
        pred_models = list(analysis['prediction_stats'].keys())
        pred_means = [analysis['prediction_stats'][m]['mean'] for m in pred_models]
        
        colors_pred = [get_color(m) for m in pred_models]
        
        bars3 = ax3.bar([m.upper() for m in pred_models], pred_means, color=colors_pred, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_ylabel('Average Change (%)')
        ax3.set_title('Average Predicted Change', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold')
    
    # 4. Bullish vs Bearish Signals
    ax4 = axes[1, 1]
    if analysis['prediction_stats']:
        models_signal = list(analysis['prediction_stats'].keys())
        bullish = [analysis['prediction_stats'][m]['bullish_count'] for m in models_signal]
        bearish = [analysis['prediction_stats'][m]['bearish_count'] for m in models_signal]
        
        x = np.arange(len(models_signal))
        width = 0.35
        
        ax4.bar(x - width/2, bullish, width, label='Bullish', color='green', alpha=0.7)
        ax4.bar(x + width/2, bearish, width, label='Bearish', color='red', alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Signal Distribution', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.upper() for m in models_signal])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MODEL PERFORMANCE SUMMARY', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'{save_dir}/model_performance_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Model performance summary saved: {save_path}")
    plt.close()


def plot_validation_results(df_validation, metrics_summary, save_dir='output/models'):
    """
    Visualize prediction vs actual results for ALL models
    Dynamically creates subplots based on available models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(metrics_summary.keys())
    n_models = len(models)
    
    # Dynamic Grid Layout
    # We want Scatter plots and Histogram plots for each model + Summary
    # Let's do a fixed layout that accommodates up to 6 models comfortably
    # 3 Columns: 
    # Row 1-2: Scatter Plots (Predicted vs Actual)
    # Row 3-4: Error Histograms
    # Row 5: Comparison Metrics
    
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
    
    # --- 1. SCATTER PLOTS (Predicted vs Actual) ---
    for i, model in enumerate(models):
        if i >= 6: break # Limit to 6 plots max for layout safety
        
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        pred_col = f'{model}_pred'
        if pred_col in df_validation.columns:
            # Plot data
            ax.scatter(df_validation['actual'], df_validation[pred_col], 
                       alpha=0.6, s=80, color=get_color(model), 
                       edgecolor='black', linewidth=0.5)
            
            # Perfect line
            min_val = min(df_validation['actual'].min(), df_validation[pred_col].min())
            max_val = max(df_validation['actual'].max(), df_validation[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
            
            # Metrics
            mape = metrics_summary[model]['MAPE']
            r, _ = pearsonr(df_validation['actual'], df_validation[pred_col])
            r2 = r**2
            
            ax.set_title(f'{model.upper()} (R²={r2:.2f}, MAPE={mape:.1f}%)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if col == 0: ax.set_ylabel('Predicted')
            if row == 1: ax.set_xlabel('Actual')

    # --- 2. ERROR HISTOGRAMS ---
    for i, model in enumerate(models):
        if i >= 6: break
        
        row = (i // 3) + 2 # Shift down 2 rows
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        err_col = f'{model}_error' # Note: In validation we saved 'error' as pct_error
        if err_col in df_validation.columns:
            ax.hist(df_validation[err_col], bins=15, color=get_color(model), 
                   alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{model.upper()} Error Dist.', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            if col == 0: ax.set_ylabel('Freq')
            if row == 3: ax.set_xlabel('Error (%)')

    # --- 3. SUMMARY METRICS (Bottom Row) ---
    
    # MAPE Comparison
    ax_mape = fig.add_subplot(gs[4, 0])
    mape_values = [metrics_summary[m]['MAPE'] for m in models]
    colors_bar = [get_color(m) for m in models]
    
    bars = ax_mape.barh([m.upper() for m in models], mape_values, color=colors_bar, alpha=0.8)
    ax_mape.set_xlabel('MAPE (%) - Lower is Better')
    ax_mape.set_title('Model Accuracy Comparison', fontweight='bold')
    
    # Highlight best
    best_idx = mape_values.index(min(mape_values))
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('gold')
    
    # MAE Comparison
    ax_mae = fig.add_subplot(gs[4, 1])
    mae_values = [metrics_summary[m]['MAE'] for m in models]
    ax_mae.barh([m.upper() for m in models], mae_values, color=colors_bar, alpha=0.8)
    ax_mae.set_xlabel('MAE (IDR) - Lower is Better')
    ax_mae.set_title('Mean Absolute Error', fontweight='bold')
    
    # Per-Stock Error Heatmap (Top 10 Stocks)
    # Using a simple bar chart instead of heatmap for clarity in this layout
    ax_stock = fig.add_subplot(gs[4, 2])
    
    # Find best model per stock
    model_wins = {m: 0 for m in models}
    for _, row in df_validation.iterrows():
        errors = {m: abs(row.get(f'{m}_error', 999)) for m in models}
        winner = min(errors, key=errors.get)
        if winner in model_wins:
            model_wins[winner] += 1
            
    ax_stock.pie(model_wins.values(), labels=[m.upper() for m in model_wins.keys()], 
                 colors=[get_color(m) for m in model_wins.keys()], autopct='%1.1f%%')
    ax_stock.set_title('Best Model by Stock Count', fontweight='bold')

    plt.suptitle('VALIDATION RESULTS: PREDICTED vs ACTUAL', 
                 fontsize=20, fontweight='bold', y=0.99)
    
    save_path = f'{save_dir}/validation_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Validation Chart: {save_path}")
    plt.close()