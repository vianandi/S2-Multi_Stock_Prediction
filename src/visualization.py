import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

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
    
    # ✅ TAMBAHAN: Auto-save plot
    os.makedirs('output/plots', exist_ok=True)
    save_path = f'output/plots/{stock_code}_forecast.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {save_path}")
    
    # Show plot (untuk interactive mode)
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

import pandas as pd

def plot_model_comparison(df_comparison, save_dir='output/plots'):
    """
    Visualisasi perbandingan 3 model (ARIMA, LSTM, Ensemble)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter only valid data
    df_valid = df_comparison.dropna(subset=['arima_forecast', 'lstm_forecast', 'ensemble_forecast'])
    
    if len(df_valid) == 0:
        print("⚠️ No valid data for comparison visualization")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Forecast Comparison - Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df_valid))
    width = 0.25
    
    ax1.bar(x - width, df_valid['arima_forecast'], width, label='ARIMA', alpha=0.8, color='#FF6B6B')
    ax1.bar(x, df_valid['lstm_forecast'], width, label='LSTM', alpha=0.8, color='#4ECDC4')
    ax1.bar(x + width, df_valid['ensemble_forecast'], width, label='Ensemble', alpha=0.8, color='#45B7D1')
    ax1.plot(x, df_valid['current_price'], 'ko-', linewidth=2, markersize=8, label='Current Price')
    
    ax1.set_xlabel('Stock')
    ax1.set_ylabel('Price (IDR)')
    ax1.set_title('Forecast Comparison: ARIMA vs LSTM vs Ensemble', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_valid['stock'], rotation=45, ha='right')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Percentage Change Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(df_valid))
    width = 0.25
    
    ax2.bar(x - width, df_valid['arima_change_pct'], width, label='ARIMA', alpha=0.8, color='#FF6B6B')
    ax2.bar(x, df_valid['lstm_change_pct'], width, label='LSTM', alpha=0.8, color='#4ECDC4')
    ax2.bar(x + width, df_valid['ensemble_change_pct'], width, label='Ensemble', alpha=0.8, color='#45B7D1')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Stock')
    ax2.set_ylabel('Change (%)')
    ax2.set_title('Predicted Price Change (%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_valid['stock'], rotation=45, ha='right')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Execution Time Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    
    avg_times = {
        'ARIMA': df_comparison['arima_time'].mean(),
        'LSTM': df_comparison['lstm_time'].mean(),
        'Ensemble': df_comparison['ensemble_time'].mean()
    }
    
    colors_time = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax3.bar(avg_times.keys(), avg_times.values(), color=colors_time, alpha=0.8)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Average Execution Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Agreement - Scatter Plot
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(df_valid['arima_change_pct'], df_valid['lstm_change_pct'], 
               s=100, alpha=0.6, c='purple', edgecolors='black')
    
    # Add diagonal line (perfect agreement)
    lims = [
        np.min([ax4.get_xlim(), ax4.get_ylim()]),
        np.max([ax4.get_xlim(), ax4.get_ylim()]),
    ]
    ax4.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    
    ax4.set_xlabel('ARIMA Change (%)')
    ax4.set_ylabel('LSTM Change (%)')
    ax4.set_title('Model Agreement: ARIMA vs LSTM', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Calculate and display correlation
    corr = df_valid['arima_change_pct'].corr(df_valid['lstm_change_pct'])
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax4.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Prediction Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    
    data_to_plot = [
        df_valid['arima_change_pct'].dropna(),
        df_valid['lstm_change_pct'].dropna(),
        df_valid['ensemble_change_pct'].dropna()
    ]
    labels = ['ARIMA', 'LSTM', 'Ensemble']
    colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_ylabel('Change (%)')
    ax5.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MODEL COMPARISON DASHBOARD', fontsize=16, fontweight='bold', y=0.995)
    
    save_path = f'{save_dir}/model_comparison_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Model comparison dashboard saved: {save_path}")
    plt.close()

def plot_model_performance_summary(analysis, save_dir='output/plots'):
    """
    Summary visualization dari analysis results
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
    
    colors_success = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(models, success_rates, color=colors_success, alpha=0.7)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Model Success Rate', fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Speed Comparison
    ax2 = axes[0, 1]
    if analysis['speed_ranking']:
        speed_models = [item[0] for item in analysis['speed_ranking']]
        speed_times = [item[1] for item in analysis['speed_ranking']]
        
        colors_speed = []
        for model in speed_models:
            if 'ARIMA' in model.upper():
                colors_speed.append('#FF6B6B')
            elif 'LSTM' in model.upper():
                colors_speed.append('#4ECDC4')
            else:
                colors_speed.append('#45B7D1')
        
        bars2 = ax2.barh(speed_models, speed_times, color=colors_speed, alpha=0.7)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Execution Speed (Lower is Better)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}s',
                    ha='left', va='center', fontweight='bold')
    
    # 3. Average Prediction
    ax3 = axes[1, 0]
    if analysis['prediction_stats']:
        pred_models = list(analysis['prediction_stats'].keys())
        pred_means = [analysis['prediction_stats'][m]['mean'] for m in pred_models]
        
        colors_pred = ['#FF6B6B' if 'arima' in m else '#4ECDC4' if 'lstm' in m else '#45B7D1' 
                      for m in pred_models]
        
        bars3 = ax3.bar(pred_models, pred_means, color=colors_pred, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_ylabel('Average Change (%)')
        ax3.set_title('Average Predicted Change', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.2f}%',
                    ha='center', va='bottom' if height > 0 else 'top', 
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
        ax4.set_xticklabels(models_signal)
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
    Visualize prediction vs actual results with error analysis
    
    Parameters:
    -----------
    df_validation : DataFrame
        Contains columns: stock, actual, arima_pred, lstm_pred, ensemble_pred, *_error, etc.
    metrics_summary : dict
        Contains MAE, RMSE, MAPE for each model
    save_dir : str
        Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme for models
    colors = {
        'arima': '#FF6B6B',
        'lstm': '#4ECDC4',
        'ensemble': '#45B7D1',
        'actual': '#2D3436'
    }
    
    # 1. Predicted vs Actual - ARIMA
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df_validation['actual'], df_validation['arima_pred'], 
               alpha=0.6, s=100, color=colors['arima'], edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(df_validation['actual'].min(), df_validation['arima_pred'].min())
    max_val = max(df_validation['actual'].max(), df_validation['arima_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Price (IDR)', fontweight='bold')
    ax1.set_ylabel('Predicted Price (IDR)', fontweight='bold')
    ax1.set_title('ARIMA: Predicted vs Actual', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² score
    from scipy.stats import pearsonr
    r, _ = pearsonr(df_validation['actual'], df_validation['arima_pred'])
    r_squared = r**2
    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}\nMAPE = {metrics_summary["arima"]["MAPE"]:.2f}%',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Predicted vs Actual - LSTM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df_validation['actual'], df_validation['lstm_pred'], 
               alpha=0.6, s=100, color=colors['lstm'], edgecolor='black', linewidth=0.5)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price (IDR)', fontweight='bold')
    ax2.set_ylabel('Predicted Price (IDR)', fontweight='bold')
    ax2.set_title('LSTM: Predicted vs Actual', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    r, _ = pearsonr(df_validation['actual'], df_validation['lstm_pred'])
    r_squared = r**2
    ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}\nMAPE = {metrics_summary["lstm"]["MAPE"]:.2f}%',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Predicted vs Actual - Ensemble
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df_validation['actual'], df_validation['ensemble_pred'], 
               alpha=0.6, s=100, color=colors['ensemble'], edgecolor='black', linewidth=0.5)
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    
    ax3.set_xlabel('Actual Price (IDR)', fontweight='bold')
    ax3.set_ylabel('Predicted Price (IDR)', fontweight='bold')
    ax3.set_title('Ensemble: Predicted vs Actual', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    r, _ = pearsonr(df_validation['actual'], df_validation['ensemble_pred'])
    r_squared = r**2
    ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}\nMAPE = {metrics_summary["ensemble"]["MAPE"]:.2f}%',
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Error Distribution - ARIMA
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df_validation['arima_error'], bins=20, color=colors['arima'], alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Prediction Error (%)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('ARIMA: Error Distribution', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Error Distribution - LSTM
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df_validation['lstm_error'], bins=20, color=colors['lstm'], alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.set_xlabel('Prediction Error (%)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('LSTM: Error Distribution', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Error Distribution - Ensemble
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df_validation['ensemble_error'], bins=20, color=colors['ensemble'], alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax6.set_xlabel('Prediction Error (%)', fontweight='bold')
    ax6.set_ylabel('Frequency', fontweight='bold')
    ax6.set_title('Ensemble: Error Distribution', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Per-Stock Error Comparison (Top 10 by actual price)
    ax7 = fig.add_subplot(gs[2, :2])
    df_sorted = df_validation.nlargest(10, 'actual')
    
    x = np.arange(len(df_sorted))
    width = 0.25
    
    bars1 = ax7.bar(x - width, df_sorted['arima_abs_error'], width, 
                    label='ARIMA', color=colors['arima'], alpha=0.8)
    bars2 = ax7.bar(x, df_sorted['lstm_abs_error'], width, 
                    label='LSTM', color=colors['lstm'], alpha=0.8)
    bars3 = ax7.bar(x + width, df_sorted['ensemble_abs_error'], width, 
                    label='Ensemble', color=colors['ensemble'], alpha=0.8)
    
    ax7.set_ylabel('Absolute Error (%)', fontweight='bold')
    ax7.set_xlabel('Stock Code', fontweight='bold')
    ax7.set_title('Per-Stock Absolute Error (Top 10 by Price)', fontweight='bold', fontsize=12)
    ax7.set_xticks(x)
    ax7.set_xticklabels(df_sorted['stock'], rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Model Performance Comparison (MAPE)
    ax8 = fig.add_subplot(gs[2, 2])
    models = list(metrics_summary.keys())
    mape_values = [metrics_summary[m]['MAPE'] for m in models]
    mae_values = [metrics_summary[m]['MAE'] for m in models]
    
    model_colors = [colors[m] for m in models]
    
    bars = ax8.barh(models, mape_values, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax8.set_xlabel('MAPE (%)', fontweight='bold')
    ax8.set_title('Overall Model Accuracy (Lower is Better)', fontweight='bold', fontsize=12)
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, mape, mae) in enumerate(zip(bars, mape_values, mae_values)):
        width = bar.get_width()
        ax8.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{mape:.2f}%\n(MAE: Rp{mae:,.0f})',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Highlight best model
    best_idx = mape_values.index(min(mape_values))
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('gold')
    
    plt.suptitle('VALIDATION RESULTS: PREDICTED vs ACTUAL (IDX - 9 Jan 2023)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    save_path = f'{save_dir}/validation_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Validation Chart: {save_path}")
    plt.close()