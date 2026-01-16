import matplotlib.pyplot as plt
import os

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
    plt.show()
    plt.close()