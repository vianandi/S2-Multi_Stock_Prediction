import ta
import pandas as pd

def add_indicators(df):
    """
    Tambahkan berbagai technical indicators:
    - Moving Averages (MA20, MA50, EMA12, EMA26)
    - Momentum Indicators (RSI, MACD, Stochastic)
    - Volatility Indicators (Bollinger Bands, ATR)
    - Trend Indicators (ADX)
    - Volume Indicators (OBV, MFI)
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Moving Averages
    df['MA20'] = ta.trend.sma_indicator(close, 20)
    df['MA50'] = ta.trend.sma_indicator(close, 50)
    df['EMA12'] = ta.trend.ema_indicator(close, 12)
    df['EMA26'] = ta.trend.ema_indicator(close, 26)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.rsi(close, 14)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Width'] = bollinger.bollinger_wband()
    
    # ATR (Average True Range) - Volatility
    df['ATR'] = ta.volatility.average_true_range(high, low, close, 14)
    
    # ADX (Average Directional Index) - Trend Strength
    adx = ta.trend.ADXIndicator(high, low, close)
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()
    
    # OBV (On-Balance Volume)
    df['OBV'] = ta.volume.on_balance_volume(close, volume)
    
    # MFI (Money Flow Index)
    df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, 14)
    
    # Signal generation based on indicators
    df['MA_Signal'] = (df['MA20'] > df['MA50']).astype(int)  # 1 = bullish, 0 = bearish
    df['MACD_Signal_Buy'] = (df['MACD'] > df['MACD_Signal']).astype(int)
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['BB_Position'] = (close - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])  # Position dalam BB (0-1)
    
    return df.dropna()