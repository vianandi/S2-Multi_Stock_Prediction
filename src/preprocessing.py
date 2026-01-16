import ta

def add_indicators(df):
    df['MA20'] = ta.trend.sma_indicator(df['Close'],20)
    df['RSI'] = ta.momentum.rsi(df['Close'],14)
    return df.dropna()