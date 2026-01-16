import pandas as pd

def load_stock(path):
    df = pd.read_csv(path)
    # Rename timestamp to Date for consistency
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')