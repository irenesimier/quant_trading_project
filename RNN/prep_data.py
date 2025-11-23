import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CSV_FILE = "daily_prices_raw.csv"
OUT_FILE = "daily_prices_preprocessed.csv"

data = pd.read_csv(CSV_FILE)
data['date'] = pd.to_datetime(data['date'])

# Pivot
data_pivot = data.pivot(index='date', columns='ticker', values=['prc', 'ret'])
data_pivot.columns = ['_'.join(col) for col in data_pivot.columns]
data_pivot = data_pivot.dropna()

# Compute return difference
data_pivot['ret_diff'] = data_pivot['ret_XLK'] - data_pivot['ret_QTEC']

# Features to scale
features = ['prc_XLK', 'ret_XLK', 'prc_QTEC', 'ret_QTEC']
scaler = MinMaxScaler()
data_pivot[features] = scaler.fit_transform(data_pivot[features])

# Save preprocessed CSV
data_pivot.to_csv(OUT_FILE)
print(f"Preprocessed data saved to {OUT_FILE}")
