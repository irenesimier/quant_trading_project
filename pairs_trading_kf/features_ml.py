import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--spread_csv", type=str, required=True)
parser.add_argument("--features_csv", type=str, required=True)
parser.add_argument("--roll_window", type=int, required=True)
parser.add_argument("--entry_thresh", type=float, required=True)
args = parser.parse_args()

SPREAD_CSV = args.spread_csv
FEATURES_CSV = args.features_csv
ROLL_WINDOW = args.roll_window
ENTRY_THRESH = args.entry_thresh

df = pd.read_csv(SPREAD_CSV, parse_dates=['date'])

df.rename(columns={
    df.columns[1]: 'X',  
    df.columns[2]: 'Y'   
}, inplace=True)

if 'spread' not in df.columns:
    df['spread'] = df['Y'] - df['X']

# Median + MAD z-score
spread_median = df['spread'].rolling(ROLL_WINDOW).median()
spread_mad = df['spread'].rolling(ROLL_WINDOW).apply(lambda x: np.median(np.abs(x - np.median(x))))
df['spread_z'] = (df['spread'] - spread_median) / spread_mad

# Features
df['ret_X'] = df['X'].pct_change()
df['ret_Y'] = df['Y'].pct_change()
df['spread_lag1'] = df['spread'].shift(1)
df['spread_lag2'] = df['spread'].shift(2)
df['ret_X_lag1'] = df['ret_X'].shift(1)
df['ret_Y_lag1'] = df['ret_Y'].shift(1)

# Target: only trade when z-score > threshold
df['target'] = np.where(df['spread_z'].shift(-1) > ENTRY_THRESH, -1,
                        np.where(df['spread_z'].shift(-1) < -ENTRY_THRESH, 1, 0))

df = df.dropna()
df.to_csv(FEATURES_CSV, index=False)
print(f"Features and target saved to {FEATURES_CSV}")