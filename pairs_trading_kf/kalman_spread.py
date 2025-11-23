import argparse
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", required=True)
parser.add_argument("--raw_csv", type=str, required=True)
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
parser.add_argument("--spread_csv", type=str, default="kalman_spread.csv")
args = parser.parse_args()

TICKERS = args.tickers
RAW_CSV = args.raw_csv
START_DATE = args.start_date
END_DATE = args.end_date
SPREAD_CSV = args.spread_csv

# Load raw prices and restructure
df = pd.read_csv(RAW_CSV, parse_dates=['date'])
df = df[df['date'].between(START_DATE, END_DATE)]
price_df = df.pivot(index='date', columns='ticker', values='prc')
available_tickers = [t for t in TICKERS if t in price_df.columns]
if len(available_tickers) < 2:
    raise ValueError(f"Not enough tickers with data. Available: {available_tickers}")
price_df = price_df[available_tickers].interpolate().ffill().bfill()

# Kalman filter with 2 tickers
y1 = price_df[available_tickers[0]].values
y2 = price_df[available_tickers[1]].values
delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
kf = KalmanFilter(
    n_dim_obs=1,
    n_dim_state=2,
    initial_state_mean=[0, 0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=np.vstack([y2, np.ones(len(y2))]).T[:, np.newaxis, :],
    observation_covariance=1.0,
    transition_covariance=trans_cov
)
state_means, state_covs = kf.filter(y1)
beta = state_means[:, 0]
spread = y1 - beta * y2

# Save results
result = pd.DataFrame({
    'date': price_df.index,
    f'{available_tickers[0]}': y1,
    f'{available_tickers[1]}': y2,
    'beta': beta,
    'spread': spread
})
result.to_csv(SPREAD_CSV, index=False)
print(f"Kalman spread saved to {SPREAD_CSV}")
