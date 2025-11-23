import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--features_csv", type=str, required=True)
parser.add_argument("--ml_estimators", type=int, required=True)
parser.add_argument("--train_ratio", type=float, required=True)
parser.add_argument("--leverage", type=float, required=True)
parser.add_argument("--fee", type=float, required=True)
args = parser.parse_args()

FEATURES_CSV = args.features_csv
ML_ESTIMATORS = args.ml_estimators
TRAIN_RATIO = args.train_ratio
LEVERAGE = args.leverage
FEE = args.fee

df = pd.read_csv(FEATURES_CSV, parse_dates=['date'])
features = ['spread','spread_z','spread_lag1','spread_lag2','ret_X','ret_Y','ret_X_lag1','ret_Y_lag1']
X_feat = df[features].values
y_target = df['target'].values

split = int(TRAIN_RATIO * len(df))
X_train, X_test = X_feat[:split], X_feat[split:]
y_train, y_test = y_target[:split], y_target[split:]
dates_test = df['date'].iloc[split:]
ret_X_test = df['ret_X'].iloc[split:]
ret_Y_test = df['ret_Y'].iloc[split:]

# ML as filter
clf = RandomForestClassifier(n_estimators=ML_ESTIMATORS, random_state=42)
clf.fit(X_train, y_train)
positions = clf.predict(X_test)

strategy_ret = positions * (ret_Y_test - ret_X_test) * LEVERAGE
strategy_ret -= FEE * np.abs(np.diff(np.concatenate([[0], positions])))
cum_strategy = (1 + strategy_ret).cumprod()
cum_X = (1 + ret_X_test).cumprod()
cum_Y = (1 + ret_Y_test).cumprod()

plt.figure(figsize=(12,6))
plt.plot(dates_test, cum_strategy, label='Kalman Filter Strategy')
plt.plot(dates_test, cum_Y, label='Y')
plt.plot(dates_test, cum_X, label='X')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Pairs Trading Backtest (Weekly + Robust Z-score)')
plt.grid(True)
plt.show()
