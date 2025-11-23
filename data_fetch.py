import argparse
import pandas as pd
import wrds

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", required=True)
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
parser.add_argument("--raw_csv", type=str, required=True)
args = parser.parse_args()

TICKERS = args.tickers
START_DATE = args.start_date
END_DATE = args.end_date
OUT_RAW_CSV = args.raw_csv

# Connect to WRDS
conn = wrds.Connection()

# Find permnos
permnos_df = conn.raw_sql(f"""
    SELECT permno, ticker
    FROM crsp.msenames
    WHERE ticker IN {tuple(TICKERS)}
""")
if permnos_df.empty:
    raise ValueError(f"No permnos found for the tickers: {TICKERS}")

# Keep the latest permno for each ticker
permnos_df = permnos_df.sort_values(['ticker', 'permno']).drop_duplicates('ticker', keep='last')
permnos = permnos_df['permno'].tolist()
permno_map = dict(zip(permnos, permnos_df['ticker']))

# Fetch daily stock data
data = conn.raw_sql(f"""
    SELECT date, permno, prc, vol, ret
    FROM crsp.dsf
    WHERE permno IN {tuple(permnos)}
    AND date BETWEEN '{START_DATE}' AND '{END_DATE}'
    ORDER BY date
""")

# Add ticker column
data['ticker'] = data['permno'].map(permno_map)
data['date'] = pd.to_datetime(data['date'])
data['prc'] = data['prc'].abs()

data.to_csv(OUT_RAW_CSV, index=False)
print(f"Raw daily data saved to {OUT_RAW_CSV}")
