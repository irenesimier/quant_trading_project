import subprocess
import sys

CONFIG = {
    "TICKERS": ["XLK", "QTEC"],
    "START_DATE": "2007-01-01",
    "END_DATE": "2023-12-31",
    "ROLL_WINDOW": 10,
    "ENTRY_THRESH": 1.5,
    "ML_ESTIMATORS": 200,
    "TRAIN_RATIO": 0.75,
    "LEVERAGE": 1.0,
    "FEE": 0.0005,
    "RAW_CSV": "daily_prices_raw.csv",
    "SPREAD_CSV": "kalman_spread.csv",
    "FEATURES_CSV": "features_target.csv",
}

def run_script(script_name, extra_args=None):
    """Run a Python script with optional arguments and exit if it fails."""
    cmd = [sys.executable, script_name]
    if extra_args:
        cmd += extra_args
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {script_name} failed!")
        sys.exit(1)

if __name__ == "__main__":
    # data_fetch.py
    # args = [
    #     "--tickers",
    #     *CONFIG["TICKERS"],
    #     f"--start_date={CONFIG['START_DATE']}",
    #     f"--end_date={CONFIG['END_DATE']}",
    #     f"--raw_csv={CONFIG['RAW_CSV']}"
    # ]
    # run_script("data_fetch.py", extra_args=args)

    # kalman_spread.py
    args = [
        "--raw_csv", CONFIG["RAW_CSV"],
        "--spread_csv", CONFIG["SPREAD_CSV"],
        "--start_date", CONFIG["START_DATE"],
        "--end_date", CONFIG["END_DATE"],
        "--tickers"
    ] + CONFIG["TICKERS"]

    run_script("kalman_spread.py", extra_args=args)


    # features_ml.py
    args = [
        f"--spread_csv={CONFIG['SPREAD_CSV']}",
        f"--features_csv={CONFIG['FEATURES_CSV']}",
        f"--roll_window={CONFIG['ROLL_WINDOW']}",
        f"--entry_thresh={CONFIG['ENTRY_THRESH']}"
    ]
    run_script("features_ml.py", extra_args=args)

    # ml_backtest.py
    args = [
        f"--features_csv={CONFIG['FEATURES_CSV']}",
        f"--ml_estimators={CONFIG['ML_ESTIMATORS']}",
        f"--train_ratio={CONFIG['TRAIN_RATIO']}",
        f"--leverage={CONFIG['LEVERAGE']}",
        f"--fee={CONFIG['FEE']}"
    ]
    run_script("ml_backtest.py", extra_args=args)

    print("\nAll steps completed successfully!")
