import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = "test_predictions.csv"

# Load predictions
pred_df = pd.read_csv(PRED_CSV, parse_dates=['date'])

# Plot predicted vs true return difference
plt.figure(figsize=(14,6))
plt.plot(pred_df['date'], pred_df['true_ret_diff'], label='True ret_diff', color='blue')
plt.plot(pred_df['date'], pred_df['predicted_ret_diff'], label='Predicted ret_diff', color='red', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Return Difference (XLK - QTEC)")
plt.title("Predicted vs True Return Difference on Test Set")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()  # Format dates nicely
plt.show()
