import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import DiffLSTM
from datasets import StockDiffDataset

SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
TRAIN_RATIO = 0.8
CSV_FILE = "daily_prices_preprocessed.csv"
FEATURES = ['prc_XLK', 'ret_XLK', 'prc_QTEC', 'ret_QTEC']
TARGET = 'ret_diff'
PRED_CSV = "test_predictions.csv"

# Load preprocessed data
data = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)

# Chronological train/test split
n_train = int(len(data) * TRAIN_RATIO)
train_df = data.iloc[:n_train]
test_df = data.iloc[n_train:]

# Create datasets
train_dataset = StockDiffDataset(train_df, FEATURES, target_col=TARGET, seq_len=SEQ_LEN)
test_dataset = StockDiffDataset(test_df, FEATURES, target_col=TARGET, seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffLSTM(input_size=len(FEATURES)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for seq, target in train_loader:
        seq, target = seq.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.6f}")

# Testing and saving predictions
model.eval()
preds, trues, dates = [], [], []
test_dates = test_df.index[SEQ_LEN:]

with torch.no_grad():
    for i, (seq, target) in enumerate(test_dataset):
        seq = seq.unsqueeze(0).float().to(device)
        pred = model(seq)
        preds.append(pred.item())
        trues.append(target.item())
        dates.append(test_dates[i])

pred_df = pd.DataFrame({
    'date': dates,
    'true_ret_diff': trues,
    'predicted_ret_diff': preds
})
pred_df.to_csv(PRED_CSV, index=False)
print(f"Test predictions saved to {PRED_CSV}")
