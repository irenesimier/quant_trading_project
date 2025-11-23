import torch
from torch.utils.data import Dataset

class StockDiffDataset(Dataset):
    """
    Creates sequences from stock data to predict difference in returns.
    """
    def __init__(self, df, features, target_col='ret_diff', seq_len=20):
        self.seq_len = seq_len
        self.features = features
        self.target_col = target_col
        self.data = df

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = self.data[self.features].iloc[idx:idx+self.seq_len].values
        target = self.data[self.target_col].iloc[idx+self.seq_len]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
