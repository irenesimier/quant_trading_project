from torch import nn

class DiffLSTM(nn.Module):
    """
    LSTM model to predict return difference between two stocks.
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last timestep
        out = self.fc(out)
        return out.squeeze()