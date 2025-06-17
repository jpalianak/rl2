import torch.nn as nn


class LSTM_QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128, num_layers=1):
        super(LSTM_QNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size,
                            num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Tomar la salida del último paso de la secuencia
        out = lstm_out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
