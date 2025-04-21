import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """Efficient Channel Attention Block."""
    def __init__(self, k_size=3):
        super(ECABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding="same", bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        batch, channels, seq_len = x.shape
        gap = self.global_avg_pool(x)
        gap = gap.permute(0, 2, 1)
        conv = self.conv(gap)
        conv = self.sigmoid(conv)
        conv = conv.permute(0, 2, 1)
        return x * conv


class StockPredictionModel(nn.Module):
    """Stock prediction model using Conv1D, BiLSTM, and ECA."""
    def __init__(self, sequence_length, n_features):
        super(StockPredictionModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=1, padding="same")
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        self.eca = ECABlock(k_size=3)

        self.flatten = nn.Flatten()
        # Corrected flatten size calculation: BiLSTM output is (batch, seq_len, 2 * hidden_size)
        # After ECA and permute back to (batch, seq_len, 2*hidden_size), flatten combines seq_len and features
        self.fc1 = nn.Linear(64 * 2 * sequence_length, 128) # Corrected from original script
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        x = self.relu(self.conv1(x)) # -> (batch, 64, seq_len)

        x = x.permute(0, 2, 1)  # -> (batch, seq_len, 64)
        x, _ = self.bilstm(x) # -> (batch, seq_len, 128) (64 * 2 directions)

        # Permute for ECA: ECA expects (batch, channels, seq_len)
        # Here, channels = BiLSTM output features = 128
        x = x.permute(0, 2, 1) # -> (batch, 128, seq_len)
        x = self.eca(x) # -> (batch, 128, seq_len)

        # Flatten expects (batch, *)
        # Option 1: Flatten directly (if ECA output shape is suitable)
        # x = self.flatten(x) # -> (batch, 128 * seq_len) - Mismatches fc1 input

        # Option 2: Permute back before flatten (Matches original intent more closely?)
        # Let's assume the original flatten size intended features * seq_len
        # The original code had x.permute(0, 2, 1) before flatten, let's rethink that.
        # After BiLSTM: (batch, seq_len, 128)
        # After ECA (if applied on feature dim 128): (batch, 128, seq_len)

        # The original code applied ECA on the output of BiLSTM after permutation
        # BiLSTM output: (batch, seq_len, 128)
        # Permute for ECA: (batch, 128, seq_len)
        # ECA output: (batch, 128, seq_len)
        # Original Flatten was after ECA: flattened (batch, 128, seq_len) -> (batch, 128 * seq_len)
        # Original fc1 Linear layer: Linear(64 * 2 * sequence_length, 128) -> Linear(128 * sequence_length, 128)

        # Let's stick to the original logic's implied shape for fc1
        x = self.flatten(x) # -> (batch, 128 * seq_len)

        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # -> (batch, 1)
        return x
