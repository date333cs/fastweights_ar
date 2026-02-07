# src/models.py
from __future__ import annotations

import torch
import torch.nn as nn


class IRNN(nn.Module):
    """
    Simple ReLU RNN baseline (IRNN).
    - Embedding -> ReLU RNN with W initialized as 0.05 * I
    - Readout from final hidden state -> 10 classes
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.U = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)

        with torch.no_grad():
            self.W.weight.zero_()
            self.W.weight += 0.05 * torch.eye(hidden_dim)
            self.W.bias.zero_()

        self.relu = nn.ReLU()
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        x = self.emb(x_ids)  # (B, T, emb_dim)
        h = torch.zeros(B, self.W.in_features, device=x_ids.device, dtype=x.dtype)
        for t in range(T):
            h = self.relu(self.W(h) + self.U(x[:, t, :]))
        return self.readout(h)


class LSTMClassifier(nn.Module):
    """
    LSTM baseline.
    - Embedding -> (single-layer) LSTM -> readout from final hidden state -> 10 classes
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(x_ids)  # (B, T, emb_dim)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]     # (B, hidden_dim)
        return self.readout(h_last)

