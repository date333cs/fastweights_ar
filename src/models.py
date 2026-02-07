# src/models.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


class IRNN(nn.Module):
    """
    Simple ReLU RNN used as baseline in arXiv:1610.06258 (Sec.4.1).
    - Input: token ids (0..36) -> embedding (emb_dim)
    - Recurrent: h_{t+1} = ReLU( W h_t + U x_t + b )
      with W initialized as 0.05 * I (identity-scaled), bias=0.
    - Output: linear from final hidden state to 10 classes (digits).
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.U = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Initialize W as 0.05 * I, bias=0
        with torch.no_grad():
            self.W.weight.zero_()
            self.W.weight += 0.05 * torch.eye(hidden_dim)
            self.W.bias.zero_()

        self.relu = nn.ReLU()
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: (B, T) long
        returns logits: (B, 10)
        """
        B, T = x_ids.shape
        x = self.emb(x_ids)  # (B, T, emb_dim)

        h = torch.zeros(B, self.hidden_dim, device=x_ids.device, dtype=x.dtype)
        for t in range(T):
            h = self.relu(self.W(h) + self.U(x[:, t, :]))
        logits = self.readout(h)
        return logits
