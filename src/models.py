# src/models.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class IRNN(nn.Module):
    """
    Simple ReLU RNN baseline (IRNN).
    - Embedding -> ReLU RNN with W initialized as 0.05 * I
    - Readout from final hidden state -> (100 ReLU) -> 10 classes
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_classes: int = 10,
        head_dim: int = 100,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.U = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)

        with torch.no_grad():
            self.W.weight.zero_()
            self.W.weight += 0.05 * torch.eye(hidden_dim)
            self.W.bias.zero_()

        self.relu = nn.ReLU()

        # (shared-style head) hidden_dim -> head_dim -> num_classes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        x = self.emb(x_ids)  # (B, T, emb_dim)
        h = torch.zeros(B, self.W.in_features, device=x_ids.device, dtype=x.dtype)
        for t in range(T):
            h = self.relu(self.W(h) + self.U(x[:, t, :]))
        return self.head(h)


class LSTMClassifier(nn.Module):
    """
    LSTM baseline.
    - Embedding -> (single-layer) LSTM -> readout from final hidden state -> (100 ReLU) -> 10 classes
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_classes: int = 10,
        head_dim: int = 100,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # (shared-style head) hidden_dim -> head_dim -> num_classes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(x_ids)  # (B, T, emb_dim)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]     # (B, hidden_dim)
        return self.head(h_last)


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


class FastWeightsClassifier(nn.Module):
    """
    Fast weights RNN (Ba et al., 2016) for associative retrieval.
    - Input: token ids (B,T)
    - Embedding -> recurrent core with fast matrix A(t)
    - Readout: final h -> (100 ReLU) -> 10 classes

    Key implementation points:
      (1) h0 = ReLU(u)  where u = W h + U x  (NO LayerNorm here)
      (2) inner loop: hs = ReLU( LayerNorm( u + A @ hs ) ) repeated S times
      (3) A update: A = lam * A + eta * (h outer h)   (do NOT detach by default)
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_classes: int = 10,
        head_dim: int = 100,
        fw_eta: float = 0.5,
        fw_lam: float = 0.9,
        fw_S: int = 1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        # slow weights
        self.U = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # init: W = 0.05 * I (paper)
        with torch.no_grad():
            self.W.weight.zero_()
            self.W.weight += 0.05 * torch.eye(hidden_dim)

        # init: U uniform(-1/sqrt(H), +1/sqrt(H)) where H = fan_out (paper)
        # For Linear(emb_dim -> hidden_dim), fan_out = hidden_dim.
        bound = 1.0 / math.sqrt(hidden_dim)
        with torch.no_grad():
            self.U.weight.uniform_(-bound, bound)

        self.ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

        self.fw_eta = fw_eta
        self.fw_lam = fw_lam
        self.fw_S = fw_S

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        x = self.emb(x_ids)  # (B,T,emb_dim)

        h = torch.zeros(B, self.W.in_features, device=x_ids.device, dtype=x.dtype)
        A = torch.zeros(B, self.W.in_features, self.W.in_features, device=x_ids.device, dtype=x.dtype)

        for t in range(T):
            # sustained boundary condition (u = W h + U x_t)
            u = self.W(h) + self.U(x[:, t, :])  # (B,R)

            # preliminary state (NO LayerNorm here)
            hs = self.relu(u)

            # inner loop: apply LayerNorm each iteration
            for _ in range(self.fw_S):
                Ah = torch.bmm(A, hs.unsqueeze(2)).squeeze(2)  # (B,R)
                hs = self.relu(self.ln(u + Ah))

            h = hs

            # fast weight update (once per time step)
            # IMPORTANT: do NOT detach by default
            outer = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))  # (B,R,R)
            A = self.fw_lam * A + self.fw_eta * outer

        return self.head(h)

        B, T = x_ids.shape
        device = x_ids.device
        R = self.hidden_dim

        # h(t), A(t): per-sequence in minibatch
        h = torch.zeros(B, R, device=device)
        A = torch.zeros(B, R, R, device=device)

        for t in range(T):
            e = self.emb(x_ids[:, t])  # (B, emb_dim)

            # sustained boundary condition: W h(t) + C x(t) + b
            # (B,R) = (B,R) + (B,R) + (R,)
            boundary = h @ self.W.T + self.C(e) + self.b

            # preliminary state
            hs = F.relu(self.ln(boundary))

            # inner loop: hs <- f( LN[ boundary + A * hs ] )
            # A * hs: (B,R,R) x (B,R,1) -> (B,R)
            for _ in range(self.inner_steps):
                Ah = torch.bmm(A, hs.unsqueeze(2)).squeeze(2)
                hs = F.relu(self.ln(boundary + Ah))

            # commit new hidden state
            h = hs

            # update fast weights: A <- lam*A + eta*h*h^T
            # (B,R,1) x (B,1,R) -> (B,R,R)
            hhT = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))
            A = self.fw_lam * A + self.fw_eta * hhT

        # classify from final hidden state
        logits = self.head(h)
        return logits