# src/models.py
from __future__ import annotations

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
    Fast Weights RNN for associative retrieval.
    - token -> embedding (emb_dim)
    - recurrent core with hidden_dim = R (ReLU)
      * slow weights: W (R x R), C (R x emb_dim)
      * fast weights: A(t) (R x R), updated by A <- lam*A + eta*h*h^T
      * inner-loop settling: run S steps per time step
      * layer norm applied to summed input before ReLU
    - head: 100 ReLU -> logits(10)
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
        inner_steps: int = 1,
        w_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.hidden_dim = hidden_dim
        self.fw_eta = float(fw_eta)
        self.fw_lam = float(fw_lam)
        self.inner_steps = int(inner_steps)

        # slow weights
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.C = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        # layer norm (applied to hidden_dim vector)
        self.ln = nn.LayerNorm(hidden_dim)

        # head (shared style with other models)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes),
        )

        self.reset_parameters(w_scale=w_scale)

    def reset_parameters(self, w_scale: float = 0.05) -> None:
        # W := w_scale * I
        with torch.no_grad():
            self.W.zero_()
            self.W += w_scale * torch.eye(self.hidden_dim)

        # C (input-to-hidden) like standard linear init
        nn.init.kaiming_uniform_(self.C.weight, a=5 ** 0.5)

        # embedding
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

        # bias already zeros

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: (B, T) int64
        returns logits: (B, 10)
        """
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