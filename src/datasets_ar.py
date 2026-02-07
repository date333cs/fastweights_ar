# src/datasets_ar.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ARDatasetInfo:
    K: int
    T: int
    vocab_size: int


class NpzSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        self.x = torch.from_numpy(x).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_npz_dataset(npz_path: str | Path, split: Split) -> tuple[NpzSequenceDataset, ARDatasetInfo]:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    if split == "train":
        x = data["x_train"]
        y = data["y_train"]
    elif split == "val":
        x = data["x_val"]
        y = data["y_val"]
    elif split == "test":
        x = data["x_test"]
        y = data["y_test"]
    else:
        raise ValueError(f"Unknown split: {split!r}")

    info = ARDatasetInfo(
        K=int(data["K"]),
        T=int(data["T"]),
        vocab_size=int(data["vocab_size"]),
    )
    return NpzSequenceDataset(x, y), info