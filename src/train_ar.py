# src/train_ar.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import random
from typing import Tuple

import torch
import torch.nn.functional as F

from src.data_ar import VOCAB_SIZE, batch_encoded
from src.models import IRNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(batch_size: int, K: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys, _ = batch_encoded(batch_size=batch_size, K=K, seed=seed)
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(model: torch.nn.Module, K: int, batch_size: int, n_batches: int, seed0: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for i in range(n_batches):
        x, y = make_batch(batch_size, K, seed0 + 10_000 + i, device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=50, help="hidden size")
    p.add_argument("--emb", type=int, default=100, help="embedding dim")
    p.add_argument("--K", type=int, default=8, help="number of key-value pairs")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--out", type=str, default="runs/tmp")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    model = IRNN(vocab_size=VOCAB_SIZE, emb_dim=args.emb, hidden_dim=args.R).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Save config
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    log_path = outdir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,acc\n", encoding="utf-8")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(args.batch, args.K, seed=args.seed + step, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            acc = evaluate(model, K=args.K, batch_size=args.batch, n_batches=args.eval_batches,
                           seed0=args.seed, device=device)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f},{acc:.6f}\n")
            print(f"[step {step:6d}] loss={loss.item():.6f} acc={acc:.4f}")

    # Save final model
    torch.save(model.state_dict(), outdir / "model.pt")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()