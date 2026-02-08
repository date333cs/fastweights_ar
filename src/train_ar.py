# src/train_ar.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data_ar import VOCAB_SIZE, batch_encoded
from src.datasets_ar import load_npz_dataset
from src.models import IRNN, LSTMClassifier, FastWeightsClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, R: int, emb: int, device: torch.device,
                fw_eta: float, fw_lam: float, fw_S: int) -> torch.nn.Module:
    name = name.lower()
    if name == "irnn":
        return IRNN(vocab_size=VOCAB_SIZE, emb_dim=emb, hidden_dim=R).to(device)
    if name == "lstm":
        return LSTMClassifier(vocab_size=VOCAB_SIZE, emb_dim=emb, hidden_dim=R).to(device)
    if name == "fw":
        return FastWeightsClassifier(
            vocab_size=VOCAB_SIZE, emb_dim=emb, hidden_dim=R,
            fw_eta=fw_eta, fw_lam=fw_lam, fw_S=fw_S,
        ).to(device)
    raise ValueError(f"Unknown model: {name!r} (use irnn|lstm|fw)")


def maybe_freeze_embedding(model: torch.nn.Module, freeze: bool) -> None:
    if not freeze:
        return
    if not hasattr(model, "emb"):
        raise ValueError("Model has no attribute 'emb' to freeze.")
    model.emb.weight.requires_grad_(False)


def make_batch_online(batch_size: int, K: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys, _ = batch_encoded(batch_size=batch_size, K=K, seed=seed)
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    returns (avg_loss, accuracy) over loader
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate_online(model: torch.nn.Module, K: int, batch_size: int, n_batches: int, seed0: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for i in range(n_batches):
        x, y = make_batch_online(batch_size, K, seed0 + 10_000 + i, device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="irnn", choices=["irnn", "lstm", "fw"])
    p.add_argument("--fw_eta", type=float, default=0.5)
    p.add_argument("--fw_lam", type=float, default=0.9)
    p.add_argument("--fw_S", type=int, default=1)
    p.add_argument("--R", type=int, default=50, help="hidden size")
    p.add_argument("--emb", type=int, default=100, help="embedding dim")

    # task
    p.add_argument("--K", type=int, default=8, help="number of letter-digit pairs (used only in online mode)")

    # training control
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)

    # evaluation (online mode)
    p.add_argument("--eval_batches", type=int, default=50)

    # fixed dataset mode
    p.add_argument("--data_npz", type=str, default="", help="Path to .npz dataset; if set, use fixed train/val/test.")
    p.add_argument("--num_workers", type=int, default=0)

    # misc
    p.add_argument("--freeze_emb", action="store_true", help="Freeze token embedding (keep 37->emb_dim representation fixed).")
    p.add_argument("--out", type=str, default="runs/tmp")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    model = build_model(args.model, R=args.R, emb=args.emb, device=device,fw_eta=args.fw_eta, fw_lam=args.fw_lam, fw_S=args.fw_S)

    maybe_freeze_embedding(model, args.freeze_emb)

    opt = torch.optim.Adam((pp for pp in model.parameters() if pp.requires_grad), lr=args.lr)

    # save config
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    log_path = outdir / "log.csv"
    if not log_path.exists():
        if args.data_npz:
            log_path.write_text("step,train_loss,val_loss,val_acc\n", encoding="utf-8")
        else:
            log_path.write_text("step,loss,acc\n", encoding="utf-8")

    if args.data_npz:
        # --- fixed dataset mode ---
        train_ds, info_tr = load_npz_dataset(args.data_npz, "train")
        val_ds, info_va = load_npz_dataset(args.data_npz, "val")
        test_ds, info_te = load_npz_dataset(args.data_npz, "test")

        # consistency check
        assert info_tr.K == info_va.K == info_te.K
        assert info_tr.T == info_va.T == info_te.T
        assert info_tr.vocab_size == info_va.vocab_size == info_te.vocab_size == VOCAB_SIZE

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

        # cycle through train loader for exactly args.steps updates
        train_iter = iter(train_loader)

        model.train()
        for step in range(1, args.steps + 1):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % args.eval_every == 0 or step == 1:
                val_loss, val_acc = evaluate_loader(model, val_loader, device=device)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{loss.item():.6f},{val_loss:.6f},{val_acc:.6f}\n")
                frz = " freeze_emb" if args.freeze_emb else ""
                print(f"[{args.model:4s}{frz} step {step:6d}] train_loss={loss.item():.6f}  val_loss={val_loss:.6f}  val_acc={val_acc:.4f}")

        # final test (1 time)
        test_loss, test_acc = evaluate_loader(model, test_loader, device=device)
        (outdir / "test.json").write_text(json.dumps({"test_loss": test_loss, "test_acc": test_acc}, indent=2), encoding="utf-8")
        print(f"[final test] loss={test_loss:.6f} acc={test_acc:.6f}")

    else:
        # --- online generation mode (previous behavior) ---
        model.train()
        for step in range(1, args.steps + 1):
            x, y = make_batch_online(args.batch, args.K, seed=args.seed + step, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % args.eval_every == 0 or step == 1:
                acc = evaluate_online(model, K=args.K, batch_size=args.batch, n_batches=args.eval_batches,
                                      seed0=args.seed, device=device)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{loss.item():.6f},{acc:.6f}\n")
                frz = " freeze_emb" if args.freeze_emb else ""
                print(f"[{args.model:4s}{frz} step {step:6d}] loss={loss.item():.6f} acc={acc:.4f}")

    torch.save(model.state_dict(), outdir / "model.pt")
    print(f"Saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
