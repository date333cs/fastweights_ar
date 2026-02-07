# src/make_dataset_ar.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data_ar import batch_encoded, VOCAB_SIZE


def gen_split(n: int, K: int, seed0: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n examples (x_ids, y) for a fixed K.
    x_ids: (n, T) int64, y: (n,) int64
    Deterministic given (seed0, n, K, chunk).
    """
    xs_list = []
    ys_list = []
    done = 0
    chunk_idx = 0
    while done < n:
        bs = min(chunk, n - done)
        # Different seed per chunk for determinism and independence
        xs, ys, _ = batch_encoded(batch_size=bs, K=K, seed=seed0 + chunk_idx)
        xs_list.append(np.asarray(xs, dtype=np.int64))
        ys_list.append(np.asarray(ys, dtype=np.int64))
        done += bs
        chunk_idx += 1
    x = np.concatenate(xs_list, axis=0)
    y = np.concatenate(ys_list, axis=0)
    return x, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--train_n", type=int, default=100_000)
    p.add_argument("--val_n", type=int, default=10_000)
    p.add_argument("--test_n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    K = args.K
    T = 2 * K + 3

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data") / f"ar_K{K}_seed{args.seed}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr = gen_split(args.train_n, K=K, seed0=args.seed + 0, chunk=args.chunk)
    x_va, y_va = gen_split(args.val_n,   K=K, seed0=args.seed + 1_000_000, chunk=args.chunk)
    x_te, y_te = gen_split(args.test_n,  K=K, seed0=args.seed + 2_000_000, chunk=args.chunk)

    # sanity shapes
    assert x_tr.shape == (args.train_n, T)
    assert x_va.shape == (args.val_n, T)
    assert x_te.shape == (args.test_n, T)
    assert y_tr.shape == (args.train_n,)
    assert y_va.shape == (args.val_n,)
    assert y_te.shape == (args.test_n,)

    np.savez_compressed(
        out_path,
        x_train=x_tr, y_train=y_tr,
        x_val=x_va,   y_val=y_va,
        x_test=x_te,  y_test=y_te,
        K=np.int64(K),
        T=np.int64(T),
        vocab_size=np.int64(VOCAB_SIZE),
    )
    print(f"Saved dataset: {out_path.resolve()}")
    print(f"K={K}  T={T}  vocab_size={VOCAB_SIZE}")
    print(f"train={args.train_n}  val={args.val_n}  test={args.test_n}")


if __name__ == "__main__":
    main()
