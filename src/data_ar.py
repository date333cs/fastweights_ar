# src/data_ar.py
"""
Associative retrieval task data generator (Fast Weights paper, Sec. 4.1).

Sequence format (example, K=4):
  c9k8j3f1??c   -> target 9

Rules:
- Sample K distinct letters (without replacement) from the English alphabet.
- For each letter, sample a digit 0..9 (digits may repeat).
- Concatenate K (letter,digit) pairs, then the separator '??', then a query letter
  chosen uniformly from the K letters.
- The label is the digit associated with the query letter.

This module is intentionally dependency-free (standard library only),
so it can be validated locally before moving to Colab/GPU training.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import string
from typing import Dict, List, Tuple, Optional


ALPHABET: str = string.ascii_lowercase  # 26 letters
DIGITS: str = "0123456789"
SEP: str = "?"  # separator token, used twice: "??"

# Vocabulary over single-character tokens: 26 letters + 10 digits + '?'
VOCAB: str = ALPHABET + DIGITS + SEP
VOCAB_SIZE: int = len(VOCAB)

TOKEN_TO_ID: Dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: ch for i, ch in enumerate(VOCAB)}


@dataclass(frozen=True)
class Sample:
    seq: str                 # e.g., "c9k8j3f1??c"
    target_digit: str        # e.g., "9"
    query: str               # e.g., "c"
    mapping: Dict[str, str]  # letter -> digit (for the K keys)


def encode(seq: str) -> List[int]:
    """Map a string of tokens to integer ids."""
    ids: List[int] = []
    for ch in seq:
        if ch not in TOKEN_TO_ID:
            raise ValueError(f"Unknown token: {ch!r}")
        ids.append(TOKEN_TO_ID[ch])
    return ids


def decode(ids: List[int]) -> str:
    """Map integer ids back to token string."""
    out_chars: List[str] = []
    for i in ids:
        if i not in ID_TO_TOKEN:
            raise ValueError(f"Unknown id: {i!r}")
        out_chars.append(ID_TO_TOKEN[i])
    return "".join(out_chars)


def sample_one(K: int, rng: Optional[random.Random] = None) -> Sample:
    """
    Generate one associative retrieval sample.

    Length:
      T = 2*K + 3
      (K pairs => 2K chars) + "??" (2 chars) + query (1 char)
    """
    if rng is None:
        rng = random.Random()

    if not (1 <= K <= len(ALPHABET)):
        raise ValueError(f"K must be between 1 and {len(ALPHABET)} (got {K}).")

    keys: List[str] = rng.sample(ALPHABET, K)  # without replacement
    vals: List[str] = [DIGITS[rng.randrange(10)] for _ in range(K)]
    mapping: Dict[str, str] = {k: v for k, v in zip(keys, vals)}

    query: str = keys[rng.randrange(K)]
    target_digit: str = mapping[query]

    pairs: str = "".join([k + v for (k, v) in zip(keys, vals)])
    seq: str = pairs + (SEP * 2) + query

    return Sample(seq=seq, target_digit=target_digit, query=query, mapping=mapping)


def sample_one_encoded(
    K: int, rng: Optional[random.Random] = None
) -> Tuple[List[int], int, Sample]:
    """
    Return (token_ids, label_int, meta_sample).
    label_int is 0..9.
    """
    s = sample_one(K=K, rng=rng)
    x_ids = encode(s.seq)
    y = int(s.target_digit)
    return x_ids, y, s


def batch_encoded(
    batch_size: int, K: int, seed: Optional[int] = None
) -> Tuple[List[List[int]], List[int], List[Sample]]:
    """
    Generate a batch as Python lists (no numpy).
      xs: list of length B, each is a list[int] of length T
      ys: list of length B, each is int in 0..9
      metas: list of Sample
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rng = random.Random(seed)

    xs: List[List[int]] = []
    ys: List[int] = []
    metas: List[Sample] = []
    for _ in range(batch_size):
        x_ids, y, meta = sample_one_encoded(K=K, rng=rng)
        xs.append(x_ids)
        ys.append(y)
        metas.append(meta)
    return xs, ys, metas


def parse_mapping_from_seq(seq: str, K: int) -> Tuple[Dict[str, str], str]:
    """
    Parse the (letter->digit) mapping and query from seq.
    This is used for sanity checks / tests.

    Expected seq structure:
      positions 0..2K-1: alternating letter, digit
      positions 2K..2K+1: "??"
      position 2K+2: query letter
    """
    expected_len = 2 * K + 3
    if len(seq) != expected_len:
        raise ValueError(f"Expected length {expected_len}, got {len(seq)}: {seq!r}")

    mapping: Dict[str, str] = {}
    for i in range(0, 2 * K, 2):
        k = seq[i]
        v = seq[i + 1]
        if k not in ALPHABET:
            raise ValueError(f"Expected letter at pos {i}, got {k!r}")
        if v not in DIGITS:
            raise ValueError(f"Expected digit at pos {i+1}, got {v!r}")
        if k in mapping:
            raise ValueError(f"Duplicate key in pairs: {k!r}")
        mapping[k] = v

    if seq[2 * K] != SEP or seq[2 * K + 1] != SEP:
        raise ValueError(f"Expected '??' separator at pos {2*K}, got {seq[2*K:2*K+2]!r}")

    query = seq[2 * K + 2]
    if query not in mapping:
        raise ValueError(f"Query {query!r} not among keys.")
    return mapping, query


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(3):
        s = sample_one(K=4, rng=rng)
        print(s.seq, "->", s.target_digit)



