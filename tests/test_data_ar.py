# tests/test_data_ar.py
"""
Minimal, dependency-free sanity tests for src/data_ar.py.

Run from repository root:
  python tests/test_data_ar.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import random

# --- add repo root to sys.path so that `import src...` works ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_ar import (
    ALPHABET, DIGITS, SEP, VOCAB, VOCAB_SIZE,
    encode, decode,
    sample_one, sample_one_encoded,
    parse_mapping_from_seq,
)


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_vocab() -> None:
    assert_true(VOCAB_SIZE == 37, f"VOCAB_SIZE should be 37 (got {VOCAB_SIZE})")
    assert_true(len(VOCAB) == 37, f"VOCAB length should be 37 (got {len(VOCAB)})")
    assert_true(VOCAB.endswith(SEP), "VOCAB should include '?' as last token")
    # letters + digits + '?'
    assert_true(set(ALPHABET).issubset(set(VOCAB)), "Alphabet tokens missing from vocab")
    assert_true(set(DIGITS).issubset(set(VOCAB)), "Digit tokens missing from vocab")
    assert_true(SEP in VOCAB, "Separator token '?' missing from vocab")


def test_encode_decode_roundtrip() -> None:
    s = "c9k8j3f1??c"
    ids = encode(s)
    s2 = decode(ids)
    assert_true(s2 == s, f"encode/decode roundtrip failed: {s2!r} != {s!r}")


def test_sample_structure() -> None:
    rng = random.Random(123)
    K = 7
    sample = sample_one(K=K, rng=rng)

    # length check
    expected_len = 2 * K + 3
    assert_true(len(sample.seq) == expected_len, f"seq length mismatch: {len(sample.seq)} vs {expected_len}")

    # separator check
    assert_true(sample.seq[2*K:2*K+2] == "??", f"separator not found: {sample.seq!r}")

    # keys are distinct
    keys = list(sample.mapping.keys())
    assert_true(len(keys) == K, "mapping size mismatch")
    assert_true(len(set(keys)) == K, "keys are not unique")

    # query is among keys
    assert_true(sample.query in sample.mapping, "query not in mapping")

    # target matches mapping
    assert_true(sample.target_digit == sample.mapping[sample.query], "target != mapping[query]")

    # digits are 0..9
    assert_true(sample.target_digit in DIGITS, "target digit out of range")


def test_parse_mapping_matches_sample() -> None:
    rng = random.Random(999)
    K = 10
    sample = sample_one(K=K, rng=rng)
    mapping2, query2 = parse_mapping_from_seq(sample.seq, K=K)
    assert_true(query2 == sample.query, "parsed query mismatch")
    assert_true(mapping2 == sample.mapping, "parsed mapping mismatch")
    assert_true(mapping2[query2] == sample.target_digit, "parsed target mismatch")


def test_encoded_label_matches() -> None:
    rng = random.Random(7)
    K = 5
    x_ids, y, meta = sample_one_encoded(K=K, rng=rng)
    assert_true(isinstance(y, int), "label must be int")
    assert_true(0 <= y <= 9, f"label out of range: {y}")
    assert_true(int(meta.target_digit) == y, "label != int(target_digit)")
    # decode back to verify ids correspond to the same sequence
    seq2 = decode(x_ids)
    assert_true(seq2 == meta.seq, "encoded ids do not decode back to original seq")


def run_all() -> None:
    test_vocab()
    test_encode_decode_roundtrip()
    test_sample_structure()
    test_parse_mapping_matches_sample()
    test_encoded_label_matches()
    print("OK: all tests passed.")


if __name__ == "__main__":
    run_all()