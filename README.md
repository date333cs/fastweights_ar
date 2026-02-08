# fastweights_ar

Reproducible implementation of the **Associative Retrieval** experiment (Section 4.1) from:

- **Using Fast Weights to Attend to the Recent Past**  
  Jimmy Ba, Geoffrey E. Hinton, Volodymyr Mnih, Joel Z. Leibo, Cătălin Ionescu  
  *NeurIPS 2016*  
  - arXiv: https://arxiv.org/abs/1610.06258  
  - NeurIPS abstract page: https://papers.nips.cc/paper_files/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.ht
ml  

Date: 2026-02-07

---

## What this repo contains

This repository implements and compares the following models on the **Associative Retrieval (AR)** task:

- **IRNN** (ReLU RNN baseline with identity initialization)
- **LSTM** baseline
- **Fast Weights** model (Ba et al., 2016)

The goal is to reproduce the behavior reported in the paper and provide a clean, runnable reference implementation.

---

## Task: Associative Retrieval (Section 4.1)

The input is a sequence encoding **K letter–digit bindings** followed by a **query letter**:

```
a 1 b 2 c 3 ... ? b
```

The model must output the digit associated with the query letter (here, `2`).  
This is a 10-class classification problem (digits 0–9).

Sequence length is:

- **T = 2K + 3**

---

## Setup

### Requirements

- Python 3.10+ recommended (works on newer versions as well)
- PyTorch

Install dependencies (example):

```bash
pip install -r requirements.txt
```

If you do not have `requirements.txt`, install at least:

```bash
pip install torch numpy
```

---

## Quickstart

### 1) Run unit tests

```bash
python tests/test_data_ar.py
```

### 2) Generate a fixed dataset (train/val/test)

The paper uses fixed splits:

- 100,000 training examples  
- 10,000 validation examples  
- 20,000 test examples  

Generate a dataset for a given `K`:

```bash
python -m src.make_dataset_ar --K 4 --seed 0 --out data/ar_K4_seed0.npz
python -m src.make_dataset_ar --K 16 --seed 0 --out data/ar_K16_seed0.npz
```

### 3) Train a model on the fixed dataset

Examples:

```bash
# Fast Weights (debug-friendly small K)
python -m src.train_ar --model fw   --R 50 --emb 100   --steps 20000 --batch 128 --lr 1e-3   --seed 0 --data_npz data/ar_K
4_seed0.npz   --fw_eta 0.5 --fw_lam 0.9 --fw_S 1   --eval_every 500 --out runs/fw_K4_R50

# LSTM baseline
python -m src.train_ar --model lstm   --R 50 --emb 100   --steps 20000 --batch 128 --lr 1e-3   --seed 0 --data_npz data/ar
_K16_seed0.npz   --eval_every 500 --out runs/lstm_K16_R50

# IRNN baseline
python -m src.train_ar --model irnn   --R 50 --emb 100   --steps 20000 --batch 128 --lr 1e-3   --seed 0 --data_npz data/ar
_K16_seed0.npz   --eval_every 500 --out runs/irnn_K16_R50
```

Outputs are written under `runs/.../`:

- `log.csv` (training log)
- `test.json` (final test metrics)
- `model.pt` (saved parameters)

---

## Reproducibility: run multiple seeds and summarize

Example: `K=16, R=50`, seeds 0–2.

```bash
for model in irnn lstm fw; do
  for seed in 0 1 2; do
    OUT="runs/${model}_K16_R50_fixed_seed${seed}"
    python -m src.train_ar       --model ${model}       --R 50 --emb 100       --steps 20000 --batch 128 --lr 1e-3       -
-seed ${seed}       --data_npz data/ar_K16_seed0.npz       --fw_eta 0.5 --fw_lam 0.9 --fw_S 1       --eval_every 500      
 --out ${OUT}
  done
done
```

Summarize `test_acc` across runs:

```bash
python - <<'PY'
import glob, json, statistics as st

def read_accs(pattern):
    vals = []
    for p in sorted(glob.glob(pattern)):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        vals.append(float(d["test_acc"]))
    return vals

def summarize(vals):
    mean = sum(vals) / len(vals)
    std = st.pstdev(vals)
    return mean, std

print("model,K,n,mean_test_acc,std_test_acc")

K = 16
for model in ["irnn", "lstm", "fw"]:
    vals = read_accs(f"runs/{model}_K{K}_R50_fixed_seed*/test.json")
    if not vals:
        continue
    mean, std = summarize(vals)
    print(f"{model},{K},{len(vals)},{mean:.6f},{std:.6f}")
PY
```

---

## Fast Weights hyperparameters

The Fast Weights model uses:

- `fw_eta`: update strength (η)
- `fw_lam`: decay (λ)
- `fw_S`: inner-loop settling steps (S)

Typical starting point (as used in this repo):

- `fw_eta = 0.5`
- `fw_lam = 0.9`
- `fw_S = 1`

---

## Notes

- Metrics shown during training use the **validation split**.
- Final metrics in `test.json` are computed once on the **test split** after training.

---

## References / Videos

- Jimmy Ba (NeurIPS 2016): https://www.youtube.com/watch?v=Hd20zGKAdoI  
- Geoffrey Hinton (2017): https://youtu.be/GLmptInTNSw?t=1540  

---

## License

MIT

