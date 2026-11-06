"""
run_amplitude_reupload_seeds.py
─────────────────────────────────
Multi-seed version of run_amplitude_reupload.py, to put error bars on the
k=1 vs k=2 tensor-re-embedding comparison (Section 7, Table 5). For each
seed we resample the training subset and re-initialise the model (both
driven by that seed), holding the test set fixed, so the reported spread
captures both data-sampling and training/initialisation variance. k=1 and
k=2 share the same per-seed training subset, so the comparison is paired.

Settings match run_amplitude_reupload.py (identical per-dataset subsample
sizes, N_LAYERS, EPOCHS, LR, BATCH). Results are written incrementally to
guard a long run.

Outputs:
    results/qnn-amplitude/table_reupload_seeds.csv        (per run)
    results/qnn-amplitude/table_reupload_summary.csv      (mean +/- std)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.datasets import load_ecg_beats, load_eeg_seizure, load_mnist1d  # noqa: E402
from code.qnn import AmplitudeReuploadClassifier, to_nonneg_domain  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-amplitude"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = OUT_DIR / "table_reupload_seeds.csv"
SUMMARY_CSV = OUT_DIR / "table_reupload_summary.csv"

SEEDS = [42, 43, 44, 45, 46]
N_LAYERS = 3
EPOCHS = 15
LR = 0.05
BATCH_SIZE = 16
OMEGA_MAX = {"MNIST1D (4-class)": "10-15", "ECG (N vs. PVC)": "2-3",
             "EEG seizure": "23-24"}
ANGLE_POOLED_BEST = {"MNIST1D (4-class)": 0.562, "ECG (N vs. PVC)": 0.770,
                     "EEG seizure": 0.755}


def subsample(x, y, n_per_class, seed):
    rng = np.random.default_rng(seed)
    idx_all = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        idx_all.append(idx_c[:n_per_class])
    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return x[idx], y[idx]


def train_eval(x_train, y_train, x_test, y_test, n_classes, k, seed):
    torch.manual_seed(seed)
    model = AmplitudeReuploadClassifier(
        input_dim=x_train.shape[1], k=k, n_layers=N_LAYERS, n_classes=n_classes
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, dtype=torch.float64)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = xt.shape[0]
    t0 = time.time()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float64)).argmax(dim=1).numpy()
    return {
        "test_bacc": balanced_accuracy_score(y_test, pred),
        "n_qubits": model.n_qubits,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_time_s": round(time.time() - t0, 1),
    }


# Fixed test sets (seed 42); train subsets vary per seed inside the loop.
def datasets():
    m = load_mnist1d()
    keep = [0, 1, 2, 3]
    trm, tem = np.isin(m["y_train"], keep), np.isin(m["y_test"], keep)
    x_test, y_test = subsample(m["x_test"][tem], m["y_test"][tem], 40, 42)
    yield ("MNIST1D (4-class)", 4, m["x_train"][trm], m["y_train"][trm], 150,
           to_nonneg_domain(x_test), y_test)

    e = load_ecg_beats()
    yield ("ECG (N vs. PVC)", 2, e["x_train"], e["y_train"], 300,
           to_nonneg_domain(e["x_test"]), e["y_test"])

    s = load_eeg_seizure(n_per_class_train=200, n_per_class_test=100, seed=42)
    yield ("EEG seizure", 2, s["x_train"], s["y_train"], 200,
           to_nonneg_domain(s["x_test"]), s["y_test"])


def main():
    rows = []
    for name, n_classes, x_tr_full, y_tr_full, n_per, x_test, y_test in datasets():
        print(f"\n=== {name} ===", flush=True)
        for seed in SEEDS:
            x_tr, y_tr = subsample(x_tr_full, y_tr_full, n_per, seed)
            x_tr = to_nonneg_domain(x_tr)
            for k in (1, 2):
                r = train_eval(x_tr, y_tr, x_test, y_test, n_classes, k, seed)
                row = dict(dataset=name, omega_max=OMEGA_MAX[name], seed=seed, k=k,
                           test_bacc=round(r["test_bacc"], 4), n_qubits=r["n_qubits"],
                           n_params=r["n_params"], train_time_s=r["train_time_s"])
                rows.append(row)
                pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)  # incremental
                print(f"  seed={seed} k={k}: bacc={row['test_bacc']:.4f} "
                      f"({r['train_time_s']}s)", flush=True)

    df = pd.DataFrame(rows)
    summ = (df.groupby(["dataset", "omega_max", "k"])["test_bacc"]
              .agg(["mean", "std", "count"]).reset_index())
    summ["mean"] = summ["mean"].round(4)
    summ["std"] = summ["std"].round(4)
    summ["angle_pooled_best"] = summ["dataset"].map(ANGLE_POOLED_BEST)
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
