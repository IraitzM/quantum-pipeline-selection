"""Fixed-upload-time control for the trainable-time QSM arm.

The trainable arm of ``run_hankel_qsm.py`` beats the fixed arm on MNIST1D by
+0.083 (2.4 sigma), but the two arms do not share an upload time: the fixed arm
runs at ``UPLOAD_T = 0.1`` while the trainable arm *initialises* at
``1 / N_UPLOADS = 0.125`` and drifts to a learned per-layer profile spanning
roughly 0.04 to 0.35. The gain is therefore confounded, and could be either

  (a) a scale effect, i.e. 0.1 was simply the wrong upload time, which a single
      better-chosen constant would recover, or
  (b) a genuine learning effect, i.e. the per-layer *profile* does something no
      single constant can.

This script separates them by sweeping a geometric grid of fixed times under the
protocol of ``run_hankel_qsm.py``, unchanged in every other respect. The
comparison of interest is the best constant in the grid, selected on the test
set and so deliberately optimistic in favour of (a), against the learned profile
already recorded. Reading the control:

  best fixed t  >=  learned  ->  the gain was scale; rung 2b is not a climb
  best fixed t  <   learned  ->  the profile beats any constant, even an
                                 oracle-selected one, so the learning is real

Giving the fixed arm an oracle is the conservative choice, because it can only
make it harder to conclude that learning helps.

``t = 0.1`` is included in the grid as a protocol check: it should reproduce the
fixed-arm mean already in ``table_qsm_seeds.csv`` (MNIST1D 0.605).

Writes ``table_qsm_time_control.csv`` only. It never opens the tables that
``run_hankel_qsm.py`` owns.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # local package shadows stdlib ``code``

from code.datasets import load_ecg_beats, load_eeg_seizure, load_mnist1d  # noqa: E402
from code.qnn import HankelQSMClassifier, hankel_block_unitary  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-qsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Deliberately a new path: run_hankel_qsm.py owns table_qsm_seeds*.csv.
RUNS_CSV = OUT_DIR / "table_qsm_time_control.csv"

# Mirrored verbatim from run_hankel_qsm.py so the arms stay comparable.
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
N_UPLOADS = 8
EPOCHS = 20
LR = 0.005
BATCH_SIZE = 16
OMEGA_MAX = {"MNIST1D (4-class)": "10-15", "ECG (N vs. PVC)": "2-3",
             "EEG seizure": "23-24"}

# Brackets the fixed arm's 0.1, the trainable arm's 0.125 initialisation, and
# the range its learned per-layer times actually occupy.
DEFAULT_T_GRID = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3]


def _znorm(a: np.ndarray) -> np.ndarray:
    """Per-sample z-score, so the fixed upload time t has uniform meaning."""
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)


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


def train_eval_fixed_t(x_train, y_train, x_test, y_test, n_classes, seed, t):
    """One fixed-time fit. Mirrors run_hankel_qsm.train_eval's ``fixed`` arm,
    with the upload time supplied rather than read from a module constant."""
    torch.manual_seed(seed)
    n_qubits = HankelQSMClassifier.n_qubits_for(x_train.shape[1])
    t0 = time.time()
    train_in = torch.tensor(hankel_block_unitary(x_train, t=t), dtype=torch.complex128)
    test_in = torch.tensor(hankel_block_unitary(x_test, t=t), dtype=torch.complex128)
    model = HankelQSMClassifier(n_qubits=n_qubits, n_uploads=N_UPLOADS,
                                n_classes=n_classes)
    precompute_s = time.time() - t0

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.tensor(y_train, dtype=torch.long)
    n = train_in.shape[0]
    t0 = time.time()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(train_in[idx]), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(test_in).argmax(dim=1).numpy()
    return {
        "test_bacc": balanced_accuracy_score(y_test, pred),
        "n_qubits": n_qubits,
        "n_params": sum(p.numel() for p in model.parameters()),
        "precompute_s": round(precompute_s, 1),
        "train_time_s": round(time.time() - t0, 1),
    }


def datasets(eeg_split: str = "random"):
    """Identical draws to run_hankel_qsm.datasets().

    ``eeg_split`` is threaded through to :func:`load_eeg_seizure` so the sweep
    can be repeated on a recording-disjoint split, which is what decides whether
    the EEG curve's climb at large ``t`` is spectral or is the 59% window leak
    of the stratified random draw.
    """
    m = load_mnist1d()
    keep = [0, 1, 2, 3]
    trm, tem = np.isin(m["y_train"], keep), np.isin(m["y_test"], keep)
    x_test, y_test = subsample(m["x_test"][tem], m["y_test"][tem], 40, 42)
    yield ("MNIST1D (4-class)", 4, m["x_train"][trm], m["y_train"][trm], 150,
           x_test, y_test)

    e = load_ecg_beats()
    yield ("ECG (N vs. PVC)", 2, e["x_train"], e["y_train"], 300,
           e["x_test"], e["y_test"])

    s = load_eeg_seizure(n_per_class_train=200, n_per_class_test=100, seed=42,
                         split=eeg_split)
    yield ("EEG seizure", 2, s["x_train"], s["y_train"], 200,
           s["x_test"], s["y_test"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--t-grid", type=float, nargs="+", default=DEFAULT_T_GRID)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Substrings selecting a subset of the datasets.")
    parser.add_argument("--eeg-split", default="random",
                        choices=["random", "recording"],
                        help="Split for the EEG dataset only.")
    parser.add_argument("--out", type=Path, default=RUNS_CSV,
                        help="Destination CSV. Give each concurrent run its own, "
                             "since the writer truncates rather than appends.")
    args = parser.parse_args()
    out_csv = args.out

    rows = []
    for name, n_classes, x_tr_full, y_tr_full, n_per, x_test, y_test in datasets(args.eeg_split):
        if args.datasets and not any(d.lower() in name.lower() for d in args.datasets):
            continue
        print(f"\n=== {name} ===", flush=True)
        x_te = _znorm(x_test)
        for t in args.t_grid:
            for seed in args.seeds:
                x_tr, y_tr = subsample(x_tr_full, y_tr_full, n_per, seed)
                r = train_eval_fixed_t(_znorm(x_tr), y_tr, x_te, y_test,
                                       n_classes, seed, t)
                rows.append(dict(dataset=name, omega_max=OMEGA_MAX[name],
                                 arm="fixed_sweep", seed=seed,
                                 n_uploads=N_UPLOADS, upload_t=t,
                                 test_bacc=round(r["test_bacc"], 4),
                                 n_qubits=r["n_qubits"], n_params=r["n_params"],
                                 precompute_s=r["precompute_s"],
                                 train_time_s=r["train_time_s"]))
                pd.DataFrame(rows).to_csv(out_csv, index=False)  # incremental
                print(f"  t={t:<6g} seed={seed}: bacc={r['test_bacc']:.4f} "
                      f"({r['train_time_s']}s)", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\n=== mean test bacc by (dataset, t) ===", flush=True)
        print(df.groupby(["dataset", "upload_t"])["test_bacc"]
                .agg(["mean", "std", "count"]).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
