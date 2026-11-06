"""
run_hankel_qsm.py
─────────────────
Multi-seed evaluation of the Hankel-lifted Quantum Spectral Model on the
paper's three 1-D benchmarks, the constructive test of the
compressor-not-encoding claim (Sections 5.1 and 6).

Wang, Parampalli & Myers (arXiv:2607.22516) show input-conditioned
Hamiltonian uploads beat fixed rotation-gate support on matrix-structured
data. Their construction needs matrix inputs; here each 1-D signal is
lifted to its trajectory (Hankel) matrix and uploaded through the block
Hamiltonian H_block = [[0, H], [H^T, 0]] (``code.qnn.hankel_block_unitary``),
so the model ingests the FULL signal with no average pool — the one
encoding in the repo that sidesteps the axis-one bottleneck entirely.
The paper's diagnostics then predict:

  * MNIST1D (omega_max 10-15, angle encoding fails under the 40->8 pool):
    the QSM should recover towards the raw-input RBF level, since the
    pooled-away spectral content is no longer discarded.
  * ECG (omega_max 2-3, angle encoding already at the RBF ceiling):
    no gain — sample-conditioned support is redundant at low bandwidth,
    mirroring the k=2 amplitude-reuploading null there.
  * EEG (omega_max 23-24): gain over the pooled angle encoding, whose
    fixed grid lacks the resolution.

Two arms run under an identical protocol. ``fixed`` freezes the upload
time, so the encoder carries no trainable parameters and the model sits
cleanly at rung two. ``trainable`` restores the per-layer t_l of their
Eq. (26), initialised to 1/L, which fits the realised frequency support
to the task and so climbs off rung two. The gap between the two is what
the trainable time buys.

Protocol matches run_amplitude_reupload_seeds.py (same per-seed
train subsamples, fixed seed-42 test sets, BATCH), with two
deliberate deviations: LR=0.005 and EPOCHS=20. The repo's usual
LR=0.05 destabilises this model class (the training loss oscillates
and never leaves chance); Wang et al. train at lr=0.01, and a
learning-rate screen on MNIST1D (train bacc 0.25/0.61/0.75 at epoch
20 for lr 0.05/0.01/0.005) selects 0.005. Results are written
incrementally to guard a long run (~30 min per seed over the three
datasets).

Outputs:
    results/qnn-qsm/table_qsm_seeds.csv       (per run)
    results/qnn-qsm/table_qsm_summary.csv     (mean +/- std + references)

Usage:
    uv run experiments/run_hankel_qsm.py
    uv run experiments/run_hankel_qsm.py --seeds 42 43 44
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
from code.qnn import (  # noqa: E402
    HankelQSMClassifier,
    TrainableTimeQSMClassifier,
    hankel_block_eigh,
    hankel_block_unitary,
)

OUT_DIR = ROOT / "results" / "qnn-qsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = OUT_DIR / "table_qsm_seeds.csv"
SUMMARY_CSV = OUT_DIR / "table_qsm_summary.csv"

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
N_UPLOADS = 8
UPLOAD_T = 0.1
EPOCHS = 20
LR = 0.005
BATCH_SIZE = 16

# Context columns from the existing tables (same subsample protocol).
OMEGA_MAX = {"MNIST1D (4-class)": "10-15", "ECG (N vs. PVC)": "2-3",
             "EEG seizure": "23-24"}
# Resampled references, 3 seeds from run_classification_seeds.py (paper
# tab:ecg-mnist-seeds / tab:eeg-seeds). QSM shares their TRAINING draws on
# seeds 42-44 but uses the fixed seed-42 test set, so contrasts are
# semi-paired (train-paired, test-unpaired).
POOLED_QNN_BEST = {"MNIST1D (4-class)": 0.535, "ECG (N vs. PVC)": 0.749,
                   "EEG seizure": 0.723}
RBF_RAW = {"MNIST1D (4-class)": 0.763, "ECG (N vs. PVC)": 0.757,
           "EEG seizure": 0.735}
AMPLITUDE_K1 = {"MNIST1D (4-class)": 0.4475, "ECG (N vs. PVC)": 0.7415,
                "EEG seizure": 0.514}


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


def train_eval(x_train, y_train, x_test, y_test, n_classes, seed, arm="fixed"):
    """Train one arm. ``fixed`` freezes the upload time, ``trainable`` learns it.

    The two arms share protocol, data draws, ansatz and read-out, so the
    difference between them is exactly what Wang et al.'s trainable ``t_l``
    buys over our frozen-time rung-two variant.
    """
    torch.manual_seed(seed)
    n_qubits = HankelQSMClassifier.n_qubits_for(x_train.shape[1])
    t0 = time.time()
    if arm == "fixed":
        # The upload is the same at every layer, so precompute it once.
        train_in = (torch.tensor(hankel_block_unitary(x_train, t=UPLOAD_T),
                                 dtype=torch.complex128),)
        test_in = (torch.tensor(hankel_block_unitary(x_test, t=UPLOAD_T),
                                dtype=torch.complex128),)
        model = HankelQSMClassifier(n_qubits=n_qubits, n_uploads=N_UPLOADS,
                                    n_classes=n_classes)
    else:
        # t_l moves, so only the eigendecomposition can be precomputed.
        w_tr, v_tr = hankel_block_eigh(x_train)
        w_te, v_te = hankel_block_eigh(x_test)
        train_in = (torch.tensor(w_tr, dtype=torch.float64),
                    torch.tensor(v_tr, dtype=torch.float64))
        test_in = (torch.tensor(w_te, dtype=torch.float64),
                   torch.tensor(v_te, dtype=torch.float64))
        model = TrainableTimeQSMClassifier(n_qubits=n_qubits,
                                           n_uploads=N_UPLOADS,
                                           n_classes=n_classes)
    precompute_s = time.time() - t0

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.tensor(y_train, dtype=torch.long)
    n = train_in[0].shape[0]
    t0 = time.time()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(*(a[idx] for a in train_in)), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(*test_in).argmax(dim=1).numpy()
    out = {
        "test_bacc": balanced_accuracy_score(y_test, pred),
        "n_qubits": n_qubits,
        "n_params": sum(p.numel() for p in model.parameters()),
        "precompute_s": round(precompute_s, 1),
        "train_time_s": round(time.time() - t0, 1),
    }
    if arm == "trainable":
        with torch.no_grad():
            out["learned_t_mean"] = round(model.upload_times.mean().item(), 4)
            out["learned_t_min"] = round(model.upload_times.min().item(), 4)
            out["learned_t_max"] = round(model.upload_times.max().item(), 4)
    return out


# Fixed test sets (seed 42); train subsets vary per seed inside the loop.
def datasets():
    m = load_mnist1d()
    keep = [0, 1, 2, 3]
    trm, tem = np.isin(m["y_train"], keep), np.isin(m["y_test"], keep)
    x_test, y_test = subsample(m["x_test"][tem], m["y_test"][tem], 40, 42)
    yield ("MNIST1D (4-class)", 4, m["x_train"][trm], m["y_train"][trm], 150,
           x_test, y_test)

    e = load_ecg_beats()
    yield ("ECG (N vs. PVC)", 2, e["x_train"], e["y_train"], 300,
           e["x_test"], e["y_test"])

    s = load_eeg_seizure(n_per_class_train=200, n_per_class_test=100, seed=42)
    yield ("EEG seizure", 2, s["x_train"], s["y_train"], 200,
           s["x_test"], s["y_test"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--arms", nargs="+", default=["fixed", "trainable"],
                        choices=["fixed", "trainable"],
                        help="fixed freezes the upload time (rung two); "
                             "trainable learns t_l as in Wang et al. Eq. (26).")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Substrings selecting a subset of the datasets.")
    args = parser.parse_args()

    rows = []
    for name, n_classes, x_tr_full, y_tr_full, n_per, x_test, y_test in datasets():
        if args.datasets and not any(d.lower() in name.lower() for d in args.datasets):
            continue
        print(f"\n=== {name} ===", flush=True)
        x_test = _znorm(x_test)
        for arm in args.arms:
            for seed in args.seeds:
                x_tr, y_tr = subsample(x_tr_full, y_tr_full, n_per, seed)
                r = train_eval(_znorm(x_tr), y_tr, x_test, y_test, n_classes,
                               seed, arm=arm)
                row = dict(dataset=name, omega_max=OMEGA_MAX[name], arm=arm,
                           seed=seed, n_uploads=N_UPLOADS,
                           upload_t=UPLOAD_T if arm == "fixed" else 1 / N_UPLOADS,
                           test_bacc=round(r["test_bacc"], 4),
                           n_qubits=r["n_qubits"], n_params=r["n_params"],
                           precompute_s=r["precompute_s"],
                           train_time_s=r["train_time_s"],
                           learned_t_mean=r.get("learned_t_mean"),
                           learned_t_min=r.get("learned_t_min"),
                           learned_t_max=r.get("learned_t_max"))
                rows.append(row)
                pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)  # incremental
                print(f"  {arm:9s} seed={seed}: bacc={row['test_bacc']:.4f} "
                      f"({r['n_qubits']}q, {r['train_time_s']}s train)", flush=True)

    df = pd.DataFrame(rows)
    summ = (df.groupby(["dataset", "omega_max", "arm"])["test_bacc"]
              .agg(["mean", "std", "count"]).reset_index())
    summ["mean"] = summ["mean"].round(4)
    summ["std"] = summ["std"].round(4)
    summ["pooled_qnn_best"] = summ["dataset"].map(POOLED_QNN_BEST)
    summ["amplitude_k1"] = summ["dataset"].map(AMPLITUDE_K1)
    summ["rbf_raw"] = summ["dataset"].map(RBF_RAW)
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}")
    print(summ.to_string(index=False))

    # The delta the two arms exist to measure: what the trainable time buys.
    if set(args.arms) == {"fixed", "trainable"}:
        wide = summ.pivot(index="dataset", columns="arm", values="mean")
        wide["delta"] = (wide["trainable"] - wide["fixed"]).round(4)
        print("\nTrainable-time minus fixed-time (test balanced accuracy):")
        print(wide.to_string())


if __name__ == "__main__":
    main()
