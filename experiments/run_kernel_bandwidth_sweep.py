"""
run_kernel_bandwidth_sweep.py
──────────────────────────────
Bandwidth sweep + multi-seed evaluation for the graded-entangling-capacity
Pauli feature-map QSVMs, on both 3-D datasets (synthetic 2c checkerboard and
real NoduleMNIST3D).

Why: our earlier kernel results fixed the Havlicek input scaling at
``2*(pi - x_i)``, i.e. bandwidth lambda = 1. Bandwidth is the first-order
design choice for quantum kernels (Shaydulin & Wild 2022; Canatar et al. 2023),
and Shen (arXiv:2607.20168) finds the tuned optimum at interior bandwidths with
performance collapsing in the exponential-concentration regime. A null result
at lambda = 1 therefore does not establish a null for the encoding family. This
script gives every feature map its own tuned bandwidth and asks again whether
entangling capacity buys anything.

Protocol (kernel-swap control, after Shen):
  * every encoding and the classical RBF control share each seed's split;
  * bandwidth (quantum) and gamma (RBF) are tuned on an inner validation split
    carved from the training set, with grids of equal size, so no model gets a
    larger search budget than another;
  * the selected hyperparameter is refit on the full training set and scored
    once on the held-out test set.

We report both the tuned result and the full test-set sweep. The sweep is
diagnostic only -- picking lambda by test score would be selection on the test
set, which is the error the tuned column exists to avoid.

Outputs:
    results/qnn-3d/table_bandwidth_sweep.csv    (every seed x encoding x lambda)
    results/qnn-3d/table_bandwidth_tuned.csv    (validation-tuned, per seed)
    results/qnn-3d/table_bandwidth_summary.csv  (mean +/- std + paired deltas)
    results/qnn-3d/figure_bandwidth_sweep.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # local package shadows stdlib "code"

from code.datasets import load_2c_overlap, load_nodulemnist3d  # noqa: E402
from code.qkernel import ENCODINGS, fast_kernel_matrix, meyer_wallach_capacity  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
BANDWIDTHS = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
# Same number of grid points as BANDWIDTHS, so the classical control gets an
# equal tuning budget. Both grids are deliberately wide enough that the
# selected value is interior rather than at an endpoint (a grid-endpoint
# artefact is one of the failure modes Shen documents).
RBF_GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
VAL_FRACTION = 0.25
OVERLAPS = [0.0, 0.3, 0.5, 0.7]


def _qsvm_score(Xtr, ytr, Xev, yev, n_qubits, paulis, ent, bandwidth):
    ktr = fast_kernel_matrix(Xtr, None, n_qubits, paulis, ent, bandwidth)
    kev = fast_kernel_matrix(Xev, Xtr, n_qubits, paulis, ent, bandwidth)
    svc = SVC(kernel="precomputed", class_weight="balanced").fit(ktr, ytr)
    return balanced_accuracy_score(yev, svc.predict(kev))


def _rbf_score(Xtr, ytr, Xev, yev, gamma):
    svc = SVC(kernel="rbf", gamma=gamma, class_weight="balanced").fit(Xtr, ytr)
    return balanced_accuracy_score(yev, svc.predict(Xev))


def run_one_split(Xtr, ytr, Xte, yte, n_qubits, seed, tag, capacities):
    """Full sweep + validation-tuned evaluation for one train/test split."""
    Xin, Xval, yin, yval = train_test_split(
        Xtr, ytr, test_size=VAL_FRACTION, random_state=seed, stratify=ytr
    )

    sweep_rows, tuned_rows = [], []
    for name, paulis, ent in ENCODINGS:
        val_scores, test_scores = [], []
        for bw in BANDWIDTHS:
            v = _qsvm_score(Xin, yin, Xval, yval, n_qubits, paulis, ent, bw)
            t = _qsvm_score(Xtr, ytr, Xte, yte, n_qubits, paulis, ent, bw)
            val_scores.append(v)
            test_scores.append(t)
            sweep_rows.append(dict(dataset=tag, seed=seed, encoding=name,
                                   entangling_capacity=round(capacities[name], 4),
                                   bandwidth=bw, val_bacc=round(v, 4),
                                   test_bacc=round(t, 4)))
        best = int(np.argmax(val_scores))
        tuned_rows.append(dict(dataset=tag, seed=seed, encoding=name,
                               entangling_capacity=round(capacities[name], 4),
                               chosen_bandwidth=BANDWIDTHS[best],
                               val_bacc=round(val_scores[best], 4),
                               test_bacc=round(test_scores[best], 4),
                               test_bacc_at_bw1=round(
                                   test_scores[BANDWIDTHS.index(1.0)], 4)))

    # classical RBF control, equal tuning budget on the same validation split
    val_scores = [_rbf_score(Xin, yin, Xval, yval, g) for g in RBF_GAMMAS]
    best = int(np.argmax(val_scores))
    tuned_rows.append(dict(dataset=tag, seed=seed, encoding="RBF-SVM (tuned)",
                           entangling_capacity=np.nan,
                           chosen_bandwidth=RBF_GAMMAS[best],
                           val_bacc=round(val_scores[best], 4),
                           test_bacc=round(_rbf_score(Xtr, ytr, Xte, yte,
                                                      RBF_GAMMAS[best]), 4),
                           test_bacc_at_bw1=np.nan))
    return sweep_rows, tuned_rows


def main():
    t_start = time.time()
    sweep_all, tuned_all = [], []

    # ---------------- NoduleMNIST3D (8 qubits) ----------------
    caps8 = {n: meyer_wallach_capacity(8, p, e, n_samples=200, seed=42)
             for n, p, e in ENCODINGS}
    print("[capacity @ 8q] " + "  ".join(f"{k}={v:.3f}" for k, v in caps8.items()))
    for seed in SEEDS:
        d = load_nodulemnist3d(grid=2, n_per_class_train=150,
                               n_per_class_test=60, seed=seed)
        s, t = run_one_split(d["x_train"], d["y_train"], d["x_test"], d["y_test"],
                             8, seed, "NoduleMNIST3D", caps8)
        sweep_all += s
        tuned_all += t
        print(f"  [nodule seed {seed}] " + "  ".join(
            f"{r['encoding']}: bw={r['chosen_bandwidth']} test={r['test_bacc']:.3f}"
            for r in t))

    # ---------------- 2c checkerboard (3 qubits) ----------------
    caps3 = {n: meyer_wallach_capacity(3, p, e, n_samples=400, seed=42)
             for n, p, e in ENCODINGS}
    print("[capacity @ 3q] " + "  ".join(f"{k}={v:.3f}" for k, v in caps3.items()))
    for overlap in OVERLAPS:
        for seed in SEEDS:
            d = load_2c_overlap(overlap=overlap, samples_train=150,
                                samples_test=100, seed=seed)
            tag = f"2c overlap={overlap}"
            s, t = run_one_split(d["x_train"], d["y_train"], d["x_test"],
                                 d["y_test"], 3, seed, tag, caps3)
            sweep_all += s
            tuned_all += t
        print(f"  [2c overlap={overlap}] done")

    sweep = pd.DataFrame(sweep_all)
    tuned = pd.DataFrame(tuned_all)
    sweep.to_csv(OUT_DIR / "table_bandwidth_sweep.csv", index=False)
    tuned.to_csv(OUT_DIR / "table_bandwidth_tuned.csv", index=False)

    summary = (tuned.groupby(["dataset", "encoding"])
                    .agg(capacity=("entangling_capacity", "first"),
                         mean_test=("test_bacc", "mean"),
                         std_test=("test_bacc", "std"),
                         mean_test_bw1=("test_bacc_at_bw1", "mean"),
                         modal_bw=("chosen_bandwidth",
                                   lambda s: s.value_counts().index[0]),
                         n=("test_bacc", "size"))
                    .round(4).reset_index())
    summary.to_csv(OUT_DIR / "table_bandwidth_summary.csv", index=False)
    print(f"\nSaved 3 tables to {OUT_DIR}\n")
    print(summary.to_string(index=False))

    _paired(tuned)
    _plot(sweep)
    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")


def _paired(tuned):
    """Paired per-seed contrasts, tuned bandwidth, within each dataset."""
    print("\nPaired per-seed contrasts at validation-tuned bandwidth:")
    for tag, sub in tuned.groupby("dataset"):
        wide = sub.pivot_table(index="seed", columns="encoding", values="test_bacc")
        print(f"  {tag}:")
        pairs = [("Z (product)", "Z+ZZ linear"),
                 ("Z (product)", "Z+ZZ full"),
                 ("Z (product)", "RBF-SVM (tuned)"),
                 ("Z+ZZ full", "RBF-SVM (tuned)")]
        for a, b in pairs:
            if a in wide.columns and b in wide.columns:
                delta = (wide[b] - wide[a]).dropna()
                if len(delta) < 2:
                    continue
                se = delta.std(ddof=1) / np.sqrt(len(delta))
                sig = delta.mean() / se if se > 0 else np.nan
                print(f"    {b:16s} - {a:16s}: {delta.mean():+.4f} "
                      f"+/- {delta.std(ddof=1):.4f}  ({sig:+.1f} sigma)")


def _plot(sweep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = ["NoduleMNIST3D", "2c overlap=0.0", "2c overlap=0.5"]
    tags = [t for t in tags if t in set(sweep.dataset)]
    colors = {"Z (product)": "tab:blue", "Z+ZZ linear": "tab:orange",
              "Z+ZZ full": "tab:green"}
    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4.2),
                             squeeze=False)
    for ax, tag in zip(axes[0], tags):
        sub = sweep[sweep.dataset == tag]
        for name, _, _ in ENCODINGS:
            g = (sub[sub.encoding == name].groupby("bandwidth")
                 .agg(m=("test_bacc", "mean"), s=("test_bacc", "std")))
            ax.errorbar(g.index, g.m, yerr=g.s, marker="o", capsize=3,
                        label=name, color=colors[name], lw=1.5, ms=4)
        ax.set_xscale("log")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("bandwidth $\\lambda$")
        ax.set_ylabel("Test balanced accuracy")
        ax.set_title(tag, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Bandwidth sweep: test balanced accuracy vs. input scaling "
                 "(mean $\\pm$ std over 5 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_bandwidth_sweep.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_bandwidth_sweep.png'}")


if __name__ == "__main__":
    main()
