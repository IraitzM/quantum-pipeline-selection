"""
run_3d_entangling_capacity_seeds.py
────────────────────────────────────
Multi-seed version of run_nodule3d_entangling_capacity.py and the overlap-0
arm of run_3d_entangling_capacity.py, to put error bars on the claim that
feature-map entangling capacity helps.

Motivation: the single-seed NoduleMNIST3D run reports a monotone rise of
0.575 -> 0.600 -> 0.608 across the three encodings, a spread of 0.033 on a
120-sample test set whose per-encoding standard error is roughly 0.045. The
ordering therefore cannot be asserted from one split. This script resamples
train/test per seed and reports mean +/- std, plus the paired per-seed
difference (full - product), which is the quantity the claim rests on.

The protocol is a kernel-swap control in the sense of Shen (arXiv:2607.20168):
every encoding and the classical RBF reference share the same per-seed split,
the same solver (SVC, class_weight='balanced'), and the same lack of
hyperparameter tuning, so only the kernel is exchanged.

Outputs:
    results/qnn-3d/table_capacity_seeds.csv     (per run)
    results/qnn-3d/table_capacity_summary.csv   (mean +/- std, paired delta)
    results/qnn-3d/figure_nodule3d_seeds.png    (bars with error bars + points)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.datasets import load_nodulemnist3d  # noqa: E402
from code.qkernel import (  # noqa: E402
    ENCODINGS,
    kernel_matrix,
    make_fidelity_kernel,
    meyer_wallach_capacity,
)

OUT_DIR = ROOT / "results" / "qnn-3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = OUT_DIR / "table_capacity_seeds.csv"
SUMMARY_CSV = OUT_DIR / "table_capacity_summary.csv"

SEEDS = [42, 43, 44, 45, 46]
N_QUBITS = 8
CAP_SEED = 42  # capacity is data-independent; fix its sampling seed


def run_nodule_seed(seed, capacities):
    """One NoduleMNIST3D split: all three quantum kernels + classical refs."""
    d = load_nodulemnist3d(grid=2, n_per_class_train=150, n_per_class_test=60,
                           seed=seed)
    Xtr, ytr, Xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]

    rows = []
    for name, paulis, ent in ENCODINGS:
        t0 = time.time()
        kf = make_fidelity_kernel(N_QUBITS, paulis, ent)
        Ktr = kernel_matrix(kf, Xtr)
        Kte = kernel_matrix(kf, Xte, Xtr)
        svc = SVC(kernel="precomputed", class_weight="balanced").fit(Ktr, ytr)
        bacc = balanced_accuracy_score(yte, svc.predict(Kte))
        rows.append(dict(seed=seed, encoding=name,
                         entangling_capacity=round(capacities[name], 4),
                         model="QSVM", test_bacc=round(bacc, 4),
                         time_s=round(time.time() - t0, 1)))
        print(f"  [seed {seed}] QSVM [{name:14s}] test_bacc={bacc:.3f}")

    svc = SVC(kernel="rbf", gamma="scale", class_weight="balanced").fit(Xtr, ytr)
    bacc = balanced_accuracy_score(yte, svc.predict(Xte))
    rows.append(dict(seed=seed, encoding="RBF-SVM (8 pooled feat.)",
                     entangling_capacity=np.nan, model="RBF-SVM",
                     test_bacc=round(bacc, 4), time_s=None))
    print(f"  [seed {seed}] RBF-SVM (8 pooled feat.): test_bacc={bacc:.3f}")

    svc = SVC(kernel="rbf", gamma="scale",
              class_weight="balanced").fit(d["x_train_full"], ytr)
    bacc = balanced_accuracy_score(yte, svc.predict(d["x_test_full"]))
    rows.append(dict(seed=seed, encoding="RBF-SVM (raw full volume)",
                     entangling_capacity=np.nan, model="RBF-SVM",
                     test_bacc=round(bacc, 4), time_s=None))
    print(f"  [seed {seed}] RBF-SVM (raw full volume): test_bacc={bacc:.3f}")
    return rows


def main():
    capacities = {}
    for name, paulis, ent in ENCODINGS:
        cap = meyer_wallach_capacity(N_QUBITS, paulis, ent, n_samples=200,
                                     seed=CAP_SEED)
        capacities[name] = cap
        print(f"[capacity] {name:14s} Meyer-Wallach = {cap:.4f}")

    all_rows = []
    for seed in SEEDS:
        print(f"\n=== NoduleMNIST3D, seed={seed} ===")
        all_rows += run_nodule_seed(seed, capacities)
        pd.DataFrame(all_rows).to_csv(RUNS_CSV, index=False)  # incremental save

    df = pd.DataFrame(all_rows)
    df.to_csv(RUNS_CSV, index=False)
    print(f"\nSaved {RUNS_CSV}")

    summary = (df.groupby("encoding")
                 .agg(entangling_capacity=("entangling_capacity", "first"),
                      mean_bacc=("test_bacc", "mean"),
                      std_bacc=("test_bacc", "std"),
                      n_seeds=("test_bacc", "size"))
                 .reset_index()
                 .sort_values("mean_bacc"))
    summary["mean_bacc"] = summary["mean_bacc"].round(4)
    summary["std_bacc"] = summary["std_bacc"].round(4)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}\n")
    print(summary.to_string(index=False))

    # Paired per-seed contrasts: the quantity the monotonicity claim rests on.
    wide = df.pivot_table(index="seed", columns="encoding", values="test_bacc")
    print("\nPaired per-seed differences (positive = more capacity better):")
    for a, b in [("Z (product)", "Z+ZZ linear"),
                 ("Z (product)", "Z+ZZ full"),
                 ("Z+ZZ linear", "Z+ZZ full")]:
        if a in wide.columns and b in wide.columns:
            delta = wide[b] - wide[a]
            se = delta.std(ddof=1) / np.sqrt(len(delta))
            sigma = delta.mean() / se if se > 0 else float("nan")
            print(f"  {b:14s} - {a:14s}: {delta.mean():+.4f} "
                  f"+/- {delta.std(ddof=1):.4f} (paired, {sigma:+.1f} sigma)")
    for q in ["Z (product)", "Z+ZZ linear", "Z+ZZ full"]:
        if q in wide.columns and "RBF-SVM (8 pooled feat.)" in wide.columns:
            delta = wide[q] - wide["RBF-SVM (8 pooled feat.)"]
            se = delta.std(ddof=1) / np.sqrt(len(delta))
            sigma = delta.mean() / se if se > 0 else float("nan")
            print(f"  {q:14s} - RBF (pooled)  : {delta.mean():+.4f} "
                  f"+/- {delta.std(ddof=1):.4f} (paired, {sigma:+.1f} sigma)")

    _plot(df)


def _plot(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["Z (product)", "Z+ZZ linear", "Z+ZZ full",
             "RBF-SVM (8 pooled feat.)", "RBF-SVM (raw full volume)"]
    colors = {"Z (product)": "tab:blue", "Z+ZZ linear": "tab:orange",
              "Z+ZZ full": "tab:green", "RBF-SVM (8 pooled feat.)": "tab:gray",
              "RBF-SVM (raw full volume)": "tab:red"}
    g = (df.groupby("encoding")
           .agg(m=("test_bacc", "mean"), s=("test_bacc", "std"),
                cap=("entangling_capacity", "first"))
           .reindex(order))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(order))
    ax.bar(x, g.m, yerr=g.s, capsize=5, color=[colors[o] for o in order],
           error_kw=dict(ecolor="black", lw=1.2))
    jitter = np.random.default_rng(0)
    for i, enc in enumerate(order):  # overlay individual splits
        pts = df.loc[df.encoding == enc, "test_bacc"].values
        ax.scatter(np.full_like(pts, i, dtype=float)
                   + jitter.uniform(-0.13, 0.13, len(pts)),
                   pts, color="black", s=14, zorder=3, alpha=0.7)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(len(order) - 0.4, 0.505, "chance", color="gray", fontsize=8)

    labels = []
    for o in order:
        cap = f"\ncap={g.loc[o, 'cap']:.2f}" if not np.isnan(g.loc[o, "cap"]) else ""
        labels.append(o.replace(" (", "\n(") + cap)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test balanced accuracy")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("NoduleMNIST3D (real 3-D CT): QSVM by feature-map entangling "
                 "capacity\nmean $\\pm$ std over 5 resampled splits "
                 "(dots = individual splits)", fontsize=10)
    for i, o in enumerate(order):
        ax.text(i, g.loc[o, "m"] + g.loc[o, "s"] + 0.012,
                f"{g.loc[o, 'm']:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_nodule3d_seeds.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'figure_nodule3d_seeds.png'}")


if __name__ == "__main__":
    main()
