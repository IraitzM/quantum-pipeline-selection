"""
run_3d_rank_matching.py
───────────────────────
Product read-out vs entangling ansatz on the genuinely 3-D ``2c`` checkerboard,
scored by reachable functional rank rather than by accuracy.

The question is the 3-qubit form of the rank-2 ceiling of the paper's 2-D
argument. A product-state read-out model ``f = sum_i w_i <O_i>`` has every
observable equal to a product of the single-coordinate marginals, so across a
single-qubit cut A|B the only functions of the A-coordinate it can build are
``1`` and that qubit's own marginal. Its rank across the cut is therefore at
most 2 no matter how high the correlator order climbs (local, +ZZ, +ZZZ). An
entangling gate makes ``<Z_A>`` itself a rich function of the A-coordinate and
lifts the cut toward the ``2L+1`` bandwidth cap. This run measures the reachable
cut-rank of each configuration (training-free, over random parameter draws) and
sets it against the rank the ``2c`` decision field actually needs.

The target rank is read off the Bayes decision field ``P(y=1 | x)`` rather than
the raw label field: sliding the clusters together (rising overlap) makes the
labels increasingly random, and a random field is full-rank noise, so the raw
label SVD would report rank *rising* with overlap. The denoised probability
field instead keeps its checkerboard rank and loses its *amplitude* as the
classes coincide, which is the honest description of what overlap does here.

Outputs:
    results/qnn-3d/table_rank_matching.csv
    results/qnn-3d/figure_rank_matching.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # the local package shadows stdlib ``code``

from code.baselines import svd_effective_rank  # noqa: E402
from code.datasets import load_2c_overlap  # noqa: E402
from code.rankdesign import (  # noqa: E402
    EntanglerReadoutClassifier,
    readout_span_cut_rank,
    recommended_placement,
)

OUT_DIR = ROOT / "results" / "qnn-3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 3
DEGREE = 6                       # bandwidth L; single-qubit cut caps at 2L+1 = 13
N_GRID = 2 * DEGREE + 4          # resolves the circuit's own frequencies
SEEDS = 4                        # random parameter draws for the reachable rank
CUTS = [(0,), (1,), (2,)]        # the three single-qubit bipartitions
OVERLAPS = [0.0, 0.3, 0.5, 0.7, 1.0]
RANK_TOL = 0.10                  # singular-value floor for the target hard-rank
RING = [(0, 1), (1, 2), (2, 0)]  # symmetric entangler, crosses every cut

# Read-out ladders on the product ansatz, and the entangling ansatz with a
# plain local read-out. ``entanglers`` empty means the product state.
LOCAL = [(0,), (1,), (2,)]
PAIRS = LOCAL + [(0, 1), (0, 2), (1, 2)]
TRIPLE = PAIRS + [(0, 1, 2)]


def ring_entanglers(n_layers: int, degree: int) -> list[tuple[int, tuple[int, int]]]:
    """The full ring on ``n_layers`` interior re-upload layers."""
    layers = recommended_placement(n_layers, degree)
    return [(layer, edge) for layer in layers for edge in RING]


CONFIGS = [
    ("product / local", [], LOCAL),
    ("product / +ZZ", [], PAIRS),
    ("product / +ZZZ", [], TRIPLE),
    ("entangling ring x1", ring_entanglers(1, DEGREE), LOCAL),
    ("entangling ring x2", ring_entanglers(2, DEGREE), LOCAL),
    ("entangling ring x3", ring_entanglers(3, DEGREE), LOCAL),
]


def reachable_ranks() -> list[dict]:
    """Reachable cut-rank of every configuration on each single-qubit cut."""
    rows = []
    for name, entanglers, observables in CONFIGS:
        rng = np.random.default_rng(1)
        per_cut = {}
        for cut in CUTS:
            r = readout_span_cut_rank(
                N_QUBITS, DEGREE, entanglers, observables, cut,
                seeds=SEEDS, n_grid=N_GRID, gate="crz", rng=rng,
            )
            per_cut[cut] = r
        rows.append(dict(
            config=name,
            entangled=bool(entanglers),
            **{f"reach_cut{c[0]}": per_cut[c] for c in CUTS},
            reach_min=min(per_cut.values()),
        ))
        print(f"  {name:20s} reachable rank per cut "
              f"{[per_cut[c] for c in CUTS]}  (min {min(per_cut.values())})")
    return rows


def bayes_field(overlap: float, grid: int = 8, n_big: int = 150_000,
                sigma: float = 0.8, seed: int = 0) -> tuple[np.ndarray, float]:
    """Denoised, mean-removed Bayes decision field ``P(y=1|x) - mean`` on a
    ``grid**3`` raster of the unit cube, plus its peak amplitude."""
    d = load_2c_overlap(overlap=overlap, samples_train=n_big, samples_test=1, seed=seed)
    X, y = d["x_train"], d["y_train"].astype(float)
    lo, hi = X.min(0), X.max(0)
    idx = np.clip(((X - lo) / (hi - lo + 1e-9) * grid).astype(int), 0, grid - 1)
    sums = np.zeros((grid,) * 3)
    cnts = np.zeros((grid,) * 3)
    np.add.at(sums, (idx[:, 0], idx[:, 1], idx[:, 2]), y)
    np.add.at(cnts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    p = np.where(cnts > 0, sums / np.maximum(cnts, 1), y.mean())
    p = gaussian_filter(p, sigma=sigma, mode="nearest")
    field = p - p.mean()
    return field, float(np.abs(field).max())


def target_ranks() -> list[dict]:
    """Rank the ``2c`` decision field needs on each cut, across overlap."""
    rows = []
    for overlap in OVERLAPS:
        field, amp = bayes_field(overlap)
        grid = field.shape[0]
        eff, hard = {}, {}
        for cut in CUTS:
            rest = [w for w in range(3) if w not in cut]
            m = np.transpose(field, list(cut) + rest).reshape(grid, grid * grid)
            s = np.linalg.svd(m, compute_uv=False)
            eff[cut] = round(svd_effective_rank(m), 2)
            hard[cut] = int(np.sum(s > RANK_TOL * s[0]))
        rows.append(dict(
            overlap=overlap,
            signal_amp=round(amp, 3),
            **{f"target_eff_cut{c[0]}": eff[c] for c in CUTS},
            **{f"target_hard_cut{c[0]}": hard[c] for c in CUTS},
            target_hard_max=max(hard.values()),
        ))
        print(f"  overlap {overlap:.1f}: hard-rank per cut "
              f"{[hard[c] for c in CUTS]}  amp {amp:.3f}")
    return rows


# ─── Fitted-accuracy companion ─────────────────────────────────────────────
#
# The rank study says the product read-out cannot represent the checkerboard;
# training the same circuits confirms the shortfall lands as an accuracy gap.
# Because bandwidth is invariant to the entangler, every configuration shares
# one input-scale grid and the comparison isolates rank. A scale is picked per
# configuration on a validation split so the product ansatz is shown at its best.

ACC_OVERLAPS = [0.0, 0.5]
SCALES = [1.0, 2.0, 4.0]
FIT_SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 70
LR = 0.05
N_TRAIN, N_VAL, N_TEST = 400, 200, 300

# The classical control gets the protocol the quantum arm gets: the same
# z-normalised inputs, the same train/validation split, and one hyperparameter
# selected on validation. Leaving it at sklearn's gamma="scale" default on raw
# features scored it at 0.635 on the overlap-0 checkerboard where a tuned RBF
# reaches 0.90, which made every fitted quantum configuration look like it had
# cleared a classical baseline it had not. The grid matches the eleven points
# run_kernel_bandwidth_sweep.py gives the same control on the same dataset.
RBF_GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def _fit(entanglers, observables, x_tr, y_tr, x_va, y_va, scale, seed):
    """Train one configuration at one input scale; return the model and its
    validation balanced accuracy."""
    torch.manual_seed(seed)
    model = EntanglerReadoutClassifier(DEGREE, entanglers, observables)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr * scale)
    yt = torch.tensor(y_tr)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(model(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        va = balanced_accuracy_score(
            y_va, model(torch.tensor(x_va * scale)).argmax(1).numpy())
    return model, va


def _split(overlap, seed):
    """One seed's z-normalised train/validation/test arrays.

    The validation partition is SHUFFLED. `load_2c_overlap` returns its rows
    grouped by class, so slicing the last `N_VAL` off the end hands the
    hyperparameter search a set whose class balance is 0.43 against the training
    half's 0.53, drawn from a subset of the clusters. Selecting on it is close to
    selecting at random: the RBF control picks a gamma that scores 0.51 on test
    where the same grid over a shuffled split reaches 0.92. Both arms select a
    hyperparameter this way, so both were affected.
    """
    d = load_2c_overlap(overlap=overlap, samples_train=N_TRAIN + N_VAL,
                        samples_test=N_TEST, seed=seed)
    X, y, Xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    idx = np.arange(len(y))
    np.random.RandomState(seed).shuffle(idx)
    tr, va = idx[:N_TRAIN], idx[N_TRAIN:]
    x_tr, y_tr, x_va, y_va = X[tr], y[tr], X[va], y[va]
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
    return ((x_tr - mu) / sd, y_tr, (x_va - mu) / sd, y_va, (Xte - mu) / sd, yte)


def fitted_accuracy() -> list[dict]:
    """Train every configuration on 2c and report train/test balanced accuracy,
    with a classical RBF-SVM reference, across overlaps."""
    rows = []
    for overlap in ACC_OVERLAPS:
        print(f"\n  overlap {overlap}:")
        for name, entanglers, observables in CONFIGS:
            tr_acc, te_acc = [], []
            for seed in FIT_SEEDS:
                x_tr_z, y_tr, x_va_z, y_va, x_te_z, yte = _split(overlap, seed)
                best_va, best = -1.0, None
                for scale in SCALES:
                    model, va = _fit(entanglers, observables, x_tr_z, y_tr,
                                     x_va_z, y_va, scale, seed)
                    if va > best_va:
                        best_va, best = va, (model, scale)
                model, scale = best
                with torch.no_grad():
                    tr_acc.append(balanced_accuracy_score(
                        y_tr, model(torch.tensor(x_tr_z * scale)).argmax(1).numpy()))
                    te_acc.append(balanced_accuracy_score(
                        yte, model(torch.tensor(x_te_z * scale)).argmax(1).numpy()))
            rows.append(dict(
                overlap=overlap, config=name, entangled=bool(entanglers),
                train_bacc=round(float(np.mean(tr_acc)), 3),
                test_bacc=round(float(np.mean(te_acc)), 3),
                test_std=round(float(np.std(te_acc)), 3),
                # per-seed, so contrasts against the classical control can be
                # paired on the draw rather than compared as two summaries
                **{f"seed_{s}": round(float(v), 4)
                   for s, v in zip(FIT_SEEDS, te_acc)},
            ))
            print(f"    {name:20s} train {np.mean(tr_acc):.3f}  "
                  f"test {np.mean(te_acc):.3f} +/- {np.std(te_acc):.3f}")
        # classical reference, given the quantum arm's protocol and tuning budget
        te_acc = []
        for seed in FIT_SEEDS:
            x_tr_z, y_tr, x_va_z, y_va, x_te_z, yte = _split(overlap, seed)
            best_va, best_gamma = -1.0, RBF_GAMMAS[0]
            for gamma in RBF_GAMMAS:
                svc = SVC(kernel="rbf", gamma=gamma,
                          class_weight="balanced").fit(x_tr_z, y_tr)
                va = balanced_accuracy_score(y_va, svc.predict(x_va_z))
                if va > best_va:
                    best_va, best_gamma = va, gamma
            svc = SVC(kernel="rbf", gamma=best_gamma,
                      class_weight="balanced").fit(x_tr_z, y_tr)
            te_acc.append(balanced_accuracy_score(yte, svc.predict(x_te_z)))
        rows.append(dict(overlap=overlap, config="RBF-SVM (classical)", entangled=None,
                         train_bacc=None, test_bacc=round(float(np.mean(te_acc)), 3),
                         test_std=round(float(np.std(te_acc)), 3),
                         **{f"seed_{s}": round(float(v), 4)
                            for s, v in zip(FIT_SEEDS, te_acc)}))
        print(f"    {'RBF-SVM (classical)':20s} test {np.mean(te_acc):.3f} "
              f"+/- {np.std(te_acc):.3f}")
    return rows


def main():
    t0 = time.time()
    print("Reachable cut-rank per configuration (training-free):")
    reach = reachable_ranks()
    print("\n2c decision-field rank need vs overlap:")
    target = target_ranks()
    print("\nFitted balanced accuracy on 2c (5 seeds, scale picked on validation):")
    acc = fitted_accuracy()

    import pandas as pd

    reach_df = pd.DataFrame(reach)
    target_df = pd.DataFrame(target)
    acc_df = pd.DataFrame(acc)
    reach_df.to_csv(OUT_DIR / "table_rank_matching.csv", index=False)
    target_df.to_csv(OUT_DIR / "table_rank_matching_target.csv", index=False)
    acc_df.to_csv(OUT_DIR / "table_rank_matching_accuracy.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'table_rank_matching.csv'}")
    print(reach_df.to_string(index=False))
    print()
    print(target_df.to_string(index=False))
    print()
    print(acc_df.to_string(index=False))

    _plot(reach_df, target_df)
    _plot_rank_accuracy(reach_df, acc_df)
    print(f"\nDone in {time.time() - t0:.0f}s")


def _plot_rank_accuracy(reach_df, acc_df):
    """Fitted accuracy by configuration on 2c at overlap 0.

    The reachable *field* rank (product pinned at 2) does not bound
    classification: the checkerboard's decision boundary is a product of
    per-axis signs, so the pairwise-correlator read-out reaches it as a rank-2
    field whose sign separates the classes, matching the entangling ansatz. The
    local read-out, lacking any product term, cannot represent the XOR and stays
    at chance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reach_by = dict(zip(reach_df.config, reach_df.reach_min))
    sub = acc_df[acc_df.overlap == 0.0].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = []
    for _, r in sub.iterrows():
        if r.config.startswith("RBF"):
            colours.append("tab:gray")
        elif r.entangled:
            colours.append("tab:blue")
        else:
            colours.append("tab:orange")
    x = np.arange(len(sub))
    ax.bar(x, sub.test_bacc, yerr=sub.test_std, color=colours, capsize=3)
    for xi, (_, r) in zip(x, sub.iterrows()):
        tag = "" if r.config.startswith("RBF") else f"field rank {reach_by[r.config]}"
        ax.annotate(tag, (xi, 0.02), fontsize=7, ha="center", color="white")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.config, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test balanced accuracy (overlap 0)")
    ax.set_title("On 2c the pairwise-correlator read-out matches entanglement")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
        plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, color="tab:gray"),
    ]
    ax.legend(handles, ["product read-out", "entangling ansatz", "classical RBF"],
              fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_rank_vs_accuracy.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_rank_vs_accuracy.png'}")


def _plot(reach_df, target_df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # The structured cuts need this much; take the max over the learnable
    # overlaps (amplitude well above zero) as the rank the data demands.
    learnable = target_df[target_df.signal_amp > 0.2]
    need_lo = int(learnable.target_hard_max.min())
    need_hi = int(learnable.target_hard_max.max())

    x = np.arange(len(reach_df))
    width = 0.25
    for i, cut in enumerate(CUTS):
        ax1.bar(x + (i - 1) * width, reach_df[f"reach_cut{cut[0]}"],
                width, label=f"cut {{{cut[0]}|rest}}")
    ax1.axhspan(need_lo, need_hi, color="tab:red", alpha=0.15,
                label=f"2c rank need ({need_lo}-{need_hi})")
    ax1.axhline(2, color="gray", linestyle=":", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(reach_df.config, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Reachable rank across cut")
    ax1.set_title("Product read-out is pinned at 2; entanglement climbs")
    ax1.legend(fontsize=8)

    for cut in CUTS:
        ax2.plot(target_df.overlap, target_df[f"target_hard_cut{cut[0]}"],
                 "o-", label=f"cut {{{cut[0]}|rest}} rank")
    ax2.axhline(2, color="gray", linestyle=":", linewidth=1,
                label="product ceiling")
    ax2b = ax2.twinx()
    ax2b.plot(target_df.overlap, target_df.signal_amp, "k--^",
              label="signal amplitude")
    ax2b.set_ylabel("Decision-field amplitude")
    ax2.set_xlabel("Class overlap")
    ax2.set_ylabel("2c decision-field rank need")
    ax2.set_title("Overlap kills amplitude, not rank")
    ax2.legend(fontsize=8, loc="upper left")
    ax2b.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_rank_matching.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_rank_matching.png'}")


if __name__ == "__main__":
    main()
