"""Multi-seed re-run of a real-dataset classification comparison.

The single-seed run (``run_classification.py``, seed 42) reports one split per
dataset, and several of the orderings it produces sit well inside a test-set
standard error. EEG was the first to be resampled: the QNN's apparent win over
the RBF-SVM (0.755 vs 0.725) dissolved into ``0.723 +/- 0.046`` vs
``0.735 +/- 0.013``. The same treatment is needed for the ECG and MNIST-1D
ablation orderings, where the reported best-variant margins are 0.018 and
0.081 respectively.

This script re-runs the four-variant ablation and the classical baselines on
one dataset across several seeds. What a seed reseeds depends on the dataset's
split protocol:

  * ``eeg``     -- train/test split (``load_eeg_seizure(seed=...)``), class
                   subsample and torch initialisation. Full resampling.
  * ``mnist1d`` -- the 150-per-class train and 40-per-class test subsamples
                   drawn from MNIST1D's fixed official split, and the torch
                   initialisation. Both train and test sets move.
  * ``ecg``     -- the 300-per-class training subsample and the torch
                   initialisation only. The test set is the *full* held-out
                   set from patient-disjoint records (203, 223) and is fixed
                   by construction, so the spread here measures training-draw
                   and optimisation variance, not test-set variance.

Seed 42 reproduces ``table_performance.csv`` exactly for EEG and MNIST-1D. It
does not for ECG: the single-split run drew the ECG subsample from a generator
already advanced by the MNIST-1D draws, so seed 42 here is an independent
training subsample rather than that same one.

Writes, for dataset tag ``<tag>``:
  * ``results/qnn-classification/table_<tag>_seeds.csv``          -- per-seed rows
  * ``results/qnn-classification/table_<tag>_seeds_summary.csv``  -- mean/std plus
    the paired QNN-minus-RBF contrast
  * ``results/qnn-classification/table_<tag>_seeds_contrasts.csv`` -- every paired
    contrast between two ablation variants, which is what the ablation
    ordering claims actually rest on

Runtime: roughly 12 minutes per seed on ECG and EEG, and closer to 30 on
MNIST-1D (4 classes means four read-out observables per forward pass), on a
20-core CPU box.

Usage:
    uv run experiments/run_classification_seeds.py --dataset ecg
    uv run experiments/run_classification_seeds.py --dataset mnist1d --seeds 42 43 44
    uv run experiments/run_classification_seeds.py --dataset eeg --summarise-only
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # local package shadows stdlib ``code``

from code.datasets import (  # noqa: E402
    load_ecg_beats,
    load_eeg_seizure,
    load_mnist1d,
)

from experiments.run_classification import (  # noqa: E402
    ABLATIONS,
    TorchMLP,
    rbf_svm_scores,
    subsample,
    train_qnn_classifier,
    train_torch_mlp,
)

OUT_DIR = ROOT / "results" / "qnn-classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]

MNIST1D_CLASSES = [0, 1, 2, 3]


# ── dataset builders ────────────────────────────────────────────────────────
#
# Each returns (x_train, y_train, x_test, y_test, n_classes) for one seed, and
# mirrors the corresponding block of ``run_classification.main``.


def build_eeg(seed: int):
    s = load_eeg_seizure(n_per_class_train=300, n_per_class_test=100, seed=seed)
    return s["x_train"], s["y_train"], s["x_test"], s["y_test"], 2


def build_mnist1d(seed: int):
    rng = np.random.default_rng(seed)
    m = load_mnist1d()
    train_mask = np.isin(m["y_train"], MNIST1D_CLASSES)
    test_mask = np.isin(m["y_test"], MNIST1D_CLASSES)
    x_train, y_train = subsample(
        m["x_train"][train_mask], m["y_train"][train_mask], n_per_class=150, rng=rng
    )
    x_test, y_test = subsample(
        m["x_test"][test_mask], m["y_test"][test_mask], n_per_class=40, rng=rng
    )
    return x_train, y_train, x_test, y_test, 4


def build_ecg(seed: int):
    rng = np.random.default_rng(seed)
    e = load_ecg_beats()
    x_train, y_train = subsample(e["x_train"], e["y_train"], n_per_class=300, rng=rng)
    return x_train, y_train, e["x_test"], e["y_test"], 2


DATASETS = {
    "eeg": ("EEG seizure (UCI 388)", build_eeg),
    "mnist1d": ("MNIST1D (4-class subset)", build_mnist1d),
    "ecg": ("ECG beats (N vs. PVC)", build_ecg),
}


def run_seed(tag: str, seed: int) -> list[dict]:
    title, builder = DATASETS[tag]
    x_train, y_train, x_test, y_test, n_classes = builder(seed)
    print(
        f"\n===== {title}, seed {seed} =====  "
        f"train={x_train.shape} test={x_test.shape} classes={n_classes}",
        flush=True,
    )
    rows = []

    for label, use_ea, use_em in ABLATIONS:
        t0 = time.time()
        res = train_qnn_classifier(
            x_train, y_train, x_test, y_test, n_classes, use_ea, use_em, seed=seed
        )
        rows.append(dict(seed=seed, model=f"QNN ({label})", **res))
        print(
            f"  QNN [{label}]: test_bacc={res['test_bacc']:.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    svm = rbf_svm_scores(x_train, y_train, x_test, y_test, seed=seed)
    rows.append(dict(seed=seed, model="RBF-SVM", **svm, train_time_s=None))
    print(f"  RBF-SVM: test_bacc={svm['test_bacc']:.3f} "
          f"(gamma={svm['gamma']:.3g})", flush=True)

    for hidden, name in [(2, "MLP (iso-param, h=2)"), (32, "MLP (h=32)")]:
        mlp = TorchMLP(in_dim=x_train.shape[1], hidden=hidden, n_classes=n_classes)
        res = train_torch_mlp(mlp, x_train, y_train, x_test, y_test)
        rows.append(dict(seed=seed, model=name, **res, train_time_s=None))
        print(f"  {name}: test_bacc={res['test_bacc']:.3f}", flush=True)

    return rows


# ── summarisation ───────────────────────────────────────────────────────────


def _paired(d: pd.Series) -> tuple[float, float, float]:
    """Mean, std and t-statistic of a per-seed paired difference."""
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    t = d.mean() / se if se else np.nan
    return d.mean(), sd, t


def summarise(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    summary = (
        df.groupby("model")["test_bacc"].agg(["mean", "std", "count"]).reset_index()
    )
    wide = df.pivot(index="seed", columns="model", values="test_bacc")

    # Paired contrast against the RBF-SVM, seed by seed: this is the quantity
    # the paper's "beats the RBF-SVM outright" claims actually need.
    ref = wide["RBF-SVM"]
    deltas = {
        model: _paired(wide[model] - ref)
        for model in wide.columns
        if model != "RBF-SVM"
    }
    for col, i in [
        ("paired_delta_vs_rbf", 0),
        ("paired_delta_std", 1),
        ("paired_t", 2),
    ]:
        summary[col] = summary["model"].map(
            lambda m, i=i: deltas.get(m, (np.nan,) * 3)[i]
        )
    summary.to_csv(OUT_DIR / f"table_{tag}_seeds_summary.csv", index=False)

    # Every paired contrast between two ablation variants: the ordering claims
    # in the entanglement section are all of this form, and on ECG the reported
    # margin between the top two variants is only 0.018.
    variants = [f"QNN ({label})" for label, _, _ in ABLATIONS if f"QNN ({label})" in wide]
    rows = []
    for a, b in itertools.permutations(variants, 2):
        d = wide[a] - wide[b]
        mean, sd, t = _paired(d)
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "delta_mean": mean,
                "delta_std": sd,
                "t": t,
                "seeds_a_wins": int((d > 0).sum()),
                "n_seeds": len(d),
            }
        )
    contrasts = pd.DataFrame(rows).sort_values("delta_mean", ascending=False)
    contrasts.to_csv(OUT_DIR / f"table_{tag}_seeds_contrasts.csv", index=False)

    print(f"\nSaved {OUT_DIR / f'table_{tag}_seeds_summary.csv'}")
    print(summary.to_string(index=False))
    print("\nPaired ablation-variant contrasts (a - b):")
    print(contrasts.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="eeg")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help="recompute the summary/contrast tables from an existing per-seed CSV",
    )
    args = parser.parse_args()
    tag = args.dataset
    per_seed_path = OUT_DIR / f"table_{tag}_seeds.csv"

    if args.summarise_only:
        df = pd.read_csv(per_seed_path)
    else:
        all_rows: list[dict] = []
        for seed in args.seeds:
            all_rows += run_seed(tag, seed)
        df = pd.DataFrame(all_rows)
        df.to_csv(per_seed_path, index=False)
        print(f"\nSaved {per_seed_path}")

    summarise(df, tag)


if __name__ == "__main__":
    main()
