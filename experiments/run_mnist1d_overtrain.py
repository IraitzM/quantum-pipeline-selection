"""
run_mnist1d_overtrain.py
────────────────────────
Negative control: does the QNN-versus-classical gap on MNIST1D survive
prolonged training of the best-configuration quantum model?

The paper reports the best quantum encoding on MNIST1D (QSM, the model the
diagnostics select for a mid-bandwidth, lossy-pool signal) at $0.626 \pm 0.056$,
below both the raw-pixel RBF-SVM ($\sim 0.76$) and a tuned MLP. A natural
challenge is that the gap is an under-training artifact. This runs the exact
best-config QSM for far longer than the paper's 20 epochs, records a learning
curve on held-out validation and test data, and sets it against RBF and MLP
baselines at matched data.

The framing is a robustness check, not an advantage hunt. The expected result
is that QSM plateaus (validation accuracy peaks then flattens or declines as the
model overfits) well below the classical baselines, converting the "gap is a
ceiling, not an optimisation deficit" claim from an argument into a measurement.
Any apparent classical-beating result is to be treated as spurious until it
survives the five-seed resampling here.

Outputs:
    results/qnn-qsm/table_mnist1d_overtrain.csv        (per seed x checkpoint)
    results/qnn-qsm/table_mnist1d_overtrain_summary.csv
    results/qnn-qsm/figure_mnist1d_overtrain.png       (learning curve)
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # local package shadows stdlib ``code``

from code.baselines import rbf_svm_scores  # noqa: E402
from code.datasets import load_mnist1d  # noqa: E402
from code.qnn import HankelQSMClassifier, hankel_block_unitary  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-qsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = OUT_DIR / "table_mnist1d_overtrain.csv"
SUMMARY_CSV = OUT_DIR / "table_mnist1d_overtrain_summary.csv"

KEEP_CLASSES = [0, 1, 2, 3]
N_PER_CLASS_TRAIN = 150      # of which a fifth is held out for validation
N_PER_CLASS_TEST = 40
N_UPLOADS = 8
UPLOAD_T = 0.1
LR = 0.005                   # the QSM protocol's rate
BATCH = 16
MAX_EPOCHS = 100             # 5x the paper's 20 epochs; "prolonged"
CHECKPOINTS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 80, 100]


def _znorm(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)


def subsample(x, y, n_per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        idx.append(ci[:n_per_class])
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return x[idx], y[idx]


def _bacc(model, u, y):
    with torch.no_grad():
        return balanced_accuracy_score(y, model(u).argmax(1).numpy())


def qsm_learning_curve(u_tr, y_tr, u_va, y_va, u_te, y_te, n_classes, seed):
    torch.manual_seed(seed)
    model = HankelQSMClassifier(n_qubits=u_tr.shape[-1].bit_length() - 1,
                                n_uploads=N_UPLOADS, n_classes=n_classes)
    # n_qubits is log2(dim); dim is a power of two so bit_length-1 is exact.
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = u_tr.shape[0]
    curve = []
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss_fn(model(u_tr[idx]), yt[idx]).backward()
            opt.step()
        if epoch in CHECKPOINTS:
            curve.append(dict(epoch=epoch,
                              train=_bacc(model, u_tr, y_tr),
                              val=_bacc(model, u_va, y_va),
                              test=_bacc(model, u_te, y_te)))
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = [42]
        global MAX_EPOCHS, CHECKPOINTS
        MAX_EPOCHS, CHECKPOINTS = 3, [1, 2, 3]

    m = load_mnist1d()
    trm = np.isin(m["y_train"], KEEP_CLASSES)
    tem = np.isin(m["y_test"], KEEP_CLASSES)
    x_tr_full, y_tr_full = m["x_train"][trm], m["y_train"][trm]
    x_te_all, y_te_all = m["x_test"][tem], m["y_test"][tem]
    x_test, y_test = subsample(x_te_all, y_te_all, N_PER_CLASS_TEST, seed=42)

    rows, classical_rows = [], []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        x_all, y_all = subsample(x_tr_full, y_tr_full, N_PER_CLASS_TRAIN, seed)
        # 80/20 train/validation split, class order already shuffled
        n_val = len(y_all) // 5
        x_va, y_va = x_all[:n_val], y_all[:n_val]
        x_tr, y_tr = x_all[n_val:], y_all[n_val:]

        # QSM: per-sample z-score then Hankel upload
        t0 = time.time()
        u_tr = torch.tensor(hankel_block_unitary(_znorm(x_tr), t=UPLOAD_T), dtype=torch.complex128)
        u_va = torch.tensor(hankel_block_unitary(_znorm(x_va), t=UPLOAD_T), dtype=torch.complex128)
        u_te = torch.tensor(hankel_block_unitary(_znorm(x_test), t=UPLOAD_T), dtype=torch.complex128)
        curve = qsm_learning_curve(u_tr, y_tr, u_va, y_va, u_te, y_test, len(KEEP_CLASSES), seed)
        for c in curve:
            rows.append(dict(seed=seed, **c))
        pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)
        best = max(curve, key=lambda c: c["val"])
        print(f"  QSM: best-val epoch {best['epoch']} -> test {best['test']:.3f}; "
              f"final epoch {curve[-1]['epoch']} -> test {curve[-1]['test']:.3f} "
              f"(train {curve[-1]['train']:.3f})  [{time.time()-t0:.0f}s]", flush=True)

        # Classical baselines on the standard-scaled raw signal, matched data
        sc = StandardScaler().fit(x_tr)
        xz_tr, xz_te = sc.transform(x_tr), sc.transform(x_test)
        rbf = rbf_svm_scores(xz_tr, y_tr, xz_te, y_test, seed=seed)["test_bacc"]
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=seed)
        mlp.fit(xz_tr, y_tr)
        mlp_bacc = balanced_accuracy_score(y_test, mlp.predict(xz_te))
        classical_rows.append(dict(seed=seed, rbf=rbf, mlp=mlp_bacc))
        print(f"  classical: RBF {rbf:.3f}  MLP {mlp_bacc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(classical_rows)

    # Aggregate QSM at best-val epoch and at the final epoch, per seed.
    best_per_seed, final_per_seed = [], []
    for seed, g in df.groupby("seed"):
        best_per_seed.append(g.loc[g["val"].idxmax(), "test"])
        final_per_seed.append(g.loc[g["epoch"].idxmax(), "test"])
    summ = pd.DataFrame([
        dict(model="QSM (best-val, early stop)", test_bacc_mean=round(np.mean(best_per_seed), 4),
             test_bacc_std=round(np.std(best_per_seed), 4)),
        dict(model=f"QSM (final epoch {MAX_EPOCHS}, overtrained)",
             test_bacc_mean=round(np.mean(final_per_seed), 4),
             test_bacc_std=round(np.std(final_per_seed), 4)),
        dict(model="RBF-SVM (raw)", test_bacc_mean=round(cdf.rbf.mean(), 4),
             test_bacc_std=round(cdf.rbf.std(), 4)),
        dict(model="MLP (h=64)", test_bacc_mean=round(cdf.mlp.mean(), 4),
             test_bacc_std=round(cdf.mlp.std(), 4)),
    ])
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {RUNS_CSV}\nSaved {SUMMARY_CSV}")
    print(summ.to_string(index=False))
    _plot(df, cdf)


def _plot(df, cdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby("epoch").agg(["mean", "std"])
    ep = g.index.values
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, colour, lab in [("train", "#c9b3f5", "QSM train"),
                             ("val", "#7C3AED", "QSM validation"),
                             ("test", "tab:orange", "QSM test")]:
        ax.plot(ep, g[(col, "mean")], "o-", color=colour, label=lab)
        ax.fill_between(ep, g[(col, "mean")] - g[(col, "std")],
                        g[(col, "mean")] + g[(col, "std")], color=colour, alpha=0.15)
    ax.axhline(cdf.rbf.mean(), ls="--", color="k", lw=1, label=f"RBF raw {cdf.rbf.mean():.2f}")
    ax.axhline(cdf.mlp.mean(), ls="--", color="tab:gray", lw=1, label=f"MLP {cdf.mlp.mean():.2f}")
    ax.axhline(0.25, ls=":", color="gray", lw=1, label="chance (4-class)")
    ax.set_xlabel("Training epoch (log scale)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_title("MNIST1D: prolonged QSM training plateaus below classical")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_mnist1d_overtrain.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_mnist1d_overtrain.png'}")


if __name__ == "__main__":
    main()
