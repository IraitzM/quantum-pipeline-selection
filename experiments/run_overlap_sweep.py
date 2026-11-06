"""
run_overlap_sweep.py
─────────────────────
Independent of the periodic-signal story (MNIST1D/ECG), this experiment asks
a more basic question: how do the angle-encoding QNN and classical baselines
perform "in the first place" as raw classification difficulty increases?

We use the '2c' template from ../kernel-trainer (kernel_trainer.dataset.
DataGenerator): a 4x4 checkerboard of 3-D ellipsoid clusters whose label
pattern is inherently non-linear (XOR-like) at any overlap. The ``overlap``
parameter continuously slides the 16 cluster centres towards the cube
centre, letting us sweep from well-separated to fully overlapping classes.

This is also where the entanglement ablation matters most: with only 3
input features spread across 3 qubits, a product-state (no entangling
gates) circuit followed by a *linear* classical read-out can only represent
an additive function of the three per-qubit Fourier features -- it cannot
express the multiplicative feature interactions a checkerboard partition
needs. An entangling ansatz, by contrast, can mix information across wires
before read-out. We therefore expect entanglement to matter much more here
than it did for the single-channel ECG signal.

Outputs:
    results/qnn-overlap/table_performance.csv
    results/qnn-overlap/figure_overlap_sweep.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.baselines import TorchMLP, rbf_svm_scores, train_torch_mlp  # noqa: E402
from code.datasets import load_2c_overlap  # noqa: E402
from code.qnn import AngleEncodingClassifier  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAPS = [0.0, 0.5, 0.85, 0.95]
SEED = 42
N_QUBITS = 3
N_REUPLOADS = 4
EPOCHS = 12
LR = 0.05
BATCH_SIZE = 16
SAMPLES_TRAIN = 600
SAMPLES_TEST = 300

ABLATIONS = [
    ("entangled ansatz + local meas.", True, False),
    ("entangled ansatz + entangled meas.", True, True),
    ("product ansatz + local meas.", False, False),
    ("product ansatz + entangled meas.", False, True),
]


def scale_to_angles(x_train, x_test):
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def train_qnn(x_train, y_train, x_test, y_test, use_entangling_ansatz, use_entangling_measurement, seed=SEED):
    torch.manual_seed(seed)
    model = AngleEncodingClassifier(
        seq_len=x_train.shape[1],
        n_qubits=N_QUBITS,
        n_reuploads=N_REUPLOADS,
        n_classes=2,
        use_entangling_ansatz=use_entangling_ansatz,
        use_entangling_measurement=use_entangling_measurement,
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
            logits = model(xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred_train = model(xt).argmax(dim=1).numpy()
        pred_test = model(torch.tensor(x_test, dtype=torch.float64)).argmax(dim=1).numpy()

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "train_bacc": balanced_accuracy_score(y_train, pred_train),
        "test_bacc": balanced_accuracy_score(y_test, pred_test),
        "test_f1": f1_score(y_test, pred_test, average="macro"),
        "n_params": n_params,
        "train_time_s": round(time.time() - t0, 1),
    }


def main():
    rows = []
    for overlap in OVERLAPS:
        print(f"\n=== overlap={overlap} ===")
        data = load_2c_overlap(
            overlap=overlap, samples_train=SAMPLES_TRAIN, samples_test=SAMPLES_TEST, seed=SEED
        )
        x_train, x_test = scale_to_angles(data["x_train"], data["x_test"])
        y_train, y_test = data["y_train"], data["y_test"]

        for label, use_ea, use_em in ABLATIONS:
            res = train_qnn(x_train, y_train, x_test, y_test, use_ea, use_em)
            rows.append(dict(overlap=overlap, model=f"QNN ({label})", **res))
            print(f"  QNN [{label}]: test_bacc={res['test_bacc']:.3f} f1={res['test_f1']:.3f} "
                  f"params={res['n_params']} ({res['train_time_s']}s)")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        lr_bacc = cross_val_score(
            LogisticRegression(max_iter=500), x_train, y_train, cv=cv, scoring="balanced_accuracy"
        ).mean()
        lr_clf = LogisticRegression(max_iter=500).fit(x_train, y_train)
        lr_test_bacc = balanced_accuracy_score(y_test, lr_clf.predict(x_test))
        rows.append(dict(overlap=overlap, model="LR", train_bacc=lr_bacc, test_bacc=lr_test_bacc,
                          test_f1=f1_score(y_test, lr_clf.predict(x_test), average="macro"),
                          n_params=None, train_time_s=None))
        print(f"  LR: test_bacc={lr_test_bacc:.3f}")

        svm_res = rbf_svm_scores(x_train, y_train, x_test, y_test)
        rows.append(dict(overlap=overlap, model="RBF-SVM", **svm_res, train_time_s=None))
        print(f"  RBF-SVM: test_bacc={svm_res['test_bacc']:.3f}")

        mlp = TorchMLP(in_dim=x_train.shape[1], hidden=8, n_classes=2)
        mlp_res = train_torch_mlp(mlp, x_train, y_train, x_test, y_test)
        rows.append(dict(overlap=overlap, model="MLP (h=8)", **mlp_res, train_time_s=None))
        print(f"  MLP(h=8): test_bacc={mlp_res['test_bacc']:.3f} params={mlp_res['n_params']}")

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_performance.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'table_performance.csv'}")
    print(df.to_string(index=False))

    _plot(df)


def _plot(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in df.model.unique():
        sub = df[df.model == model].sort_values("overlap")
        style = "--" if model.startswith("QNN") else "-"
        marker = "s" if "product" in model else ("o" if model.startswith("QNN") else "^")
        ax.plot(sub.overlap, sub.test_bacc, style, marker=marker, label=model)
    ax.set_xlabel("Class overlap")
    ax.set_ylabel("Test balanced accuracy")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("2c checkerboard: performance vs. class overlap")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_overlap_sweep.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_overlap_sweep.png'}")


if __name__ == "__main__":
    main()
