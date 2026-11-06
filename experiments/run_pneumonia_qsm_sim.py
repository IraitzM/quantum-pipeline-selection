"""
run_pneumonia_qsm_sim.py
────────────────────────
Simulation-only feasibility ablation: run the Hankel QSM on PneumoniaMNIST,
the dataset the paper marks "QSM infeasible".

The paper rules QSM out on PneumoniaMNIST because its d x d Hankel lift makes
the upload an O(d^2)-gate state preparation on hardware (d = 784), against
amplitude embedding's Theta(d). That verdict is about the hardware resource
ledger, not the simulator: at d = 784 the block Hamiltonian is 785 x 785,
padded to 2^10, so QSM is a 10-qubit statevector that simulates fine. This
runner grants QSM that free state preparation and asks the capability question
the cost verdict leaves open — with the encoding handed to it for nothing, does
QSM beat amplitude embedding on these images, or does its spectral trick fail to
transfer from 1-D oscillatory signals to flattened X-rays?

Nothing here is a hardware-feasible result. Any number produced is labelled
QSM (simulated, hardware-infeasible) and is only meaningful next to the
amplitude embedding and RBF references trained on the same subsample.

Two costs force a smaller subsample than the amplitude table used: each upload
unitary is 1024 x 1024 complex128 (16.8 MB) and each needs an eigendecomposition
of the block Hamiltonian, so uploads are precomputed in chunks to cap peak
memory. The comparison is therefore reported at matched N.

Outputs:
    results/qnn-qsm/table_pneumonia_qsm_sim.csv  (per run, incremental)
    results/qnn-qsm/table_pneumonia_qsm_summary.csv
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

from code.baselines import rbf_svm_scores  # noqa: E402
from code.datasets import load_pneumoniamnist  # noqa: E402
from code.qnn import (  # noqa: E402
    AmplitudeEncodingClassifier,
    HankelQSMClassifier,
    hankel_block_unitary,
    to_nonneg_domain,
)

OUT_DIR = ROOT / "results" / "qnn-qsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = OUT_DIR / "table_pneumonia_qsm_sim.csv"
SUMMARY_CSV = OUT_DIR / "table_pneumonia_qsm_summary.csv"

N_UPLOADS = 8
UPLOAD_T = 0.1
QSM_EPOCHS = 20
QSM_LR = 0.005          # the lr=0.05 QSM trap; Wang used 0.01, screen picked 0.005
AMP_EPOCHS = 15
AMP_LR = 0.05
BATCH = 16
PRECOMPUTE_CHUNK = 32   # uploads built in blocks to cap peak memory

# Reference numbers from the paper's amplitude table (600/300 split), for
# context only. The in-run amplitude/RBF columns are the matched-N comparison.
REF_AMPLITUDE_600 = 0.840
REF_RBF_RAW_600 = 0.837
REF_ANGLE_POOLED = 0.730


def _znorm(a: np.ndarray) -> np.ndarray:
    """Per-sample z-score, matching the QSM upload's assumed input."""
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)


def precompute_uploads(x: np.ndarray, t: float, chunk: int = PRECOMPUTE_CHUNK) -> torch.Tensor:
    """Hankel block-unitaries for every row of ``x``, built in chunks so peak
    memory stays at ``chunk`` unitaries rather than the whole set."""
    n = x.shape[0]
    dim = 1 << int(np.ceil(np.log2(x.shape[1] + 1)))
    out = torch.empty((n, dim, dim), dtype=torch.complex128)
    for s in range(0, n, chunk):
        u = hankel_block_unitary(x[s:s + chunk], t=t)
        out[s:s + chunk] = torch.from_numpy(u)
        del u
    return out


def train_qsm(x_tr, y_tr, x_te, y_te, seed):
    torch.manual_seed(seed)
    n_qubits = HankelQSMClassifier.n_qubits_for(x_tr.shape[1])
    t0 = time.time()
    u_tr = precompute_uploads(_znorm(x_tr), UPLOAD_T)
    u_te = precompute_uploads(_znorm(x_te), UPLOAD_T)
    precompute_s = time.time() - t0

    model = HankelQSMClassifier(n_qubits=n_qubits, n_uploads=N_UPLOADS, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=QSM_LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = u_tr.shape[0]
    t0 = time.time()
    for _ in range(QSM_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad()
            loss_fn(model(u_tr[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        tr = balanced_accuracy_score(y_tr, model(u_tr).argmax(1).numpy())
        te = balanced_accuracy_score(y_te, model(u_te).argmax(1).numpy())
    return dict(train_bacc=tr, test_bacc=te, n_qubits=n_qubits,
                precompute_s=round(precompute_s, 1),
                train_time_s=round(time.time() - t0, 1))


def train_amplitude(x_tr, y_tr, x_te, y_te, seed):
    torch.manual_seed(seed)
    x_tr_d, x_te_d = to_nonneg_domain(x_tr), to_nonneg_domain(x_te)
    model = AmplitudeEncodingClassifier(input_dim=x_tr.shape[1], n_layers=3, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=AMP_LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr_d, dtype=torch.float64)
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = xt.shape[0]
    for _ in range(AMP_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        te = balanced_accuracy_score(
            y_te, model(torch.tensor(x_te_d, dtype=torch.float64)).argmax(1).numpy())
    return dict(test_bacc=te, n_qubits=model.n_qubits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-per-class-train", type=int, default=150)
    parser.add_argument("--n-per-class-test", type=int, default=75)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny fast run to check the pipeline end to end")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = [42]
        args.n_per_class_train, args.n_per_class_test = 12, 8
        global QSM_EPOCHS, AMP_EPOCHS
        QSM_EPOCHS, AMP_EPOCHS = 2, 2

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} "
              f"({args.n_per_class_train}/class train, {args.n_per_class_test}/class test) ===",
              flush=True)
        d = load_pneumoniamnist(n_per_class_train=args.n_per_class_train,
                                n_per_class_test=args.n_per_class_test, seed=seed)
        x_tr = d["x_train"].reshape(len(d["x_train"]), -1)
        x_te = d["x_test"].reshape(len(d["x_test"]), -1)
        y_tr, y_te = d["y_train"], d["y_test"]

        t0 = time.time()
        q = train_qsm(x_tr, y_tr, x_te, y_te, seed)
        print(f"  QSM (sim):  train {q['train_bacc']:.3f}  test {q['test_bacc']:.3f}  "
              f"({q['n_qubits']}q, precompute {q['precompute_s']}s, "
              f"train {q['train_time_s']}s)", flush=True)

        a = train_amplitude(x_tr, y_tr, x_te, y_te, seed)
        print(f"  amplitude:  test {a['test_bacc']:.3f} ({a['n_qubits']}q)", flush=True)

        rbf = rbf_svm_scores(x_tr, y_tr, x_te, y_te, seed=seed)["test_bacc"]
        print(f"  RBF (raw):  test {rbf:.3f}", flush=True)

        rows.append(dict(seed=seed, n_train=len(y_tr), n_test=len(y_te),
                         qsm_sim_test_bacc=round(q["test_bacc"], 4),
                         qsm_sim_train_bacc=round(q["train_bacc"], 4),
                         amplitude_test_bacc=round(a["test_bacc"], 4),
                         rbf_raw_test_bacc=round(rbf, 4),
                         qsm_qubits=q["n_qubits"],
                         total_s=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)  # incremental guard

    df = pd.DataFrame(rows)
    summ = pd.DataFrame([{
        "model": "QSM (simulated, hardware-infeasible)",
        "test_bacc_mean": round(df.qsm_sim_test_bacc.mean(), 4),
        "test_bacc_std": round(df.qsm_sim_test_bacc.std(), 4),
        "train_bacc_mean": round(df.qsm_sim_train_bacc.mean(), 4),
    }, {
        "model": "amplitude embedding (matched N)",
        "test_bacc_mean": round(df.amplitude_test_bacc.mean(), 4),
        "test_bacc_std": round(df.amplitude_test_bacc.std(), 4),
        "train_bacc_mean": None,
    }, {
        "model": "RBF-SVM raw pixels (matched N)",
        "test_bacc_mean": round(df.rbf_raw_test_bacc.mean(), 4),
        "test_bacc_std": round(df.rbf_raw_test_bacc.std(), 4),
        "train_bacc_mean": None,
    }])
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {RUNS_CSV}\nSaved {SUMMARY_CSV}")
    print(df.to_string(index=False))
    print()
    print(summ.to_string(index=False))
    print(f"\nContext (paper, 600/300 split): amplitude {REF_AMPLITUDE_600}, "
          f"RBF raw {REF_RBF_RAW_600}, angle pooled {REF_ANGLE_POOLED}")
    _plot(df)


def _plot(df):
    """QSM's train/test gap next to the matched-N amplitude and RBF baselines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["QSM sim\n(train)", "QSM sim\n(test)", "amplitude\n(test)", "RBF raw\n(test)"]
    cols = [df.qsm_sim_train_bacc, df.qsm_sim_test_bacc,
            df.amplitude_test_bacc, df.rbf_raw_test_bacc]
    means = [c.mean() for c in cols]
    stds = [c.std() for c in cols]
    colours = ["#c9b3f5", "#7C3AED", "tab:orange", "tab:gray"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(range(4), means, yerr=stds, color=colours, capsize=3)
    ax.axhline(0.5, ls=":", color="gray", lw=1, label="chance")
    ax.axhline(REF_AMPLITUDE_600, ls="--", color="k", lw=1,
               label=f"amplitude, paper 600/300 = {REF_AMPLITUDE_600}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Simulated QSM on PneumoniaMNIST: fits train, fails to transfer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_pneumonia_qsm_sim.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_pneumonia_qsm_sim.png'}")


if __name__ == "__main__":
    main()
