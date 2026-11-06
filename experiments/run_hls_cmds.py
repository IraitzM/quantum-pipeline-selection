"""
run_hls_cmds.py
───────────────
HLS-CMDS heart-vs-lung, run on two representations side by side to expose a
representation limitation of the time-domain Fourier-encoding framework.

Unaligned stethoscope audio is not discriminative in the raw time domain (its
information lives in the spectral envelope, and arbitrary windows scramble the
phase), so on the raw waveform every model sits at chance. A classical spectral
transform makes the task learnable. This runs the three quantum encoders
(QSM, amplitude, pooled angle) against RBF and MLP baselines on both the raw
waveform and the log-power-spectral envelope, at a QSM-cheap length (d = 40,
6 qubits), five seeds, with a recording-level train/test split.

The expected reading: on the waveform all five hover at chance; on the spectral
envelope the task is learnable and the classical baselines lead, with the
quantum encoders operating on features a classical transform already produced.
The limitation is of the framework, and the fix is classical.

Outputs:
    results/qnn-hls/table_hls_cmds.csv
    results/qnn-hls/table_hls_cmds_summary.csv
    results/qnn-hls/figure_hls_cmds.png
"""

from __future__ import annotations

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.pop("code", None)

from code.datasets import load_hls_cmds  # noqa: E402
from run_encoding_selection import dataset_diagnostics  # noqa: E402
from code.qnn import (  # noqa: E402
    AmplitudeEncodingClassifier,
    AngleEncodingClassifier,
    HankelQSMClassifier,
    hankel_block_unitary,
    to_nonneg_domain,
)

OUT_DIR = ROOT / "results" / "qnn-hls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D = 40
SEEDS = [42, 43, 44, 45, 46]
REPRESENTATIONS = ["waveform", "spectral"]
BATCH = 16


def _znorm(a):
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)


def diagnostics(X, n_pool=8):
    """omega_max, omega_eff and Delta E under the paper's definitions.

    This delegates to the shared implementation rather than repeating it. An
    earlier private version here disagreed with the paper on all three: it added
    one to the prefix-band index, so the waveform reported omega_eff = 21 against
    a Nyquist of 20 and broke the omega_eff <= omega_max ordering; it read
    omega_max off a 1%-of-peak amplitude rule instead of the top-K-at-95%-energy
    set of Eq. (omega-max); and it measured Delta E as a time-domain pooling
    residual rather than the out-of-band energy of Eq. (pool-out-energy). It also
    averaged one spectrum over the dataset where the other rows of the prediction
    table take a median over per-signal diagnostics, so the HLS rows were not
    comparable with the rest of the table.
    """
    d = dataset_diagnostics(X, n_pool)
    return d["omega_max"], d["omega_eff"], round(d["delta_e"], 3)


def _train_torch(model, x_tr, y_tr, x_te, y_te, dtype, epochs, lr):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x_tr, dtype=dtype)
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = xt.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        return balanced_accuracy_score(
            y_te, model(torch.tensor(x_te, dtype=dtype)).argmax(1).numpy())


def run_qsm(x_tr, y_tr, x_te, y_te, seed):
    torch.manual_seed(seed)
    nq = HankelQSMClassifier.n_qubits_for(x_tr.shape[1])
    u_tr = torch.tensor(hankel_block_unitary(_znorm(x_tr), t=0.1), dtype=torch.complex128)
    u_te = torch.tensor(hankel_block_unitary(_znorm(x_te), t=0.1), dtype=torch.complex128)
    model = HankelQSMClassifier(n_qubits=nq, n_uploads=8, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = u_tr.shape[0]
    for _ in range(20):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss_fn(model(u_tr[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        return balanced_accuracy_score(y_te, model(u_te).argmax(1).numpy())


def run_amplitude(x_tr, y_tr, x_te, y_te, seed):
    torch.manual_seed(seed)
    model = AmplitudeEncodingClassifier(input_dim=x_tr.shape[1], n_layers=3, n_classes=2)
    return _train_torch(model, to_nonneg_domain(x_tr), y_tr, to_nonneg_domain(x_te),
                        y_te, torch.float64, epochs=15, lr=0.05)


def run_angle(x_tr, y_tr, x_te, y_te, seed):
    torch.manual_seed(seed)
    model = AngleEncodingClassifier(seq_len=x_tr.shape[1], n_qubits=8, n_reuploads=4,
                                    n_classes=2, device_name="default.qubit").double()
    return _train_torch(model, x_tr, y_tr, x_te, y_te, torch.float64, epochs=15, lr=0.05)


def run_classical(x_tr, y_tr, x_te, y_te, seed):
    sc = StandardScaler().fit(x_tr)
    xz_tr, xz_te = sc.transform(x_tr), sc.transform(x_te)
    from code.baselines import rbf_svm_scores
    rbf = rbf_svm_scores(xz_tr, y_tr, xz_te, y_te, seed=seed)["test_bacc"]
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=seed).fit(xz_tr, y_tr)
    return rbf, balanced_accuracy_score(y_te, mlp.predict(xz_te))


ENCODERS = [("QSM", run_qsm), ("amplitude", run_amplitude), ("angle (pooled)", run_angle)]


def main():
    rows, diag_rows = [], []
    for rep in REPRESENTATIONS:
        d0 = load_hls_cmds(target_len=D, representation=rep, seed=42)
        om, oe, dE = diagnostics(d0["x_train"])
        diag_rows.append(dict(representation=rep, omega_max=om, omega_eff=oe, delta_E=dE,
                              nyquist=D // 2))
        print(f"\n=== {rep}: omega_max={om} omega_eff={oe}/{D//2} Delta_E={dE} ===", flush=True)
        for seed in SEEDS:
            d = load_hls_cmds(target_len=D, representation=rep, seed=seed)
            xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
            t0 = time.time()
            res = {}
            for name, fn in ENCODERS:
                res[name] = fn(xtr, ytr, xte, yte, seed)
            rbf, mlp = run_classical(xtr, ytr, xte, yte, seed)
            res["RBF"], res["MLP"] = rbf, mlp
            for model, bacc in res.items():
                rows.append(dict(representation=rep, seed=seed, model=model,
                                 test_bacc=round(float(bacc), 4)))
            pd.DataFrame(rows).to_csv(OUT_DIR / "table_hls_cmds.csv", index=False)
            print(f"  seed {seed}: " + "  ".join(f"{m}={res[m]:.3f}" for m in res)
                  + f"  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)
    summ = (df.groupby(["representation", "model"])["test_bacc"]
              .agg(["mean", "std"]).round(4).reset_index())
    summ.to_csv(OUT_DIR / "table_hls_cmds_summary.csv", index=False)
    diag.to_csv(OUT_DIR / "table_hls_cmds_diagnostics.csv", index=False)
    print("\n" + diag.to_string(index=False))
    print("\n" + summ.to_string(index=False))
    _plot(summ)


def _plot(summ):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["QSM", "amplitude", "angle (pooled)", "RBF", "MLP"]
    colours = {"QSM": "#7C3AED", "amplitude": "#a97cf5", "angle (pooled)": "#c9b3f5",
               "RBF": "tab:gray", "MLP": "#9aa0a6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, rep in zip(axes, REPRESENTATIONS):
        s = summ[summ.representation == rep].set_index("model").reindex(order)
        ax.bar(range(len(order)), s["mean"], yerr=s["std"],
               color=[colours[m] for m in order], capsize=3)
        ax.axhline(0.5, ls=":", color="gray", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{rep}")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Test balanced accuracy")
    fig.suptitle("HLS-CMDS heart-vs-lung: raw waveform (chance) vs spectral envelope (learnable)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_hls_cmds.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_hls_cmds.png'}")


if __name__ == "__main__":
    main()
