"""
run_signal_regression.py
─────────────────────────
Direct functional-approximation demo: fit representative MNIST1D and ECG
signal windows with (a) a single-qubit data-re-uploading QNN of Fourier
degree L and (b) a classical truncated Fourier-series least-squares fit of
the same degree, sweeping L. Also computes the classical FFT-based
omega_max/R^2 diagnostic directly on each raw signal, to check whether that
a-priori estimate predicts the re-upload depth the QNN actually needs.

Outputs:
    results/qnn-signals/regression_sweep.csv
    results/qnn-signals/regression_curves.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.baselines import classical_fourier_regression  # noqa: E402
from code.datasets import load_ecg_beats, load_mnist1d  # noqa: E402
from code.dft import _dft_magnitudes_and_coeffs, _fourier_r2, _top_frequencies  # noqa: E402
from code.qnn import FourierSeriesQNN  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEGREES = list(range(1, 9))
STEPS = 400
LR = 0.05
SEED = 42


def _effective_bandwidth(mags: np.ndarray, energy_threshold: float = 0.95) -> int:
    """Smallest prefix band |omega| <= W holding ``energy_threshold`` of the energy.

    This is ``omega_eff`` of the paper: unlike ``omega_max``, which records where
    the last individually-selected coefficient sits, it is the narrowest
    contiguous band from DC that already clears the threshold. A degree-D circuit
    accesses every frequency |omega| <= D, so this is the quantity the design rule
    D >~ omega_eff is stated in. The two coincide iff the spectrum decays
    monotonically, and always satisfy omega_eff <= omega_max.
    """
    power = mags**2
    total = power.sum() + 1e-12
    cumulative = 0.0
    for w in range(len(power)):
        cumulative += power[w] / total
        if cumulative >= energy_threshold:
            return w
    return len(power) - 1


def spectral_signal_diagnostics(y: np.ndarray) -> dict:
    """Apply a classical FFT-based spectral diagnostic directly to a raw signal (top-K dominant frequencies at a 95% energy threshold, and the R² of reconstructing the signal from just those frequencies)."""
    mags, _ = _dft_magnitudes_and_coeffs(y)
    top = _top_frequencies(mags)
    if len(top) == 0:
        top = np.array([1])
    r2 = _fourier_r2(y, top)
    return {
        "omega_max": int(top.max()),
        "omega_eff": _effective_bandwidth(mags),
        "K": int(len(top)),
        "fourier_r2": float(r2),
    }


def fit_qnn(t: np.ndarray, y: np.ndarray, degree: int) -> dict:
    torch.manual_seed(SEED)
    model = FourierSeriesQNN(degree=degree)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t_t = torch.tensor(t, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    t0 = time.time()
    for _ in range(STEPS):
        opt.zero_grad()
        pred = model(t_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        opt.step()
    dt = time.time() - t0

    with torch.no_grad():
        pred = model(t_t).numpy()
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    n_params = (degree + 1) * 3 + 2
    return {"r2": r2, "n_params": n_params, "train_time_s": dt, "y_hat": pred}


def get_signals() -> dict[str, np.ndarray]:
    m = load_mnist1d()
    e = load_ecg_beats()
    # representative raw examples (not class means) -- honest, noisy real windows
    idx0 = np.where(m["y_train"] == 0)[0][0]
    idx7 = np.where(m["y_train"] == 7)[0][0]
    idxN = np.where(e["y_train"] == 0)[0][0]
    idxV = np.where(e["y_train"] == 1)[0][0]
    return {
        "mnist1d_digit0": m["x_train"][idx0],
        "mnist1d_digit7": m["x_train"][idx7],
        "ecg_normal": e["x_train"][idxN],
        "ecg_pvc": e["x_train"][idxV],
    }


def main():
    signals = get_signals()
    t = np.linspace(-np.pi, np.pi, 40, endpoint=False)

    rows = []
    curves = {}
    for name, y in signals.items():
        y = (y - y.mean()) / (y.std() + 1e-8)
        diag = spectral_signal_diagnostics(y)
        print(f"[{name}] spectral diagnostic: omega_max={diag['omega_max']} "
              f"omega_eff={diag['omega_eff']} "
              f"K={diag['K']} fourier_r2={diag['fourier_r2']:.3f}")

        curves[name] = {"t": t, "y": y}
        for degree in DEGREES:
            classical = classical_fourier_regression(t, y, degree)
            qnn = fit_qnn(t, y, degree)
            rows.append(
                dict(
                    signal=name,
                    degree=degree,
                    omega_max=diag["omega_max"],
                    omega_eff=diag["omega_eff"],
                    fft_fourier_r2=diag["fourier_r2"],
                    classical_r2=classical["r2"],
                    classical_n_params=classical["n_params"],
                    qnn_r2=qnn["r2"],
                    qnn_n_params=qnn["n_params"],
                    qnn_train_time_s=round(qnn["train_time_s"], 2),
                )
            )
            print(
                f"  degree={degree:<2} classical_R2={classical['r2']:.3f} "
                f"qnn_R2={qnn['r2']:.3f} ({qnn['train_time_s']:.1f}s)"
            )
            if degree == diag["omega_max"] or degree == DEGREES[-1]:
                curves[name][f"classical_deg{degree}"] = classical["y_hat"]
                curves[name][f"qnn_deg{degree}"] = qnn["y_hat"]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regression_sweep.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'regression_sweep.csv'}")

    _plot(signals.keys(), df, curves)


def _plot(names, df: pd.DataFrame, curves: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 7))
    for j, name in enumerate(names):
        sub = df[df.signal == name]
        ax = axes[0, j]
        ax.plot(sub.degree, sub.classical_r2, "o-", label="classical Fourier LS")
        ax.plot(sub.degree, sub.qnn_r2, "s--", label="single-qubit QNN")
        omega = sub.omega_max.iloc[0]
        omega_e = sub.omega_eff.iloc[0]
        ax.axvline(omega, color="gray", linestyle=":", label=f"ω_max={omega}")
        # omega_eff is the prefix band the design rule D >~ omega_eff is stated
        # in; it lands on or below omega_max and tracks the measured knee more
        # closely, so mark both rather than only the conservative bound.
        if omega_e != omega:
            ax.axvline(omega_e, color="tab:green", linestyle="-.",
                       label=f"ω_eff={omega_e}")
        ax.set_title(name)
        ax.set_xlabel("Fourier degree / re-upload depth")
        ax.set_ylabel("R²")
        ax.set_ylim(-0.05, 1.05)
        if j == 0:
            ax.legend(fontsize=8)

        ax2 = axes[1, j]
        c = curves[name]
        ax2.plot(c["t"], c["y"], "k-", label="signal", linewidth=1.5)
        best_deg = sub.degree.iloc[-1]
        key = f"qnn_deg{best_deg}"
        if key in c:
            ax2.plot(c["t"], c[key], "s--", label=f"QNN fit (deg={best_deg})", markersize=3)
        ax2.set_xlabel("t")
        if j == 0:
            ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "regression_curves.png", dpi=150)
    print(f"Saved {OUT_DIR / 'regression_curves.png'}")


if __name__ == "__main__":
    main()
