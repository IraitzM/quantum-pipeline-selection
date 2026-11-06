"""
run_gradient_variance_probe.py
──────────────────────────────
Locates the paper's operating point on the vanishing-expressivity axis
(Appendix companion to the Discussion's first-axis caveat).

Wang, Parampalli & Myers (arXiv:2607.22516) measure mixer-gradient
variance of coordinate-wise rotation encoders suppressed to
log10(Var) ~= -11 (Pendigits) and -20 (synthetic spectral tasks) at 16
qubits and depth 32, with near-identity fidelity Gram matrices — the
empirical signature of the coefficient concentration that Mhiri et al.
(2025) predict and that the Discussion flags as the risk to the
bandwidth-matching condition at larger qubit counts. Our own fits only
bracket that concentration from below (single-qubit regressors track the
classical truncation bound to 0.002 in R^2). This probe measures the
same quantity for OUR circuit family (angle encoding with a
StronglyEntanglingLayers mixer, the paper's axis-2 "entangled" variant)
across a width x depth grid spanning our operating point (8 qubits,
4 re-uploads), so the bracket becomes measured rather than argued.

For each (n_qubits, n_reuploads) cell we draw S independent
(initialisation, random-input) pairs, take d<Z_0>/d(theta) for every
mixer parameter, and report log10 of the pooled gradient variance.
Random inputs are standard-normal pooled features, matching the
z-scored signals used throughout.

Outputs:
    results/qnn-qsm/table_gradient_probe.csv

Usage:
    uv run experiments/run_gradient_variance_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # local package shadows stdlib ``code``

from code.qnn import AngleEncodingClassifier  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-qsm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "table_gradient_probe.csv"

N_QUBITS_GRID = [2, 4, 8]
N_REUPLOADS_GRID = [4, 16, 32]
S_SAMPLES = 20

# External reference points from Wang et al. (16 qubits, depth 32).
WANG_LOG10_VAR = {"Pendigits": -11.0, "synthetic": -20.0}


def cell_log10_var(n_qubits: int, n_reuploads: int, s_samples: int) -> float:
    grads = []
    for s in range(s_samples):
        torch.manual_seed(10_000 + s)
        model = AngleEncodingClassifier(
            seq_len=n_qubits,  # features are pre-pooled; pass n_qubits directly
            n_qubits=n_qubits,
            n_reuploads=n_reuploads,
            n_classes=2,
            use_entangling_ansatz=True,
        )
        x = torch.randn(n_qubits, dtype=torch.float64)
        out = model._circuit(x, model.weights)[0]  # <Z_0>
        model.zero_grad()
        out.backward()
        grads.append(model.weights.grad.detach().flatten())
    pooled = torch.cat(grads)
    return float(torch.log10(pooled.var()).item())


def main():
    rows = []
    for n_qubits in N_QUBITS_GRID:
        for n_reuploads in N_REUPLOADS_GRID:
            lv = cell_log10_var(n_qubits, n_reuploads, S_SAMPLES)
            rows.append(dict(n_qubits=n_qubits, n_reuploads=n_reuploads,
                             log10_grad_var=round(lv, 3), n_samples=S_SAMPLES))
            print(f"n={n_qubits} R={n_reuploads}: log10 Var = {lv:.2f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    print(df.pivot(index="n_qubits", columns="n_reuploads",
                   values="log10_grad_var").to_string())
    print(f"\nReference (Wang et al., 16 qubits, depth 32): {WANG_LOG10_VAR}")


if __name__ == "__main__":
    main()
