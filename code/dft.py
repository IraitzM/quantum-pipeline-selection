"""One-sided DFT helpers used by the single-qubit calibration experiment.

These are verbatim copies of ``_dft_magnitudes_and_coeffs``, ``_top_frequencies``
and ``_fourier_r2`` from the working repository's ``code/main.py``. They are
extracted here so the companion repository does not carry the unrelated
fast-experiment runner that ``code/main.py`` belongs to. The only consumer is
``experiments/run_signal_regression.py``.
"""
from __future__ import annotations

import numpy as np


def _dft_magnitudes_and_coeffs(x: np.ndarray):
    """One-sided rfft: returns (magnitudes, complex_coeffs)."""
    coeffs = np.fft.rfft(x)  # complex, length N//2+1
    mags = np.abs(coeffs) / len(x)
    mags[1:-1] *= 2  # two-sided energy for non-DC/Nyquist
    return mags, coeffs


def _top_frequencies(
    magnitudes: np.ndarray, energy_threshold: float = 0.95, max_k: int = 16
) -> np.ndarray:
    """Indices of the top-K frequencies carrying ≥ energy_threshold of power."""
    power = magnitudes**2
    total = power.sum() + 1e-12
    order = np.argsort(power)[::-1]
    cumulative, selected = 0.0, []
    for idx in order:
        if idx == 0:  # skip DC
            continue
        selected.append(idx)
        cumulative += power[idx] / total
        if cumulative >= energy_threshold or len(selected) >= max_k:
            break
    return np.array(selected, dtype=int)


def _fourier_r2(x: np.ndarray, top_freqs: np.ndarray) -> float:
    """R² of reconstructing x keeping only the top Fourier components (with phase)."""
    N = len(x)
    _, coeffs = _dft_magnitudes_and_coeffs(x)
    # zero out all frequencies not in top_freqs
    mask = np.zeros(len(coeffs), dtype=complex)
    mask[0] = coeffs[0]  # always keep DC
    for k in top_freqs:
        if k < len(coeffs):
            mask[k] = coeffs[k]
    recon = np.fft.irfft(mask, n=N)
    ss_tot = np.var(x) * N + 1e-12
    ss_res = np.sum((x - recon) ** 2)
    return float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))
