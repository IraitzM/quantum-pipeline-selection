"""
run_restart_convergence.py
──────────────────────────
Are the ladder's fitted values converged, and if not, what do they converge to?

Both papers quote the ladder from the best of 64 restarts and say in the captions
that the value is a lower bound, because best-of-N was still climbing when the run
stopped. That is honest but unsatisfying, and a reviewer is entitled to ask what
the number would be with the search run to exhaustion. This script answers it from
a 256-restart extension of ``run_entangler_init_sensitivity.py``, without needing
the search to actually exhaust.

Three things make that possible. Best-of-N is an extreme-value quantity, so its
expectation over the empirical distribution of a cell's restarts has a closed form
in the order statistics and does not have to be simulated. If the R^2 distribution
of a cell has a finite upper endpoint, which it must here since R^2 is bounded by
the model class ceiling, the extreme-value index is negative and the approach to
that endpoint is a power law, so fitting ``a - b N^(-c)`` estimates the converged
value with ``a`` and the curvature ``c`` says how fast. And resampling the
restarts of each cell puts an interval on ``a``, which is what turns an
extrapolation into a quotable number.

The monotonicity control is re-run at every N up to 256 for both gates. That claim
is a count of violations rather than a margin, so it neither needs nor benefits
from convergence, but quadrupling N is the strongest available test that CRZ's
zero and CNOT's two are structural.

Inputs (merged on target_rank, target_seed, k, init):
    results/qnn-rankdesign/table_init_sensitivity_final_{crz,cnot}.csv
    results/qnn-rankdesign/table_init_sensitivity_ext256_{crz,cnot}.csv

Outputs:
    results/qnn-rankdesign/table_restart_convergence.csv
    results/qnn-rankdesign/table_restart_monotonicity.csv
    results/qnn-rankdesign/figure_restart_convergence.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "qnn-rankdesign"

GATES = ["crz", "cnot"]
BUDGETS = [0, 1, 2, 3]
# Fit the tail only. The first few restarts are dominated by how bad the worst
# basins are, which is not what the extrapolation is about.
FIT_FROM = 8
N_BOOT = 400


def load(gate: str) -> pd.DataFrame:
    keys = ["target_rank", "target_seed", "k", "init"]
    parts = []
    for tag in (f"_final_{gate}", f"_ext256_{gate}"):
        path = OUT_DIR / f"table_init_sensitivity{tag}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}; run the extension first")
        parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(keys).sum()
    if dup:
        raise SystemExit(f"{gate}: {dup} duplicated (rank, seed, k, init) rows")
    return df.sort_values(keys).reset_index(drop=True)


def expected_max(samples: np.ndarray, n: int) -> float:
    """E[max of ``n`` draws] under the empirical distribution of ``samples``.

    Exact rather than simulated. With the sample sorted ascending, the i-th value
    is the maximum of n draws exactly when all n land at or below it and not all
    land strictly below, which has probability (i/m)^n - ((i-1)/m)^n.
    """
    x = np.sort(samples)
    m = len(x)
    idx = np.arange(1, m + 1)
    w = (idx / m) ** n - ((idx - 1) / m) ** n
    return float(x @ w)


def _curve(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_limit(ns: np.ndarray, vals: np.ndarray):
    """Extrapolate best-of-N to its limit. Returns (limit, exponent) or (nan, nan).

    A negative extreme-value index makes the approach to the upper endpoint a
    power law, so this is the shape the theory asks for rather than a convenient
    curve. A flat series defeats the fit and is reported as already converged.
    """
    spread = vals.max() - vals.min()
    if spread < 1e-9:
        return float(vals[-1]), np.inf
    try:
        p, _ = curve_fit(
            _curve, ns, vals,
            p0=[vals[-1] + spread, spread, 0.5],
            bounds=([vals[-1], 0.0, 0.01], [1.0001, 10.0, 10.0]),
            maxfev=20000,
        )
    except Exception:
        return float("nan"), float("nan")
    return float(p[0]), float(p[2])


def convergence_table(df: pd.DataFrame, gate: str, n_max: int) -> pd.DataFrame:
    """Per (rank, k): observed best of 64 and of n_max, plus the extrapolated limit.

    The cell-level quantity is averaged over target draws in the same order the
    papers use, maximum within a draw and mean across draws, so the numbers here
    are directly substitutable into tab:rank-ladder.
    """
    ns = np.array([n for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
                   if n <= n_max])
    fit_ns = ns[ns >= FIT_FROM]
    rows = []
    rng = np.random.default_rng(20260806)

    for (rank, k), cell in df.groupby(["target_rank", "k"]):
        draws = [g.r2.to_numpy() for _, g in cell.groupby("target_seed")]

        curve = np.array([np.mean([expected_max(d, n) for d in draws]) for n in ns])
        limit, expo = fit_limit(fit_ns, curve[ns >= FIT_FROM])

        boot = []
        for _ in range(N_BOOT):
            res = [d[rng.integers(0, len(d), len(d))] for d in draws]
            c = np.array([np.mean([expected_max(d, n) for d in res]) for n in fit_ns])
            lim, _e = fit_limit(fit_ns, c)
            if np.isfinite(lim):
                boot.append(lim)
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > N_BOOT // 2
                  else (np.nan, np.nan))

        obs = {n: np.mean([d[:n].max() for d in draws]) for n in (64, n_max)}
        rows.append(dict(
            gate=gate, target_rank=rank, k=k,
            observed_best64=obs[64], observed_bestmax=obs[n_max], n_max=n_max,
            expected_best64=curve[ns == 64][0], expected_bestmax=curve[-1],
            gain_64_to_max=obs[n_max] - obs[64],
            slope_per8_at64=curve[ns == 64][0] - curve[ns == 48][0],
            slope_per8_at_max=(curve[-1] - curve[-2]) / (ns[-1] - ns[-2]) * 8,
            limit=limit, limit_lo=lo, limit_hi=hi, exponent=expo,
            remaining=limit - obs[n_max],
            ceiling=cell.ceiling.iloc[0], n_inits=len(draws[0]),
        ))
    return pd.DataFrame(rows), ns


def monotonicity_table(df: pd.DataFrame, gate: str, ns) -> pd.DataFrame:
    """Violations of the nested-class prediction as a function of restart count."""
    ranks = sorted(df.target_rank.unique())
    rows = []
    for n in ns:
        best = (df[df.init < n].groupby(["target_rank", "target_seed", "k"]).r2.max()
                .groupby(["target_rank", "k"]).mean())
        viol = []
        for r in ranks:
            v = [best.loc[(r, k)] for k in BUDGETS]
            viol += [f"r{r}:k{k}->{k + 1}"
                     for k, (a, b) in enumerate(zip(v, v[1:])) if b < a - 1e-9]
        rows.append(dict(gate=gate, n_inits=n, violations=len(viol),
                         of=len(ranks) * (len(BUDGETS) - 1), which=";".join(viol)))
    return pd.DataFrame(rows)


def main():
    conv, mono = [], []
    for gate in GATES:
        df = load(gate)
        n_max = int(df.init.max()) + 1
        print(f"{gate}: {len(df)} fits, inits 0..{n_max - 1}, "
              f"{df.target_seed.nunique()} target draws")
        c, ns = convergence_table(df, gate, n_max)
        conv.append(c)
        mono.append(monotonicity_table(df, gate, ns))

    conv = pd.concat(conv, ignore_index=True)
    mono = pd.concat(mono, ignore_index=True)
    conv.to_csv(OUT_DIR / "table_restart_convergence.csv", index=False)
    mono.to_csv(OUT_DIR / "table_restart_monotonicity.csv", index=False)
    _report(conv, mono)
    _plot(conv)


def _report(conv, mono):
    for gate in GATES:
        c = conv[conv.gate == gate]
        n_max = int(c.n_max.iloc[0])
        print(f"\n=== {gate.upper()}: best of 64 -> best of {n_max} -> extrapolated limit ===")
        print(f"  {'rank':>4} {'k':>2}  {'best64':>7} {'best' + str(n_max):>8} "
              f"{'limit':>7} {'95% CI':>17} {'left':>7} {'per8@' + str(n_max):>9}")
        for _, r in c.sort_values(["target_rank", "k"]).iterrows():
            ci = (f"[{r.limit_lo:.3f}, {r.limit_hi:.3f}]"
                  if np.isfinite(r.limit_lo) else "—")
            print(f"  {int(r.target_rank):>4} {int(r.k):>2}  {r.observed_best64:>7.4f} "
                  f"{r.observed_bestmax:>8.4f} {r.limit:>7.4f} {ci:>17} "
                  f"{r.remaining:>7.4f} {r.slope_per8_at_max:>9.5f}")

        print(f"\n  Ladder as it would be quoted at {n_max} restarts:")
        for rank, g in c.groupby("target_rank"):
            g = g.sort_values("k")
            print(f"    rank {int(rank)}: " + "  ".join(
                f"{v:.3f}" for v in g.observed_bestmax))

        print("\n  Do the rungs survive extrapolation? (limit, k=1 -> k=3)")
        for rank, g in c.groupby("target_rank"):
            g = g.sort_values("k").set_index("k")
            print(f"    rank {int(rank)}: observed +{g.observed_bestmax[3] - g.observed_bestmax[1]:.3f}"
                  f"   extrapolated +{g.limit[3] - g.limit[1]:.3f}")

    print("\n=== Monotonicity violations against restart count ===")
    piv = mono.pivot(index="n_inits", columns="gate", values="violations")
    print(piv.to_string())
    for gate in GATES:
        last = mono[(mono.gate == gate)].iloc[-1]
        print(f"  {gate} at N={last.n_inits}: {last.violations} of {last.of}"
              + (f"  ({last.which})" if last.violations else ""))


def _plot(conv):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sorted(conv.target_rank.unique())
    fig, axes = plt.subplots(1, len(GATES), figsize=(6.5 * len(GATES), 5), sharey=True)
    for ax, gate in zip(np.atleast_1d(axes), GATES):
        c = conv[conv.gate == gate]
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(BUDGETS)))
        width = 0.2
        for i, (col, k) in enumerate(zip(colors, BUDGETS)):
            g = c[c.k == k].sort_values("target_rank")
            x = np.arange(len(ranks)) + (i - 1.5) * width
            ax.bar(x, g.observed_bestmax, width, color=col, label=f"k={k}")
            ax.errorbar(x, g.limit, yerr=[g.limit - g.limit_lo, g.limit_hi - g.limit],
                        fmt="_", color="k", ms=8, lw=1, capsize=2)
        ax.set_xticks(np.arange(len(ranks)))
        ax.set_xticklabels([str(int(r)) for r in ranks])
        ax.set_xlabel("target rank")
        ax.set_title(f"{gate.upper()}")
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("best $R^2$ (bar = observed, tick = extrapolated limit)")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("The restart search is not converged, and the ordering does not depend on it")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_restart_convergence.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'figure_restart_convergence.png'}")


if __name__ == "__main__":
    main()
