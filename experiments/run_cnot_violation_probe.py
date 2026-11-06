"""
run_cnot_violation_probe.py
───────────────────────────
Is CNOT's second monotonicity violation a nesting failure or a search failure?

The ladder's gate contrast rests on CNOT holding decreases that restarts cannot
close, because a fixed CNOT cannot switch off and its budget-``k`` class therefore
does not contain budget-``k-1``. Two decreases survive to 256 restarts. The rank-one
one is unarguable, since the target is exactly representable at ``k=0`` and the
class at ``k=1`` provably excludes it. The rank-two one is not obviously the same
animal: it shrank from 0.045 to 0.029 as the search grew, which is what a search
failure does and not what a class limit does.

Non-nesting does not by itself predict a *decrease*. It says the two classes are
different manifolds, so either could be better. The claim under test is the
stronger one, that the budget-two class genuinely tops out lower here.

Three things separate the hypotheses, and the first two disagree on this data,
which is why the third was run.

Per-draw consistency. A class limit binds on every target, a search failure lands
where the basins happen to be awkward. The rank-one control is 6 of 6 draws at
margins of 0.05 to 0.24. The rank-two cell is 4 of 6 with the aggregate carried by
two draws.

Extrapolation. Fitting each cell's restart distribution to its upper endpoint gave
0.714 for ``k=1`` against 0.693 for ``k=2`` with intervals that do not overlap,
which argues the gap is real. CRZ's matching pair overlaps.

Direct search. Quadrupling the restarts again, to 1024 on the four contested cells,
decides it. A search failure closes, a class limit does not. CRZ at the same rank
and budgets is the negative control, a pair already known to be tied, and CNOT at
rank one is the positive control, a gap already known to be structural.

Finding: it is both, split across targets. The pooled gap falls from -0.045 at 64
restarts to -0.017 at 1024, a 63 per cent reduction over a sixteenfold search,
while the rank-one control falls by 2 per cent over the same range and stays
negative on every draw. Per draw the rank-two cell separates cleanly. Three draws
sit at or above zero, one closes by 80 per cent to -0.020, and two hold at -0.069
and -0.066 without moving. The negative control collapses to within 0.015 of zero
on all six draws by 1024, so residuals of that size are not something this protocol
produces on a gate whose classes nest.

So the rank-one violation and the rank-two one are different objects. The first is
universal, because the ``k=0`` class contains the target exactly and the ``k=1``
CNOT class provably excludes it. The second is target-dependent, which is what
non-nesting actually predicts, since a different manifold may be worse on some
targets and equivalent on others. Pooling the two under a single count of
violations was the error, and reporting either as a mean over draws hides that four
of the six rank-two cells close completely.

Inputs (rank 2, k in {1,2}, merged on target_rank, target_seed, k, init):
    results/qnn-rankdesign/table_init_sensitivity_{final,ext256,deep1024}_{crz,cnot}.csv

Outputs:
    results/qnn-rankdesign/table_cnot_violation_probe.csv
    results/qnn-rankdesign/figure_cnot_violation_probe.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_restart_convergence import (  # noqa: E402
    expected_max,
    fit_limit,
)

OUT_DIR = ROOT / "results" / "qnn-rankdesign"

N_BOOT = 600
FIT_FROM = 8


def load(gate: str) -> pd.DataFrame:
    keys = ["target_rank", "target_seed", "k", "init"]
    parts = []
    for tag in (f"_final_{gate}", f"_ext256_{gate}", f"_deep1024_{gate}"):
        path = OUT_DIR / f"table_init_sensitivity{tag}.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(keys).any(), f"{gate}: duplicated fits after merge"
    return df


def gap_limit(lo_samples, hi_samples, ns, rng):
    """Extrapolated ``best(k+1) - best(k)`` with a bootstrap interval.

    Bootstrapping the *difference* rather than each cell separately keeps the
    comparison paired on the target draw, which is where most of the variance is.
    """
    obs = _gap(lo_samples, hi_samples, ns)
    boot = []
    for _ in range(N_BOOT):
        lo = [d[rng.integers(0, len(d), len(d))] for d in lo_samples]
        hi = [d[rng.integers(0, len(d), len(d))] for d in hi_samples]
        g = _gap(lo, hi, ns)
        if np.isfinite(g):
            boot.append(g)
    return obs, np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)


def _gap(lo_samples, hi_samples, ns):
    fit_ns = ns[ns >= FIT_FROM]
    curve_lo = np.array([np.mean([expected_max(d, n) for d in lo_samples]) for n in fit_ns])
    curve_hi = np.array([np.mean([expected_max(d, n) for d in hi_samples]) for n in fit_ns])
    a_lo, _ = fit_limit(fit_ns, curve_lo)
    a_hi, _ = fit_limit(fit_ns, curve_hi)
    return a_hi - a_lo


def probe(df, gate, rank, ka, kb, label):
    """Best-of-N gap between budgets ka and kb, per draw and pooled."""
    sub = df[(df.target_rank == rank) & (df.k.isin([ka, kb]))]
    n_max = int(sub.init.max()) + 1
    ns = np.array([n for n in (8, 16, 32, 64, 128, 256, 512, 1024) if n <= n_max])
    rng = np.random.default_rng(20260806)

    draws = sorted(sub.target_seed.unique())
    lo = [sub[(sub.target_seed == s) & (sub.k == ka)].sort_values("init").r2.to_numpy()
          for s in draws]
    hi = [sub[(sub.target_seed == s) & (sub.k == kb)].sort_values("init").r2.to_numpy()
          for s in draws]

    rows = []
    for s, a, b in zip(draws, lo, hi):
        per_n = {n: b[:n].max() - a[:n].max() for n in ns}
        rows.append(dict(
            gate=gate, label=label, target_rank=rank, ka=ka, kb=kb, target_seed=s,
            n_max=n_max, gap_at_64=per_n[64], gap_at_max=per_n[n_max],
            shrank=abs(per_n[n_max]) < abs(per_n[64]) - 1e-9,
            still_negative=per_n[n_max] < -1e-9,
        ))

    pooled_n = {n: float(np.mean([b[:n].max() for b in hi])
                         - np.mean([a[:n].max() for a in lo])) for n in ns}
    g_lim, (g_lo, g_hi) = gap_limit(lo, hi, ns, rng)
    pooled = dict(
        gate=gate, label=label, target_rank=rank, ka=ka, kb=kb, target_seed=-1,
        n_max=n_max, gap_at_64=pooled_n[64], gap_at_max=pooled_n[n_max],
        gap_limit=g_lim, gap_limit_lo=g_lo, gap_limit_hi=g_hi,
        draws_negative=int(sum(r["still_negative"] for r in rows)), n_draws=len(draws),
        shrank=abs(pooled_n[n_max]) < abs(pooled_n[64]) - 1e-9,
        still_negative=pooled_n[n_max] < -1e-9,
    )
    return pd.DataFrame(rows + [pooled]), pooled_n, ns


def main():
    data = {g: load(g) for g in ("crz", "cnot")}
    cases = [
        ("cnot", 1, 0, 1, "positive control: rank 1, structural"),
        ("cnot", 2, 1, 2, "under test: rank 2"),
        ("crz", 2, 1, 2, "negative control: rank 2, tied"),
    ]

    tables, curves = [], {}
    for gate, rank, ka, kb in [c[:4] for c in cases]:
        label = next(c[4] for c in cases if c[:4] == (gate, rank, ka, kb))
        t, curve, ns = probe(data[gate], gate, rank, ka, kb, label)
        tables.append(t)
        curves[label] = (curve, ns)

    out = pd.concat(tables, ignore_index=True)
    out.to_csv(OUT_DIR / "table_cnot_violation_probe.csv", index=False)
    _report(out, curves)
    _plot(curves)
    print(f"\nSaved {OUT_DIR / 'table_cnot_violation_probe.csv'}")


def _report(out, curves):
    print("=== Pooled gap best(k+1) - best(k) against restart count ===")
    print("(negative means the higher budget fits worse, which nesting would forbid)")
    for label, (curve, ns) in curves.items():
        print(f"\n  {label}")
        print("    " + "  ".join(f"N={n}: {curve[n]:+.4f}" for n in ns))

    print("\n=== Verdict per case ===")
    for _, r in out[out.target_seed == -1].iterrows():
        ci = f"[{r.gap_limit_lo:+.4f}, {r.gap_limit_hi:+.4f}]"
        structural = r.gap_limit_hi < 0
        print(f"\n  {r.label}")
        print(f"    gap at 64 {r.gap_at_64:+.4f} -> at {int(r.n_max)} {r.gap_at_max:+.4f}"
              f"   extrapolated {r.gap_limit:+.4f} {ci}")
        print(f"    still negative on {int(r.draws_negative)} of {int(r.n_draws)} draws")
        print(f"    -> {'gap survives extrapolation, class limit' if structural else 'gap is not resolved from zero, search failure'}")

    print("\n=== Per-draw detail for the case under test ===")
    sub = out[(out.label.str.startswith("under test")) & (out.target_seed >= 0)]
    for _, r in sub.iterrows():
        print(f"    draw {int(r.target_seed)}: {r.gap_at_64:+.4f} -> {r.gap_at_max:+.4f}"
              f"   {'closed' if not r.still_negative else 'still negative'}")

    # Read the point estimates with the control in hand. The negative control is a
    # tie by construction and still extrapolates negative, because the higher budget
    # is the harder search and its curve is the one further from its limit, so the
    # fit systematically under-extrapolates it. Only the interval is trustworthy,
    # and the direct measurement at the largest N is what should carry a verdict.
    ctrl = out[(out.label.str.startswith("negative")) & (out.target_seed == -1)].iloc[0]
    print(f"\n  Bias check: the tied control extrapolates to {ctrl.gap_limit:+.4f}, "
          f"not 0. Treat point estimates as biased low by roughly that much and "
          f"lean on the interval and on the direct value at N={int(ctrl.n_max)}.")


def _plot(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"positive control: rank 1, structural": ("o-", "#b2182b"),
              "under test: rank 2": ("s-", "#2166ac"),
              "negative control: rank 2, tied": ("^--", "#777777")}
    for label, (curve, ns) in curves.items():
        fmt, col = styles[label]
        ax.plot(ns, [curve[n] for n in ns], fmt, color=col, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("restarts")
    ax.set_ylabel(r"best $R^2$ at $k{+}1$ minus best at $k$")
    ax.set_title("A search failure closes with more restarts, a class limit does not")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_cnot_violation_probe.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_cnot_violation_probe.png'}")


if __name__ == "__main__":
    main()
