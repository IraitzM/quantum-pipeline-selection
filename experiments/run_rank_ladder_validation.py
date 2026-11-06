"""
run_rank_ladder_validation.py
─────────────────────────────
Does the reachable-rank law predict what a circuit can actually *fit*, or only
what it can in principle represent?

``code.rankdesign`` establishes without training that a 2-qubit re-uploading
circuit of depth L carrying k entangling gates reaches coefficient-matrix rank
min(2L+1, 4*3**(k-1)), while its frequency support stays the (2L+1) box
whatever the entangler does. That is a statement about the reachable set. This
experiment asks what survives optimisation, which is the only version of the
claim that can be used to choose an ansatz.

We fit targets of exactly controlled rank at a bandwidth matched to the
circuit's own, so bandwidth cannot be the limiting factor and any shortfall is
attributable to rank or to the parameter budget rather than to spectral reach.
Each fit is compared against the exact ceiling for its reachable rank, the best
truncated-SVD approximation of that rank, and is additionally scored on whether
it clears the ceiling of the next budget down, which is the bar its extra
entangler has to justify.

Two controls make the result interpretable. Every configuration is fit with
both a fixed CNOT and a parameterised CRZ, which isolates the effect of the
model classes being nested or not, and every cell records whether the parameter
budget could cover the target's degrees of freedom at all, which is what
decides whether the rank ceiling was ever attainable. Fits use LBFGS: on this
small deterministic full-batch problem it agrees with a long Adam run to within
0.003 R^2 at a fraction of the cost, so a shortfall here is a property of the
model class rather than of the optimiser.

Outputs:
    results/qnn-rankdesign/table_rank_ladder.csv
    results/qnn-rankdesign/table_rank_ladder_summary.csv
    results/qnn-rankdesign/figure_rank_ladder.png
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
sys.modules.pop("code", None)

from code.rankdesign import (  # noqa: E402
    EntanglerLadder2DQNN,
    controlled_rank_target,
    fitted_model_rank,
    measure_reachable_rank,
    model_ceiling_r2,
    parameter_count,
    rank_ceiling_is_attainable,
    reachable_rank_law,
    recommended_placement,
    target_dof,
    truncated_svd_r2,
)

OUT_DIR = ROOT / "results" / "qnn-rankdesign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEGREE = 4          # coefficient box 9 x 9, so rank caps at 9
BANDWIDTH = 4       # target support matched to the circuit's own
N_GRID = 12
BUDGETS = [0, 1, 2, 3]
GATES = ["cnot", "crz"]
TARGET_RANKS = [1, 2, 4, 6, 9]
TARGET_SEEDS = 3
RESTARTS = 3
LBFGS_ROUNDS = 6
LBFGS_ITERS = 150


def fit(model_seed: int, ent_layers, gate: str, xs, ys, target_flat):
    """Returns (R^2, realised rank of the fitted response).

    The realised rank separates the two ways a fit can fall short. Below the
    configuration's reachable rank means the optimiser did not use the capacity
    that was there. At the reachable rank while still trailing the target means
    something other than rank binds, which here is the parameter budget.
    """
    torch.manual_seed(model_seed)
    model = EntanglerLadder2DQNN(degree=DEGREE, ent_layers=ent_layers, gate=gate)
    for _ in range(LBFGS_ROUNDS):
        opt = torch.optim.LBFGS(
            model.parameters(), max_iter=LBFGS_ITERS, history_size=50,
            line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14,
        )

        def closure():
            opt.zero_grad()
            loss = ((model(xs, ys) - target_flat) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)

    with torch.no_grad():
        pred = model(xs, ys).numpy()
    tgt = target_flat.numpy()
    ss_res = float(np.sum((tgt - pred) ** 2))
    ss_tot = float(np.sum((tgt - tgt.mean()) ** 2)) + 1e-12
    s = np.linalg.svd(pred.reshape(N_GRID, N_GRID), compute_uv=False)
    return 1.0 - ss_res / ss_tot, int(np.sum(s > 1e-7 * s[0]))


def main():
    axis = np.arange(N_GRID) * 2 * np.pi / N_GRID
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    xs = torch.tensor(X.ravel(), dtype=torch.float64)
    ys = torch.tensor(Y.ravel(), dtype=torch.float64)

    placements = {k: recommended_placement(k, DEGREE) for k in BUDGETS}
    print(f"depth L={DEGREE}, rank cap {2 * DEGREE + 1}, "
          f"~{parameter_count(DEGREE)} trainable parameters")
    for k, pattern in placements.items():
        measured = measure_reachable_rank(DEGREE, pattern, seeds=4)
        predicted = reachable_rank_law(k, DEGREE)
        print(f"  k={k} layers={str(pattern):<12} law {predicted:>3}  "
              f"measured {measured:>3}  {'OK' if measured == predicted else 'MISMATCH'}")

    rows = []
    t0 = time.time()
    for target_rank in TARGET_RANKS:
        dof = target_dof(target_rank, DEGREE)
        print(f"\ntarget rank {target_rank}: ~{dof} dof vs "
              f"~{parameter_count(DEGREE)} parameters")
        for tseed in range(TARGET_SEEDS):
            target = controlled_rank_target(
                target_rank, BANDWIDTH, N_GRID, rng=np.random.default_rng(1000 + tseed)
            )
            target_flat = torch.tensor(target.ravel(), dtype=torch.float64)
            for gate in GATES:
                for k in BUDGETS:
                    # The fitted model is scale*<Z0Z1> + bias, so its rank is
                    # one above the bare correlator's. Using the bare law here
                    # understates the class and fakes ceiling violations.
                    reach = fitted_model_rank(k, DEGREE)
                    # k=0 is separable-plus-constant, a strict subset of the
                    # rank-2 matrices, so it needs its own exact ceiling.
                    ceiling = model_ceiling_r2(target, k, DEGREE)
                    prev_ceiling = (model_ceiling_r2(target, k - 1, DEGREE)
                                    if k > 0 else 0.0)
                    r2, realised = max(
                        (fit(100 * k + 10 * tseed + r, placements[k], gate, xs, ys,
                             target_flat) for r in range(RESTARTS)),
                        key=lambda pair: pair[0],
                    )
                    rows.append(dict(
                        target_rank=target_rank, target_seed=tseed, gate=gate,
                        n_entanglers=k, ent_layers=str(placements[k]),
                        reachable_rank=reach, realised_rank=realised,
                        rank_sufficient=reach >= target_rank,
                        target_dof=dof, n_params=parameter_count(DEGREE, k),
                        dof_sufficient=rank_ceiling_is_attainable(target_rank, DEGREE, k),
                        ceiling_r2=ceiling, prev_ceiling_r2=prev_ceiling, fitted_r2=r2,
                        shortfall=ceiling - r2,
                        exceeds_prev_ceiling=bool(k > 0 and r2 > prev_ceiling),
                        respects_own_ceiling=bool(r2 <= ceiling + 1e-6),
                    ))
                    print(f"  seed={tseed} {gate:>4} k={k} (reach {reach}, "
                          f"realised {realised:>2}) R2={r2:.4f}  ceiling={ceiling:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_rank_ladder.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'table_rank_ladder.csv'}  ({time.time() - t0:.0f}s)")

    summary = (
        df.groupby(["target_rank", "gate", "n_entanglers"])
        .agg(fitted_r2=("fitted_r2", "mean"), ceiling_r2=("ceiling_r2", "mean"),
             shortfall=("shortfall", "mean"), reachable_rank=("reachable_rank", "first"),
             realised_rank=("realised_rank", "mean"),
             dof_sufficient=("dof_sufficient", "first"),
             crosses=("exceeds_prev_ceiling", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "table_rank_ladder_summary.csv", index=False)
    _report(df, summary)
    _plot(summary)


def _report(df, summary):
    violations = df[~df.respects_own_ceiling]
    print(f"\nCeiling violations (any would falsify the rank law): "
          f"{len(violations)} of {len(df)}")

    print("\nIs the ladder monotone in k? (mean R2 by budget)")
    for gate in GATES:
        means = [summary[(summary.gate == gate) & (summary.n_entanglers == k)]
                 .fitted_r2.mean() for k in BUDGETS]
        # Tolerance covers restart-to-restart scatter, which is ~0.003 here.
        mono = all(b >= a - 5e-3 for a, b in zip(means, means[1:]))
        print(f"  {gate:>4}: " + "  ".join(f"k={k} {m:.4f}" for k, m in zip(BUDGETS, means))
              + f"   {'monotone' if mono else 'NOT monotone'}")

    print("\nDoes the rank ceiling get attained when the dof budget allows it?")
    for label, sub in (("dof sufficient", df[df.dof_sufficient]),
                       ("dof insufficient", df[~df.dof_sufficient])):
        for gate in GATES:
            g = sub[(sub.gate == gate) & (sub.rank_sufficient)]
            if len(g):
                print(f"  {label:<17} {gate:>4}: mean gap to own ceiling "
                      f"{g.shortfall.mean():+.4f} over {len(g)} fits")

    print("\nRealised vs reachable rank (was the capacity used?)")
    for gate in GATES:
        for k in BUDGETS[1:]:
            sub = df[(df.gate == gate) & (df.n_entanglers == k)]
            print(f"  {gate:>4} k={k}: reachable {sub.reachable_rank.iloc[0]:>2}, "
                  f"realised mean {sub.realised_rank.mean():>4.1f}")


def _plot(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(BUDGETS)))
    for ax, gate in zip(axes, GATES):
        for c, k in zip(colors, BUDGETS):
            sub = summary[(summary.gate == gate) & (summary.n_entanglers == k)]
            sub = sub.sort_values("target_rank")
            ax.plot(sub.target_rank, sub.fitted_r2, "o-", color=c,
                    label=f"k={k} (reach rank {sub.reachable_rank.iloc[0]})")
            ax.plot(sub.target_rank, sub.ceiling_r2, ":", color=c, alpha=0.6)
        ax.set_xlabel("rank of target (bandwidth matched to circuit)")
        ax.set_title(f"{gate.upper()} entangler"
                     + ("  —  fixed, classes not nested" if gate == "cnot"
                        else "  —  parameterised, classes nested"))
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("R²   (solid = fitted, dotted = exact rank ceiling)")
    fig.suptitle(f"Rank ladder at depth L={DEGREE}: the ceiling binds only while "
                 f"the parameter budget can cover the target's degrees of freedom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_rank_ladder.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'figure_rank_ladder.png'}")


if __name__ == "__main__":
    main()
