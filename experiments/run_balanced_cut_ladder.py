"""
run_balanced_cut_ladder.py
──────────────────────────
What governs the rank across a balanced bipartition, and is it the crossing
count?

The n-qubit generalisation of the coefficient-matrix rank is the rank across a
bipartition, and everything measured so far used three qubits, which admits only
``1|2`` cuts. A balanced cut needs four, and it is the first setting in which a
genuinely multipartite effect could appear at all: with two coordinates on each
side there is more than one way for the same number of entanglers to straddle
the cut. This experiment asks whether that freedom matters.

It does. Holding the crossing count fixed at two and moving only which edges
carry the gates changes the reachable rank from 12 to 16, so the count that the
three-qubit work fed into the ladder cannot be the governing quantity. What
governs it is the edge structure. Crossing edges that share no qubit act as
independent channels and their ranks multiply, while gates piled onto a single
edge climb the two-qubit ladder on that one channel and saturate at ``2L+1``.
Edges that share a qubit land between the two.

The experiment also re-checks cut-locality at a balanced cut, which the
three-qubit work established only for single-qubit cuts, and re-runs the
interleaving probe of P3 in the balanced setting.

Ranks are maximised over random parameter draws, and the number of draws needed
grows with the rank being resolved, so the high-rank cells are reported at
several draw counts rather than at one. Where a cell is still climbing, it is
reported as a lower bound rather than rounded up to the prediction.

Outputs:
    results/qnn-rankdesign/table_balanced_cut.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)

from code.rankdesign import (  # noqa: E402
    bipartition_rank_law,
    crossing_count,
    measure_cut_rank,
    reachable_rank_law,
)

OUT_DIR = ROOT / "results" / "qnn-rankdesign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_WIRES = 4
BALANCED = (0, 1)          # the {0,1} | {2,3} cut
SINGLE = (0,)              # the {0} | {1,2,3} cut, for continuity
SEED_LADDER = (3, 6, 12)   # resolving a high rank needs more draws


def probe(degree: int, entanglers, subset, seeds: int) -> int:
    return measure_cut_rank(
        N_WIRES, degree, sorted(entanglers), subset, seeds=seeds,
        n_grid=2 * degree + 3, rng=np.random.default_rng(7),
    )


def row(label: str, degree: int, entanglers, subset, family: str) -> dict:
    measured = {s: probe(degree, entanglers, subset, s) for s in SEED_LADDER}
    best = max(measured.values())
    law = bipartition_rank_law(entanglers, subset, degree, N_WIRES)
    count_law = reachable_rank_law(crossing_count(entanglers, subset), degree)
    saturated = measured[SEED_LADDER[-1]] == measured[SEED_LADDER[-2]]
    return dict(
        family=family, label=label, degree=degree,
        cut="|".join(("".join(map(str, subset)),
                      "".join(str(w) for w in range(N_WIRES) if w not in subset))),
        entanglers=str(sorted(entanglers)),
        n_crossing=crossing_count(entanglers, subset),
        count_law=count_law, factorised_law=law, measured=best,
        search_saturated=saturated,
        matches_factorised=(law is not None and best == law),
        count_law_undercounts=best > count_law,
        **{f"seeds_{s}": v for s, v in measured.items()},
    )


def main():
    rows, t0 = [], time.time()

    # 1. Cut-locality, now across a balanced cut.
    for degree in (4, 6):
        rows.append(row("no crossing entangler", degree,
                        [(1, (0, 1)), (2, (2, 3)), (3, (0, 1))],
                        BALANCED, "cut-locality"))

    # 2. The crossing count held fixed at two while the edges move.
    for label, ent in (
        ("both gates on (0,2)", [(2, (0, 2)), (4, (0, 2))]),
        ("(0,2) and (0,3), share qubit 0", [(2, (0, 2)), (4, (0, 3))]),
        ("(0,2) and (1,3), disjoint", [(2, (0, 2)), (4, (1, 3))]),
        ("(0,3) and (1,2), disjoint", [(2, (0, 3)), (4, (1, 2))]),
    ):
        rows.append(row(label, 6, ent, BALANCED, "same count, different edges"))

    # 3. The multiplicative law over disjoint edges.
    for degree, a_layers, b_layers in (
        (4, [1], []), (4, [1], [2]), (4, [1, 3], [2]), (4, [1, 3], [1, 3]),
        (6, [1], [3]), (6, [1, 5], [3]), (6, [1, 5], [1, 5]),
        (6, [1, 3, 5], [1, 5]),
    ):
        ent = ([(l, (0, 2)) for l in a_layers]
               + [(l, (1, 3)) for l in b_layers])
        rows.append(row(f"(0,2)x{a_layers} + (1,3)x{b_layers}", degree, ent,
                        BALANCED, "disjoint edges multiply"))

    # 4. Interleaving, the P3 effect, in the balanced setting.
    base = [(1, (0, 2)), (5, (0, 2))]
    for label, extra in (
        ("no interleaved gate", []),
        ("(0,1) before both, layer 0", [(0, (0, 1))]),
        ("(0,1) interleaved, layer 3", [(3, (0, 1))]),
        ("(2,3) interleaved, layer 3", [(3, (2, 3))]),
        ("both sides interleaved, layer 3", [(3, (0, 1)), (3, (2, 3))]),
    ):
        rows.append(row(label, 6, base + extra, BALANCED, "interleaving"))

    # 5. The single-qubit cut, where the two-qubit law should still hold.
    for kc in (1, 2, 3):
        layers = list(dict.fromkeys(
            int(round(v)) for v in np.linspace(1, 5, kc)))
        rows.append(row(f"{kc} crossing on (0,1)", 6,
                        [(l, (0, 1)) for l in layers], SINGLE,
                        "single-qubit cut"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_balanced_cut.csv", index=False)
    _report(df)
    print(f"\nSaved {OUT_DIR / 'table_balanced_cut.csv'}  ({time.time() - t0:.0f}s)")


def _report(df):
    for family, sub in df.groupby("family", sort=False):
        print(f"\n=== {family} ===")
        for _, r in sub.iterrows():
            law = "n/a" if pd.isna(r.factorised_law) else f"{int(r.factorised_law):>3}"
            trail = "" if r.search_saturated else "  (still climbing, lower bound)"
            print(f"  L={r.degree} {r.label:<34} cut {r.cut:<8} "
                  f"crossings {r.n_crossing}  count-law {r.count_law:>3}  "
                  f"factorised {law}  measured {r.measured:>3}"
                  f"{trail}")

    print("\nIs the crossing count sufficient to fix the rank?")
    fixed = df[df.family == "same count, different edges"]
    print(f"  at {fixed.n_crossing.iloc[0]} crossings and L={fixed.degree.iloc[0]}, "
          f"measured rank spans {fixed.measured.min()} to {fixed.measured.max()} "
          f"across edge placements, against a single count-law value of "
          f"{fixed.count_law.iloc[0]}")

    disjoint = df[df.factorised_law.notna() & (df.family == "disjoint edges multiply")]
    ok = disjoint.matches_factorised.sum()
    print(f"\nDoes the factorised law hold on disjoint edges? "
          f"{ok} of {len(disjoint)} exact")
    for _, r in disjoint[~disjoint.matches_factorised].iterrows():
        print(f"  short: L={r.degree} {r.label} predicted {int(r.factorised_law)}, "
              f"reached {r.measured} at {SEED_LADDER[-1]} draws "
              f"({'saturated' if r.search_saturated else 'still climbing'})")

    print(f"\nCut-locality holds (rank exactly 1 with no crossing gate): "
          f"{(df[df.family == 'cut-locality'].measured == 1).all()}")


if __name__ == "__main__":
    main()
