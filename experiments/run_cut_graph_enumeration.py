"""
run_cut_graph_enumeration.py
────────────────────────────
Which graph invariant, if any, governs the rank across a balanced cut?

Two candidate laws died in `run_cut_rank_graph_law.py`. The rank of the graph's
cut block, borrowed from the finite-field rank-purity duality, predicts the
channel count but not the value. A connected-component reading fixes the
shared-qubit case and then fails on a path, which measures 47 where the channel
picture allows at most 16. Rather than guess a third time, this enumerates every
crossing-edge multiset up to three gates on a ``2|2`` cut and tabulates the
measured rank against every invariant to hand, so that whichever invariant does
separate the cells can be read off rather than proposed.

Invariants recorded per configuration: gate count, distinct edge count, rank of
the 0/1 cut block, number of connected components of the crossing-edge graph,
maximum matching size, and the sorted degree sequence. The read-out reports, for
each invariant, whether configurations agreeing on it agree on the measured rank.
An invariant that fails that test cannot be the governing quantity on its own.

Placement is not the object of study here but does affect the value, so each
configuration is maximised over the orderings of its edges on a fixed spread of
layers, and ranks are maximised over parameter draws as everywhere else.

Outputs:
    results/qnn-rankdesign/table_cut_graph_enumeration.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)

from code.rankdesign import graph_cut_rank, measure_cut_rank  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-rankdesign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_WIRES = 4
CUT = (0, 1)
CROSSING_EDGES = [(0, 2), (0, 3), (1, 2), (1, 3)]


def components(edges) -> int:
    """Connected components of the crossing-edge graph, counting only touched wires."""
    parent = {w: w for e in edges for w in e}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(w) for w in parent})


def matching_number(edges) -> int:
    """Largest set of pairwise disjoint edges. The graph is inside K_{2,2}, so 2 caps it."""
    best = 0
    uniq = sorted(set(edges))
    for size in (2, 1):
        for combo in itertools.combinations(uniq, size):
            touched = [w for e in combo for w in e]
            if len(set(touched)) == len(touched):
                return size
    return best


def degree_sequence(edges) -> str:
    deg = Counter(w for e in set(edges) for w in e)
    return "-".join(str(d) for d in sorted(deg.values(), reverse=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=6)
    ap.add_argument("--max-gates", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    degree = args.degree
    cap = (2 * degree + 1) ** len(CUT)
    print(f"degree L={degree}, cut {{0,1}}|{{2,3}}, cut cap {cap}, "
          f"multisets up to {args.max_gates} gates")

    rows = []
    t0 = time.time()
    for n_gates in range(1, args.max_gates + 1):
        # Layers spread over the interior; layer 0 is the weakest slot.
        layers = [1 + i * (degree - 2) // max(n_gates - 1, 1) for i in range(n_gates)]
        layers = sorted(set(layers)) if len(set(layers)) == n_gates else list(range(1, n_gates + 1))

        for multiset in itertools.combinations_with_replacement(CROSSING_EDGES, n_gates):
            best = 0
            for perm in set(itertools.permutations(multiset)):
                ent = sorted(zip(layers, perm))
                best = max(best, measure_cut_rank(
                    N_WIRES, degree, ent, CUT, seeds=args.seeds, n_grid=2 * degree + 4,
                    rng=np.random.default_rng(20260805)))
            edges = list(multiset)
            rows.append({
                "edges": str(multiset), "n_gates": n_gates,
                "n_distinct": len(set(edges)),
                "block_rank": graph_cut_rank(list(zip(layers, multiset)), CUT, N_WIRES),
                "components": components(edges),
                "matching": matching_number(edges),
                "degrees": degree_sequence(edges),
                "measured_rank": best,
            })
            print(f"  {str(multiset):<34} gates={n_gates} "
                  f"blk={rows[-1]['block_rank']} cmp={rows[-1]['components']} "
                  f"mat={rows[-1]['matching']} deg={rows[-1]['degrees']:<7} "
                  f"rank={best}")

    df = pd.DataFrame(rows)
    out = OUT_DIR / "table_cut_graph_enumeration.csv"
    df.to_csv(out, index=False)

    print("\n── which invariants separate the cells ──")
    singles = ["n_gates", "n_distinct", "block_rank", "components", "matching", "degrees"]
    for inv in singles:
        bad = [(key, sorted(set(g.measured_rank)))
               for key, g in df.groupby(inv) if g.measured_rank.nunique() > 1]
        status = "SEPARATES" if not bad else f"collides on {len(bad)} value(s)"
        print(f"  {inv:<12} {status}")
        for key, vals in bad[:3]:
            print(f"      {inv}={key} spans ranks {vals}")

    print("\n  combinations:")
    for combo in itertools.combinations(singles, 2):
        n_bad = sum(g.measured_rank.nunique() > 1
                    for _, g in df.groupby(list(combo)))
        if n_bad == 0:
            print(f"    {' + '.join(combo)}  SEPARATES")

    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
