"""
run_cut_rank_graph_law.py
─────────────────────────
Does the rank of the entangling graph across a cut predict the reachable rank,
and what exactly is the shared-qubit anomaly?

The balanced-cut run established that edge structure rather than gate count
governs a multipartite cut, and left one case uncovered. Two crossing edges that
share a qubit reach 13 at degree 6, which is neither the 12 of a single channel
nor the 16 of two, so :func:`bipartition_rank_law` declines to predict them.

There is a candidate for the missing rule. For quadratic phase states over a
finite field the Rényi-2 entropy across a cut equals the rank of the
corresponding off-diagonal block of the defining symmetric matrix, and the
diagonal drops out as local content (Dahbi and Zaïr, arXiv:2605.05167). Reading
that structurally rather than literally gives a prediction for our setting: what
should matter is the rank of the entangling graph's cut block, so gates piled on
one edge and edges sharing a qubit should both behave as one channel while
disjoint edges behave as two. On the three placements already measured that
prediction is right twice and wrong by exactly one on the shared-qubit case.

This experiment measures the fourth structure, which fixes whether the anomaly
is symmetric in which side of the cut the shared qubit sits on, sweeps the gate
count on shared edges to see whether the discrepancy stays at one or grows, and
tests the obvious explanation. Because the fitted model carries a constant, an
all-ones component contributes a rank of its own throughout this work, so each
cell is reported both raw and with the DC mode projected out. If the anomaly is
the constant, centring removes it from the shared-qubit cell and leaves the
others alone.

Ranks are maximised over random parameter draws and the number of draws needed
grows with the rank being resolved, so cells that are still climbing at the
largest draw count are lower bounds rather than values.

Outputs:
    results/qnn-rankdesign/table_cut_rank_graph_law.csv
"""

from __future__ import annotations

import argparse
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
    circuit_response_nd,
    crossing_count,
    cut_rank,
    graph_cut_rank,
    reachable_rank_law,
)

OUT_DIR = ROOT / "results" / "qnn-rankdesign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_WIRES = 4
CUT = (0, 1)               # the {0,1} | {2,3} bipartition
GATE = "cnot"              # matches the balanced-cut run; rank is gate-indifferent


def structures(degree: int) -> list[tuple[str, list, str]]:
    """The four edge structures at a fixed crossing count of two.

    Layers are held at the same two spread slots throughout so that only the
    edges vary. Layer 0 is the weakest slot and is avoided.
    """
    early, late = 1, degree - 1
    return [
        ("one_edge",
         [(early, (0, 2)), (late, (0, 2))],
         "both gates on a single crossing edge"),
        ("share_inside",
         [(early, (0, 2)), (late, (0, 3))],
         "two edges sharing the wire inside the cut"),
        ("share_outside",
         [(early, (0, 2)), (late, (1, 2))],
         "two edges sharing the wire outside the cut"),
        ("disjoint",
         [(early, (0, 2)), (late, (1, 3))],
         "two edges sharing no wire"),
    ]


def shared_edge_sweep(degree: int) -> list[tuple[str, list, str]]:
    """Shared-qubit structures at growing gate counts.

    If the anomaly is a constant offset it stays at one gate-count after
    gate-count. If it tracks the ladder it grows.
    """
    slots = [1, degree - 1, degree - 2, 2]
    out = []
    for n_gates in (2, 3, 4):
        if n_gates > len(slots):
            break
        edges = [(0, 2), (0, 3)] * 3
        ents = [(slots[i], edges[i]) for i in range(n_gates)]
        out.append((f"share_inside_x{n_gates}", ents,
                    f"{n_gates} gates alternating over two edges sharing wire 0"))
        # The control that fixes the interpretation. A single channel is capped
        # at 2L+1, so if piling gates on one edge also arrives at that cap then
        # sharing a qubit is not opening anything new, it is only climbing the
        # one channel faster. Without this the shared cell's value cannot be
        # told apart from the ceiling it happens to sit on.
        out.append((f"one_edge_x{n_gates}",
                    [(slots[i], (0, 2)) for i in range(n_gates)],
                    f"{n_gates} gates all on the single edge (0,2)"))
    return out


def _double_centre(mat: np.ndarray) -> np.ndarray:
    """Remove the additive part of ``mat``, leaving the interaction part.

    Subtracting the grid mean alone is the wrong probe and was tried first: an
    all-ones matrix is generally *outside* the response's column space, so
    subtracting it adds a direction and raises the rank rather than lowering it.
    What has to come off is everything of the form ``u 1^T + 1 v^T``, the
    additive-in-the-two-sides structure of which a bare constant is the special
    case. This is the matricised form of the separable-plus-constant subtraction
    the two-qubit work already uses.
    """
    return (mat - mat.mean(axis=1, keepdims=True)
            - mat.mean(axis=0, keepdims=True) + mat.mean())


def measure_both(degree: int, entanglers, seeds: int, rng) -> tuple[int, int]:
    """Reachable cut rank, raw and after the additive part is removed.

    Both are maximised over the same parameter draws, so the difference between
    the two columns is whether the structure separating one cell from another is
    genuinely an interaction across the cut or merely additive.
    """
    n_grid = max(12, 2 * degree + 4)
    size = n_grid ** len(CUT)
    best_raw = best_centred = 0
    for _ in range(seeds):
        rot_w = rng.standard_normal((degree + 1, N_WIRES, 3))
        resp = circuit_response_nd(N_WIRES, degree, entanglers, rot_w, n_grid, GATE)
        best_raw = max(best_raw, cut_rank(resp, CUT))
        rest = [m for m in range(N_WIRES) if m not in CUT]
        mat = np.transpose(resp, list(CUT) + rest).reshape(size, size)
        s = np.linalg.svd(_double_centre(mat), compute_uv=False)
        best_centred = max(best_centred, int(np.sum(s > 1e-7 * s[0])))
    return best_raw, best_centred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--seeds", type=int, default=6,
                    help="parameter draws per cell; ranks are maxima over these")
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for degree in args.degrees:
        cap = (2 * degree + 1) ** len(CUT)
        print(f"\ndegree L={degree}, cut {set(CUT)}|{{2,3}}, "
              f"per-axis box {2 * degree + 1}, cut cap {cap}")
        print(f"  {'structure':<18} {'cross':>5} {'gcut':>5} {'law':>5} "
              f"{'rank':>5} {'cent':>5}  verdict")

        cases = structures(degree) + shared_edge_sweep(degree)
        for name, ents, blurb in cases:
            rng = np.random.default_rng(20260805)
            n_cross = crossing_count(ents, CUT)
            g_rank = graph_cut_rank(ents, CUT, N_WIRES)
            law = bipartition_rank_law(ents, CUT, degree, N_WIRES)
            raw, no_dc = measure_both(degree, ents, args.seeds, rng)

            # The channel reading of the graph rank: every open channel climbs
            # its own two-qubit ladder, and with the gates split evenly the
            # per-channel budget is the gate count divided by the channels.
            per_channel = n_cross // max(g_rank, 1)
            channel_pred = min(reachable_rank_law(per_channel, degree) ** max(g_rank, 1), cap)

            if law is None:
                verdict = f"law silent, channels predict {channel_pred}"
            elif law == raw:
                verdict = "law exact"
            else:
                verdict = f"law {law} vs measured {raw}"

            print(f"  {name:<18} {n_cross:>5} {g_rank:>5} "
                  f"{str(law) if law is not None else '-':>5} "
                  f"{raw:>5} {no_dc:>5}  {verdict}")

            rows.append({
                "degree": degree, "structure": name, "description": blurb,
                "entanglers": str(ents), "crossing_count": n_cross,
                "graph_cut_rank": g_rank, "channel_prediction": channel_pred,
                "bipartition_law": law, "measured_rank": raw,
                "measured_rank_centred": no_dc, "seeds": args.seeds,
                "cut_cap": cap,
            })

    df = pd.DataFrame(rows)
    out = OUT_DIR / "table_cut_rank_graph_law.csv"
    df.to_csv(out, index=False)

    print(f"\n── read-out ──")
    for degree in args.degrees:
        sub = df[(df.degree == degree) & (df.structure.isin(
            ["one_edge", "share_inside", "share_outside", "disjoint"]))]
        by_g = sub.groupby("graph_cut_rank")["measured_rank"].apply(list).to_dict()
        print(f"  L={degree}: measured rank grouped by graph cut rank {by_g}")
        agrees = all(len(set(v)) == 1 for v in by_g.values())
        print(f"    equal graph rank gives equal reachable rank: "
              f"{'YES' if agrees else 'NO'}")
        shared = sub[sub.structure.str.startswith("share")]
        if len(shared):
            drops = (shared.measured_rank - shared.measured_rank_centred).tolist()
            print(f"    shared-qubit rank lost to additive structure: {drops}")

    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
