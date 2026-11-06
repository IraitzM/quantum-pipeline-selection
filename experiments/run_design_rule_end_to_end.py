"""
run_design_rule_end_to_end.py
─────────────────────────────
Does the design rule actually choose a good ansatz on real data?

Everything in ``code.rankdesign`` so far establishes bounds on controlled
synthetic targets: what rank a budget of entanglers reaches, that bandwidth is
indifferent to them, that a fixed entangler breaks nesting. None of it shows
that reading two numbers off an image and turning them into a circuit produces
a circuit worth having. That is what this experiment tests, and it is the only
form of the claim a practitioner can use.

The procedure is :func:`code.rankdesign.choose_ansatz`, which takes the 2-D
effective bandwidth and the SVD effective rank of a target and returns a depth,
an entangler budget and a placement without training anything. We compare the
circuit it picks against two ways of not using it. The first is a fixed default,
depth 5 with an entangler in every layer, which is the top of the sweep the
paper's existing 2-D figure runs and a fair account of what one would build
knowing nothing about the target. The second is maximal entanglement at the
depth the rule chose, which strips the comparison down to the entangler budget
alone and asks the question the paper's "budget entanglement, don't maximise it"
line rests on. Against both sits a hand-tuned oracle, the best cell of the full
depth-by-budget grid, which costs 28 fits per image to find and is exactly the
search the rule is meant to replace.

The oracle turns out to be the largest cell of the grid on every real image,
because none of these targets saturates the model and capacity always pays. That
makes it a poor test of the rule on its own: scored against the biggest circuit
available the rule can only lose, and it loses for having chosen a cheaper
circuit rather than a wrong one. So the grid is kept in full and the rule is also
scored against the best cell costing no more than the one it chose. That
iso-parameter comparison is the calibration question, and it has a sharp answer
per image, namely whether the rule's cell is the argmax among the cells it could
have afforded.

Images are drawn from nine 2-D MedMNIST modalities and stratified by the budget
the rule assigns them, because an unstratified sample does not test the rule.
Sampling the first images of any single modality returns ``k = 2`` almost
without exception, so a constant recommendation would score as well as a
responsive one and the experiment would be vacuous. Stratifying spans the ladder
the rule can express: retinal OCT is layered and nearly separable, colon
histology is textured and high-rank, and the rule sees the difference before any
fitting happens.

Budget and modality are confounded and cannot be separated on this data. Only
OCT supplies ``k = 1`` images in any number and only colon histology supplies
``k = 3``, so the buckets are close to single-modality at both ends of the
ladder. That is a property of the images rather than of the sampling, and no
reweighting removes it. It bounds what the per-bucket rows below can support: a
difference that appears only in one bucket is a statement about that modality as
much as about that budget. The paired comparisons are unaffected, because the
rule, the oracle and both defaults are scored on the same image. The bucket
sizes are equal by construction and are not the population proportions, in which
``k = 2`` would be the overwhelming majority.

Replication is across images rather than across optimiser seeds, and every
comparison is paired within an image, so the reported spread is variation in the
targets rather than in the initialisation. Fits are best-of-two LBFGS restarts
at three rounds, which the timing probe found indistinguishable from six.

Outputs:
    results/qnn-rankdesign/table_design_rule.csv
    results/qnn-rankdesign/table_design_rule_summary.csv
    results/qnn-rankdesign/figure_design_rule.png
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

from code.baselines import separable_plus_const_r2, svd_effective_rank  # noqa: E402
from code.rankdesign import (  # noqa: E402
    EntanglerLadder2DQNN,
    choose_ansatz,
    fitted_model_rank,
    omega_eff_2d,
    parameter_count,
    recommended_placement,
    target_dof,
    truncated_svd_r2,
)

OUT_DIR = ROOT / "results" / "qnn-rankdesign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 16                       # pixels per axis, and the coordinate grid
DEGREES = list(range(2, 9))     # the hand-tuned search the rule replaces
BUDGETS = [0, 1, 2, 3]
DEFAULT_DEGREE = 5              # top of the existing 2-D sweep, chosen blind
RESTARTS = 2
LBFGS_ROUNDS = 3
LBFGS_ITERS = 150
PER_BUCKET = 8                  # images per entangler budget the rule assigns
BUCKETS = [1, 2, 3]
CANDIDATES_PER_SET = 80        # k=3 is ~1 image in 6, and only in histology

DATASETS = [
    "octmnist", "dermamnist", "breastmnist", "retinamnist", "bloodmnist",
    "pneumoniamnist", "tissuemnist", "organamnist", "pathmnist",
]


def _load_grey(name: str, n: int) -> list[np.ndarray]:
    """First ``n`` training images of a 2-D MedMNIST set, greyscale and rescaled
    to the coordinate grid. Colour sets are collapsed by luminance rather than
    by channel mean, so the result matches the perceived structure."""
    import medmnist
    from skimage.transform import resize

    cls = getattr(medmnist, medmnist.INFO[name]["python_class"])
    ds = cls(split="train", download=True, size=28)
    out = []
    for i in range(min(n, len(ds))):
        a = np.asarray(ds[i][0], dtype=np.float64)
        if a.ndim == 3:
            a = a @ np.array([0.299, 0.587, 0.114])
        a = resize(a / (a.max() + 1e-12), (GRID, GRID), anti_aliasing=True)
        out.append((a - a.mean()) / (a.std() + 1e-8))
    return out


def build_pool() -> list[dict]:
    """Stratified image sample, ``PER_BUCKET`` per entangler budget.

    Candidates are scored by :func:`choose_ansatz` alone, which touches no
    labels and trains nothing, so the stratification cannot leak fit quality
    into the selection. Buckets are filled round-robin across modalities so a
    single dataset cannot supply a whole bucket and confound budget with
    modality.
    """
    by_bucket: dict[int, list[dict]] = {k: [] for k in BUCKETS}
    per_set: dict[str, dict[int, list[dict]]] = {}
    for name in DATASETS:
        try:
            imgs = _load_grey(name, CANDIDATES_PER_SET)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: unavailable ({type(exc).__name__}), skipped")
            continue
        buckets: dict[int, list[dict]] = {k: [] for k in BUCKETS}
        for i, img in enumerate(imgs):
            rec = choose_ansatz(img)
            if rec["n_entanglers"] in buckets:
                buckets[rec["n_entanglers"]].append(
                    dict(dataset=name, index=i, img=img, rec=rec)
                )
        per_set[name] = buckets
        print(f"  {name}: " + "  ".join(f"k={k} {len(v)}" for k, v in buckets.items()),
              flush=True)

    for k in BUCKETS:
        round_robin = 0
        while len(by_bucket[k]) < PER_BUCKET:
            added = False
            for name in per_set:
                pool = per_set[name][k]
                if round_robin < len(pool) and len(by_bucket[k]) < PER_BUCKET:
                    by_bucket[k].append(pool[round_robin])
                    added = True
            if not added:
                break
            round_robin += 1
        if len(by_bucket[k]) < PER_BUCKET:
            print(f"  WARNING: bucket k={k} holds only {len(by_bucket[k])} images")

    return [rec for k in BUCKETS for rec in by_bucket[k]]


def fit(img_flat, xs, ys, degree: int, ent_layers, seed: int) -> float:
    torch.manual_seed(seed)
    model = EntanglerLadder2DQNN(degree=degree, ent_layers=ent_layers, gate="crz")
    for _ in range(LBFGS_ROUNDS):
        opt = torch.optim.LBFGS(
            model.parameters(), max_iter=LBFGS_ITERS, history_size=50,
            line_search_fn="strong_wolfe", tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

        def closure():
            opt.zero_grad()
            loss = ((model(xs, ys) - img_flat) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)

    with torch.no_grad():
        pred = model(xs, ys).numpy()
    tgt = img_flat.numpy()
    ss_res = float(np.sum((tgt - pred) ** 2))
    ss_tot = float(np.sum((tgt - tgt.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def best_fit(img_flat, xs, ys, degree: int, ent_layers, tag: int) -> float:
    return max(fit(img_flat, xs, ys, degree, ent_layers, 1000 * tag + r)
               for r in range(RESTARTS))


def main():
    print(f"Building stratified pool at grid {GRID} "
          f"({PER_BUCKET} images per budget, buckets {BUCKETS})")
    pool = build_pool()
    print(f"\n{len(pool)} images selected\n")

    axis = np.arange(GRID) * 2 * np.pi / GRID
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    xs = torch.tensor(X.ravel(), dtype=torch.float64)
    ys = torch.tensor(Y.ravel(), dtype=torch.float64)

    rows, grid_rows, t0 = [], [], time.time()
    for n, item in enumerate(pool):
        img, rec = item["img"], item["rec"]
        img_flat = torch.tensor(img.ravel(), dtype=torch.float64)
        tag = n * 100

        grid_scores = {}
        for degree in DEGREES:
            for k in BUDGETS:
                grid_scores[(degree, k)] = best_fit(
                    img_flat, xs, ys, degree, recommended_placement(k, degree),
                    tag + 10 * degree + k,
                )

        # The oracle is the best of the hand-tuned grid and nothing else. The
        # rule's own cell is normally inside that grid, but when it is not it is
        # fitted separately rather than folded in, so the oracle stays the same
        # fixed search on every image instead of quietly borrowing the rule's
        # answer on some of them.
        (oracle_degree, oracle_k), oracle_r2 = max(
            grid_scores.items(), key=lambda kv: kv[1]
        )
        rule_key = (rec["degree"], rec["n_entanglers"])
        rule_r2 = grid_scores.get(rule_key)
        if rule_r2 is None:
            rule_r2 = best_fit(img_flat, xs, ys, rec["degree"],
                               rec["ent_layers"], tag + 90)

        # On real targets the oracle is always the largest cell in the grid,
        # because nothing here saturates and capacity always pays. Scored
        # against it the rule can only lose, and it loses for choosing a smaller
        # circuit rather than for choosing the wrong one, which is not a
        # calibration test. The question that does discriminate is whether the
        # rule picked the best configuration available *at its own cost*, so we
        # also record the best cell no more expensive than the one it chose.
        rule_params = parameter_count(rec["degree"], rec["n_entanglers"])
        affordable = {cell: r2 for cell, r2 in grid_scores.items()
                      if parameter_count(*cell) <= rule_params}
        # The rule's own cell has to be in its own affordable set, which it is
        # not when the grid did not contain it and it was fitted separately.
        affordable[rule_key] = rule_r2
        (iso_degree, iso_k), iso_r2 = max(affordable.items(), key=lambda kv: kv[1])

        for (degree, k), r2 in sorted(grid_scores.items()):
            grid_rows.append(dict(
                image=f"{item['dataset']}[{item['index']}]", degree=degree,
                n_entanglers=k, r2=r2, n_params=parameter_count(degree, k),
                is_rule_cell=(degree, k) == rule_key,
            ))

        # Maximal entanglement at the rule's own depth isolates the budget.
        maxent_r2 = best_fit(img_flat, xs, ys, rec["degree"],
                             range(rec["degree"]), tag + 91)
        # The blind default: fixed depth, an entangler everywhere.
        default_r2 = best_fit(img_flat, xs, ys, DEFAULT_DEGREE,
                              range(DEFAULT_DEGREE), tag + 92)

        rows.append(dict(
            image=f"{item['dataset']}[{item['index']}]", dataset=item["dataset"],
            omega_eff=rec["omega_eff"], effective_rank=rec["effective_rank"],
            separable_ceiling=separable_plus_const_r2(img),
            rule_degree=rec["degree"], rule_k=rec["n_entanglers"],
            rule_r2=rule_r2,
            oracle_degree=oracle_degree, oracle_k=oracle_k, oracle_r2=oracle_r2,
            iso_degree=iso_degree, iso_k=iso_k, iso_r2=iso_r2,
            gap_to_iso=iso_r2 - rule_r2,
            rule_is_iso_best=bool((iso_degree, iso_k) == rule_key),
            n_affordable_cells=len(affordable),
            maxent_k=rec["degree"], maxent_r2=maxent_r2,
            default_degree=DEFAULT_DEGREE, default_k=DEFAULT_DEGREE,
            default_r2=default_r2,
            gap_to_oracle=oracle_r2 - rule_r2,
            gain_over_default=rule_r2 - default_r2,
            gain_over_maxent=rule_r2 - maxent_r2,
            entanglers_saved_vs_maxent=rec["degree"] - rec["n_entanglers"],
            svd_ceiling_at_rule_rank=truncated_svd_r2(
                img, min(fitted_model_rank(rec["n_entanglers"], rec["degree"]), GRID)
            ),
            rule_params=parameter_count(rec["degree"], rec["n_entanglers"]),
            default_params=parameter_count(DEFAULT_DEGREE, DEFAULT_DEGREE),
            target_dof=target_dof(int(round(rec["effective_rank"])), rec["degree"]),
            dof_sufficient=rec["dof_sufficient"],
        ))
        r = rows[-1]
        print(f"[{n + 1:>2}/{len(pool)}] {r['image']:<22} "
              f"w={r['omega_eff']} er={r['effective_rank']:.1f} -> "
              f"L={r['rule_degree']} k={r['rule_k']}  "
              f"rule {rule_r2:.3f}  iso {iso_r2:.3f} "
              f"(L={iso_degree} k={iso_k}{'=rule' if r['rule_is_iso_best'] else ''})  "
              f"oracle {oracle_r2:.3f} (L={oracle_degree} k={oracle_k})  "
              f"maxent {maxent_r2:.3f}  default {default_r2:.3f}   "
              f"[{time.time() - t0:.0f}s]", flush=True)
        # Each image costs ~28 fits, so checkpoint rather than risk the lot.
        pd.DataFrame(rows).to_csv(OUT_DIR / "table_design_rule.csv", index=False)
        pd.DataFrame(grid_rows).to_csv(OUT_DIR / "table_design_rule_grid.csv",
                                       index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_design_rule.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT_DIR / "table_design_rule_grid.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'table_design_rule.csv'}  "
          f"({time.time() - t0:.0f}s)")

    _report(df)
    _plot(df)


def _paired(df, column: str) -> str:
    """Mean paired difference with its standard error, in sigma."""
    v = df[column].to_numpy()
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    sigma = v.mean() / sem if sem else float("nan")
    return f"{v.mean():+.4f} +/- {sem:.4f}  ({sigma:+.1f} sigma, n={len(v)})"


def _report(df):
    print("\nPaired against the rule, across images")
    print(f"  gap to best cell at the same cost {_paired(df, 'gap_to_iso')}")
    print(f"  gap to hand-tuned oracle          {_paired(df, 'gap_to_oracle')}")
    print(f"  gain over blind default           {_paired(df, 'gain_over_default')}")
    print(f"  gain over max entanglement        {_paired(df, 'gain_over_maxent')}")
    print(f"  entanglers saved vs maxent "
          f"{df.entanglers_saved_vs_maxent.mean():.1f} of "
          f"{df.rule_degree.mean():.1f} layers on average")

    print("\nIs the rule the best choice at its own cost?")
    print(f"  rule cell is the iso-parameter argmax in {df.rule_is_iso_best.mean():.0%} "
          f"of images (mean {df.n_affordable_cells.mean():.1f} cells affordable)")

    print("\nIs the oracle just the biggest circuit in the grid?")
    corner = ((df.oracle_degree == max(DEGREES)) & (df.oracle_k == max(BUDGETS))).mean()
    print(f"  oracle sits at the grid corner in {corner:.0%} of images")

    print("\nDoes the rule land on the oracle's cell?")
    same_k = (df.rule_k == df.oracle_k).mean()
    same_both = ((df.rule_k == df.oracle_k) & (df.rule_degree == df.oracle_degree)).mean()
    print(f"  same budget {same_k:.0%}, same budget and depth {same_both:.0%}")
    print(f"  oracle budget distribution {dict(sorted(df.oracle_k.value_counts().items()))}")
    print(f"  rule   budget distribution {dict(sorted(df.rule_k.value_counts().items()))}")

    print("\nBy the budget the rule assigned")
    for k, sub in df.groupby("rule_k"):
        print(f"  k={k} (n={len(sub)}): rule {sub.rule_r2.mean():.3f}  "
              f"oracle {sub.oracle_r2.mean():.3f}  "
              f"maxent {sub.maxent_r2.mean():.3f}  "
              f"default {sub.default_r2.mean():.3f}")

    print("\nIs the rank ceiling the binding constraint?")
    print(f"  mean fitted R2 at the rule's cell   {df.rule_r2.mean():.3f}")
    print(f"  mean SVD ceiling at that same rank  {df.svd_ceiling_at_rule_rank.mean():.3f}")
    print(f"  dof budget sufficient in            {df.dof_sufficient.mean():.0%} of images")

    summary = (
        df.groupby("rule_k")
        .agg(n=("rule_r2", "size"), rule_r2=("rule_r2", "mean"),
             oracle_r2=("oracle_r2", "mean"), maxent_r2=("maxent_r2", "mean"),
             default_r2=("default_r2", "mean"), iso_r2=("iso_r2", "mean"),
             gap_to_iso=("gap_to_iso", "mean"),
             rule_is_iso_best=("rule_is_iso_best", "mean"),
             gap_to_oracle=("gap_to_oracle", "mean"),
             gain_over_default=("gain_over_default", "mean"),
             gain_over_maxent=("gain_over_maxent", "mean"),
             svd_ceiling=("svd_ceiling_at_rule_rank", "mean"),
             dof_sufficient=("dof_sufficient", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "table_design_rule_summary.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'table_design_rule_summary.csv'}")


def _plot(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    order = df.sort_values(["rule_k", "effective_rank"]).reset_index(drop=True)
    idx = np.arange(len(order))
    ax = axes[0]
    ax.plot(idx, order.oracle_r2, "k_", ms=14, label="hand-tuned oracle")
    ax.plot(idx, order.rule_r2, "o", ms=6, color="#2c7fb8", label="design rule")
    ax.plot(idx, order.maxent_r2, "v", ms=5, color="#d95f02", alpha=0.8,
            label="max entanglement, rule's depth")
    ax.plot(idx, order.default_r2, "x", ms=5, color="#7570b3", alpha=0.8,
            label=f"blind default (L={DEFAULT_DEGREE}, k=L)")
    for k in BUCKETS[:-1]:
        edge = (order.rule_k <= k).sum() - 0.5
        ax.axvline(edge, color="0.8", lw=1)
    ax.set_xlabel("image, grouped by the budget the rule assigned")
    ax.set_ylabel("R²")
    ax.set_title("Every image, rule against the alternatives")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for col, label, colour in (("gap_to_iso", "best at same cost − rule", "#1b9e77"),
                               ("gap_to_oracle", "oracle − rule", "#2c7fb8"),
                               ("gain_over_maxent", "rule − max entanglement", "#d95f02"),
                               ("gain_over_default", "rule − blind default", "#7570b3")):
        v = order[col].to_numpy()
        ax.barh(label, v.mean(),
                xerr=v.std(ddof=1) / np.sqrt(len(v)), color=colour, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("paired difference in R²  (error bar = s.e.m. across images)")
    ax.set_title("Paired comparisons")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[2]
    sc = ax.scatter(df.effective_rank, df.rule_r2, c=df.rule_k,
                    cmap="viridis", s=45, label="rule")
    ax.scatter(df.effective_rank, df.svd_ceiling_at_rule_rank, marker="_",
               color="0.4", s=90, label="SVD ceiling at that rank")
    ax.set_xlabel("SVD effective rank of the target")
    ax.set_ylabel("R²")
    ax.set_title("What binds: rank ceiling sits far above the fit")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="entangler budget the rule chose")

    fig.suptitle("The design rule applied end to end on real 2-D medical images")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_design_rule.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_design_rule.png'}")


if __name__ == "__main__":
    main()
