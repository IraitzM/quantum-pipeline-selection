"""
run_medmnist_regression_seeds.py
────────────────────────────────
Seed-stability check for the 2-D regression result of
``run_medmnist_regression.py``, which is single-seed.

The paper's reading of that result turns on whether the product ansatz's
shortfall against its separable-plus-constant ceiling is a stable property or
an unlucky initialisation, so the degree-5 fits are repeated over several
seeds. Both ansatz variants are run, since the entangled margin needs the
same treatment.

Outputs:
    results/qnn-medmnist/regression_2d_seeds.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.baselines import separable_plus_const_r2  # noqa: E402

from run_medmnist_regression import (  # noqa: E402
    DEGREES,
    IMG_SIZE,
    OUT_DIR,
    downsample,
    fit_2d_qnn,
    get_images,
)

SEEDS = [42, 0, 1, 2, 3]


def main():
    coords = np.linspace(-np.pi, np.pi, IMG_SIZE, endpoint=False)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    degree = max(DEGREES)

    rows = []
    for name, raw in get_images().items():
        img = downsample(raw)
        img = (img - img.mean()) / (img.std() + 1e-8)
        ceiling = separable_plus_const_r2(img)
        for entangled in (False, True):
            vals = [fit_2d_qnn(X, Y, img, degree, entangled, seed=s)["r2"] for s in SEEDS]
            rows.append(
                dict(
                    image=name,
                    degree=degree,
                    ansatz="entangled" if entangled else "product",
                    product_ceiling_r2=ceiling,
                    mean_r2=float(np.mean(vals)),
                    std_r2=float(np.std(vals)),
                    n_seeds=len(SEEDS),
                    **{f"seed_{s}": v for s, v in zip(SEEDS, vals)},
                )
            )
            print(
                f"[{name}] {rows[-1]['ansatz']:9s} degree={degree} "
                f"R2={rows[-1]['mean_r2']:.4f} +/- {rows[-1]['std_r2']:.4f} "
                f"(product ceiling={ceiling:.4f})",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regression_2d_seeds.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'regression_2d_seeds.csv'}")


if __name__ == "__main__":
    main()
