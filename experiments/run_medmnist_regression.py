"""
run_medmnist_regression.py
────────────────────────────
2-D analogue of run_signal_regression.py: does approximating a genuinely
two-dimensional signal (a real chest X-ray from PneumoniaMNIST) need
entanglement?

A 2-qubit re-uploading circuit with no entangling gate between the wires
stays a product state, so <Z0 Z1> = <Z0><Z1> is *provably* a separable
function g(x) h(y) -- the exact 2-D analogue of a rank-1 matrix
approximation. Entangling the two wires (a CNOT per re-upload) lifts this
restriction. We fit both variants to real (downsampled) chest X-ray images
and compare against classical references: the best separable-plus-constant
fit, which is the ceiling for the product-ansatz case because the model
carries a trainable output offset on top of <Z0 Z1>, and a full
non-separable 2-D truncated Fourier least-squares fit as the reference for
the entangled case. The plain rank-1 SVD is kept as a reported column for
continuity, but it is the ceiling for s*g(x)h(y) with no offset and so is
not the bound the fitted product model actually faces.

Outputs:
    results/qnn-medmnist/regression_2d_sweep.csv
    results/qnn-medmnist/figure_2d_regression.png
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

from code.baselines import (  # noqa: E402
    classical_fourier_regression_2d,
    rank1_svd_r2,
    separable_plus_const_r2,
    svd_effective_rank,
)
from code.datasets import load_pneumoniamnist  # noqa: E402
from code.qnn import FourierSeries2DQNN  # noqa: E402

OUT_DIR = ROOT / "results" / "qnn-medmnist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 10
DEGREES = [1, 2, 3, 4, 5]
STEPS = 150
LR = 0.05
SEED = 42


def downsample(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    from skimage.transform import resize

    return resize(img, (size, size), anti_aliasing=True)


def fit_2d_qnn(X, Y, img, degree, entangled, seed=SEED):
    torch.manual_seed(seed)
    model = FourierSeries2DQNN(degree=degree, entangled=entangled)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x_t = torch.tensor(X.flatten(), dtype=torch.float64)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float64)
    z_t = torch.tensor(img.flatten(), dtype=torch.float64)

    t0 = time.time()
    for _ in range(STEPS):
        opt.zero_grad()
        pred = model(x_t, y_t)
        loss = ((pred - z_t) ** 2).mean()
        loss.backward()
        opt.step()
    dt = time.time() - t0

    with torch.no_grad():
        pred = model(x_t, y_t).numpy()
    ss_res = np.sum((img.flatten() - pred) ** 2)
    ss_tot = np.sum((img.flatten() - img.mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    n_params = (degree + 1) * 2 * 3 + 2
    return {"r2": r2, "n_params": n_params, "train_time_s": round(dt, 1)}


def get_images():
    p = load_pneumoniamnist()
    idx_normal = np.where(p["y_train"] == 0)[0][0]
    idx_pneumonia = np.where(p["y_train"] == 1)[0][0]
    return {
        "chestxray_normal": downsample(p["x_train"][idx_normal]),
        "chestxray_pneumonia": downsample(p["x_train"][idx_pneumonia]),
    }


def main():
    images = get_images()
    coords = np.linspace(-np.pi, np.pi, IMG_SIZE, endpoint=False)
    X, Y = np.meshgrid(coords, coords, indexing="ij")

    rows = []
    for name, img in images.items():
        # Every diagnostic below is deliberately computed on the centred image,
        # because the fitted model carries a free bias term. Effective rank is
        # scale-invariant but not shift-invariant, so the recorded ranks are the
        # centred ones (5.15 and 5.77); the raw images give 2.24 and 1.93.
        img = (img - img.mean()) / (img.std() + 1e-8)
        rank1_r2 = rank1_svd_r2(img)
        # The fitted product model is scale * g(x) h(y) + bias, so the rank-1
        # SVD is not its ceiling; the separable-plus-constant fit is.
        product_ceiling = separable_plus_const_r2(img)
        eff_rank = svd_effective_rank(img)
        print(
            f"[{name}] rank-1 SVD R2={rank1_r2:.3f}  "
            f"product ceiling (separable+const) R2={product_ceiling:.3f}  "
            f"effective rank={eff_rank:.2f}"
        )

        for degree in DEGREES:
            classical_full = classical_fourier_regression_2d(X, Y, img, degree)
            qnn_entangled = fit_2d_qnn(X, Y, img, degree, entangled=True)
            qnn_product = fit_2d_qnn(X, Y, img, degree, entangled=False)
            rows.append(
                dict(
                    image=name,
                    degree=degree,
                    rank1_svd_r2=rank1_r2,
                    product_ceiling_r2=product_ceiling,
                    effective_rank=eff_rank,
                    classical_full_r2=classical_full["r2"],
                    classical_full_n_params=classical_full["n_params"],
                    qnn_entangled_r2=qnn_entangled["r2"],
                    qnn_product_r2=qnn_product["r2"],
                    qnn_n_params=qnn_entangled["n_params"],
                )
            )
            print(
                f"  degree={degree} classical_full_R2={classical_full['r2']:.3f} "
                f"qnn_entangled_R2={qnn_entangled['r2']:.3f} "
                f"qnn_product_R2={qnn_product['r2']:.3f} "
                f"(product ceiling={product_ceiling:.3f})"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regression_2d_sweep.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'regression_2d_sweep.csv'}")

    _plot(images, df)


def _plot(images, df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(images.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 8))
    for j, name in enumerate(names):
        sub = df[df.image == name]
        ax = axes[0, j]
        ax.plot(sub.degree, sub.qnn_entangled_r2, "o-", label="QNN (entangled)")
        ax.plot(sub.degree, sub.qnn_product_r2, "s--", label="QNN (product)")
        ax.plot(sub.degree, sub.classical_full_r2, "^:", label="classical 2D Fourier LS")
        ax.axhline(sub.product_ceiling_r2.iloc[0], color="gray", linestyle=":",
                   label=f"product-ansatz ceiling, separable+const "
                         f"({sub.product_ceiling_r2.iloc[0]:.2f})")
        ax.axhline(sub.rank1_svd_r2.iloc[0], color="gray", linestyle="-.", alpha=0.5,
                   label=f"rank-1 SVD ({sub.rank1_svd_r2.iloc[0]:.2f})")
        ax.set_title(f"{name}\n(effective rank={sub.effective_rank.iloc[0]:.2f})")
        ax.set_xlabel("Fourier degree / re-upload depth")
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.legend(fontsize=7)

        ax2 = axes[1, j]
        ax2.imshow(images[name], cmap="gray")
        ax2.set_title("original image (downsampled)")
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_2d_regression.png", dpi=150)
    print(f"Saved {OUT_DIR / 'figure_2d_regression.png'}")


if __name__ == "__main__":
    main()
