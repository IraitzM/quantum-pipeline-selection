"""
baselines.py
────────────
Classical reference points for the angle-encoding QNN experiments.

* ``classical_fourier_regression`` — explicit truncated Fourier-series least
  squares fit of a 1-D signal, degree-matched to the QNN's re-upload count.
  This is the classical analogue of what the single-qubit re-uploading
  circuit computes implicitly, so a head-to-head comparison isolates
  whether the quantum circuit adds anything beyond "compute the same
  Fourier basis classically".

* ``TorchMLP`` — a tiny feed-forward classifier, instantiated at two
  capacities: parameter-matched to the QNN, and a conventional
  moderately-sized network, so both an iso-parameter and an iso-practice
  comparison are available.

* ``rbf_svm_scores`` — RBF-kernel SVC, the standard nonlinear classical
  kernel-machine reference used throughout this project. Its bandwidth is
  selected on an 11-point geometric grid so that the classical baseline gets a
  search budget comparable to the quantum models', which is what the paper
  claims for it.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn


def classical_fourier_regression(t: np.ndarray, y: np.ndarray, degree: int) -> dict:
    """Least-squares fit of y(t) with a degree-`degree` Fourier series."""
    design = [np.ones_like(t)]
    for k in range(1, degree + 1):
        design.append(np.cos(k * t))
        design.append(np.sin(k * t))
    A = np.stack(design, axis=1)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    n_params = len(coeffs)
    return {"r2": r2, "y_hat": y_hat, "n_params": n_params}


def rank1_svd_r2(img: np.ndarray, k: int = 1) -> float:
    """R^2 of the best rank-k SVD reconstruction. A separable function g(x) h(y)
    is exactly a rank-1 matrix, so this bounds the correlator <Z0 Z1> of a
    product-state 2-qubit (x, y) circuit.

    It does *not* bound the fitted model, which adds a trainable offset and so
    has rank 2; use :func:`separable_plus_const_r2` for that.
    """
    u, s, vt = np.linalg.svd(img, full_matrices=False)
    recon = (u[:, :k] * s[:k]) @ vt[:k, :]
    ss_res = np.sum((img - recon) ** 2)
    ss_tot = np.sum((img - img.mean()) ** 2) + 1e-12
    return float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))


def separable_plus_const_r2(img: np.ndarray, n_coarse: int = 401, n_fine: int = 401) -> float:
    """R^2 ceiling for the model class ``s * g(x) h(y) + b`` -- i.e. a separable
    function plus a constant offset.

    This, not :func:`rank1_svd_r2`, is the ceiling the product-ansatz circuit
    actually faces. ``FourierSeries2DQNN`` returns ``scale * <Z0 Z1> + bias``,
    and with no entangling gate ``<Z0 Z1> = <Z0><Z1>`` is separable, so on a
    coordinate grid the model output is a rank-1 matrix plus a constant one:
    rank at most 2, matching the bound proved in the paper's rank appendix.
    A rank-1 SVD is the ceiling for ``s * g(x) h(y)`` alone and is neither an
    upper nor a lower bound on this class.

    Because increasing the re-upload degree only enlarges the reachable set,
    this is a valid upper bound at every degree. It is tight only once the
    degree reaches the grid's Nyquist limit, where ``g`` and ``h`` can be
    arbitrary; below that the circuit's band-limit binds as well. It also
    ignores the constraints the single-qubit coefficient set places on ``g``
    and ``h``, so the reachable ceiling is somewhat lower still.

    For a fixed offset ``b`` the best separable term is the rank-1 SVD of
    ``img - b``, so we minimise over the scalar ``b`` alone, by a coarse sweep
    over +/- 3 standard deviations followed by a refinement around the best
    point.
    """

    def _best_for_offset(b: float) -> float:
        u, s, vt = np.linalg.svd(img - b, full_matrices=False)
        recon = np.outer(u[:, 0] * s[0], vt[0, :]) + b
        ss_res = np.sum((img - recon) ** 2)
        ss_tot = np.sum((img - img.mean()) ** 2) + 1e-12
        return float(1 - ss_res / ss_tot)

    span = 3.0 * img.std() + 1e-12
    coarse = np.linspace(-span, span, n_coarse)
    scores = np.array([_best_for_offset(b) for b in coarse])
    best_idx = int(np.argmax(scores))
    step = coarse[1] - coarse[0]
    fine = np.linspace(coarse[best_idx] - step, coarse[best_idx] + step, n_fine)
    return float(np.clip(max(scores[best_idx], max(_best_for_offset(b) for b in fine)), 0.0, 1.0))


def svd_effective_rank(img: np.ndarray) -> float:
    """Entropy-based effective rank of an image's singular value spectrum.

    Low effective rank (near 1) means the image is close to separable
    (g(x) h(y)); high effective rank means genuine 2-D structure that a
    rank-1 / product-state model cannot capture.
    """
    s = np.linalg.svd(img, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def classical_fourier_regression_2d(
    x: np.ndarray, y: np.ndarray, img: np.ndarray, degree: int
) -> dict:
    """Least-squares fit of a 2-D signal img(x, y) with a full (non-separable)
    truncated 2-D Fourier series of degree `degree` in each axis -- the
    classical analogue of what an *entangled* 2-qubit circuit can represent.
    """
    x_flat, y_flat, z_flat = x.flatten(), y.flatten(), img.flatten()
    design = [np.ones_like(x_flat)]
    for m in range(degree + 1):
        for n in range(degree + 1):
            if m == 0 and n == 0:
                continue
            design.append(np.cos(m * x_flat + n * y_flat))
            design.append(np.sin(m * x_flat + n * y_flat))
    A = np.stack(design, axis=1)
    coeffs, *_ = np.linalg.lstsq(A, z_flat, rcond=None)
    z_hat = A @ coeffs
    ss_res = np.sum((z_flat - z_hat) ** 2)
    ss_tot = np.sum((z_flat - z_flat.mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    return {"r2": r2, "n_params": len(coeffs), "z_hat": z_hat.reshape(img.shape)}


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_torch_mlp(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = xt.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred_test = model(torch.tensor(x_test, dtype=torch.float32)).argmax(dim=1).numpy()
        pred_train = model(xt).argmax(dim=1).numpy()

    return {
        "train_bacc": balanced_accuracy_score(y_train, pred_train),
        "test_bacc": balanced_accuracy_score(y_test, pred_test),
        "test_f1": f1_score(y_test, pred_test, average="macro"),
        "n_params": model.n_params,
    }


def gamma_grid(x: np.ndarray, n_grid: int = 11) -> np.ndarray:
    """Geometric grid of ``n_grid`` bandwidths centred on scikit-learn's ``scale``.

    ``gamma="scale"`` is ``1 / (n_features * X.var())``, so centring here and
    spanning three decades either side puts the library default at the middle of
    the grid. Two consequences matter. The grid contains the default as an
    interior point, so a selected gamma can only lose to it by validation noise;
    and because the centre is derived from the data rather than fixed, the grid
    tracks the input dimension. An absolute grid does not: at ``d = 784`` the
    optimum sits near ``0.00128``, an order of magnitude below the lowest point
    of the 0.01--1000 grid the kernel-bandwidth experiments use.
    """
    var = float(np.asarray(x).var())
    centre = 1.0 / (x.shape[1] * var) if x.shape[1] and var > 0 else 1.0
    return np.geomspace(centre * 1e-3, centre * 1e3, n_grid)


def rbf_svm_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tune: bool = True,
    n_grid: int = 11,
    seed: int = 0,
) -> dict:
    """RBF-SVC scored on a held-out test set, with gamma selected on validation.

    The bandwidth is chosen on a stratified quarter of the training draw and the
    winner is refit on all of it, so the test set never enters selection. Passing
    ``tune=False`` recovers the single-heuristic ``gamma="scale"`` fit; every
    table in the paper uses the tuned form, so that the classical reference gets
    a search budget comparable to the quantum models' bandwidth sweep.
    """
    if tune:
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
        )
        best_gamma, best_val = None, -np.inf
        for g in gamma_grid(x_train, n_grid):
            val = balanced_accuracy_score(
                y_val, SVC(kernel="rbf", gamma=g).fit(x_fit, y_fit).predict(x_val)
            )
            if val > best_val:
                best_gamma, best_val = float(g), float(val)
    else:
        best_gamma = "scale"

    clf = SVC(kernel="rbf", gamma=best_gamma)
    clf.fit(x_train, y_train)
    pred_test = clf.predict(x_test)
    pred_train = clf.predict(x_train)
    return {
        "train_bacc": balanced_accuracy_score(y_train, pred_train),
        "test_bacc": balanced_accuracy_score(y_test, pred_test),
        "test_f1": f1_score(y_test, pred_test, average="macro"),
        "n_support": int(clf.support_.shape[0]),
        "gamma": float(clf._gamma),
    }
