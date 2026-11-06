"""
rankdesign.py
─────────────
Turning the separability-rank analysis of :mod:`code.qnn` into an ansatz
*design* rule driven by training-free classical diagnostics.

``FourierSeries2DQNN`` offers a binary choice, a CNOT at every re-upload or
none at all, and the paper reads the resulting gap as evidence that
entanglement is necessary for non-separable 2-D structure. That contrast is
the two endpoints of a graded ladder. Placing an entangler in only ``k`` of
the ``L`` re-upload layers interpolates between them, and the reachable rank
of the 2-D Fourier coefficient matrix along that ladder is a computable
quantity that needs no training.

Two facts make the ladder useful as a design tool. First, the accessible
frequency support is exactly the ``(2L+1) x (2L+1)`` box for *every* choice of
entangler, including none, so depth alone fixes the bandwidth and the
entanglers move only the rank inside a box of fixed size. Bandwidth and rank
are therefore independent knobs rather than confounded ones, which is what
makes a comparison across entangling mechanisms meaningful in the first place.
Second, the reachable rank depends on the *number and placement* of
entanglers but not on which two-qubit gate is used, so a free CZ reaches
exactly what an arbitrary 15-parameter SU(4) reaches.

Measured over ``L`` from 4 to 24, the maximum reachable rank follows

    rank(k, L) = min(2L + 1, 4 * 3**(k-1))     for k >= 1,   and 1 for k = 0,

giving the ladder 1, 4, 12, 36 as entanglers are added. The law is verified
exactly for ``k <= 3``; see :func:`reachable_rank_law` for the caveat at
``k = 4``. Because ``4 * 3**(k-1)`` overtakes ``2L+1`` at ``k = 3`` for every
``L <= 17``, three well-placed entanglers already saturate everything a
realistic-depth circuit can express, which is why adding more buys nothing.

Rank is necessary but not sufficient, and two further facts decide whether the
ceiling it sets is actually attained. A rank-``r`` matrix in the ``R x R``
coefficient box carries ``r(2R - r)`` real degrees of freedom
(:func:`target_dof`) while the circuit carries only about ``6(L+1)`` parameters
(:func:`parameter_count`), so the ceiling is reached exactly when the budget
covers the target's degrees of freedom and is missed otherwise, however many
entanglers are present. And a *fixed* entangler such as a CNOT cannot be
switched off, so the ``k`` and ``k-1`` model classes are not nested and adding
an entangler can strictly lose accuracy on targets the smaller class fits
exactly. A parameterised entangler contains the identity at angle zero, which
restores nesting and makes the ladder monotone. That non-nesting is on its own
enough to produce the "more entanglement hurts" signature this project keeps
running into, without any appeal to optimisation difficulty.

The design rule that follows has three parts rather than one. The re-upload
depth is set by the signal's bandwidth ``omega_eff``, the entangler budget by
its SVD effective rank, and the parameter budget by its degrees of freedom, all
three read off training-free diagnostics that :mod:`code.main` and
:mod:`code.baselines` already compute. The entanglers should be parameterised,
spread across interior layers, and are otherwise free to be whichever two-qubit
gate the hardware prefers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pennylane as qml
import torch
from torch import nn

TWO_QUBIT_GATES = {
    "cz": lambda w, p: qml.CZ(wires=w),
    "cnot": lambda w, p: qml.CNOT(wires=w),
    "crz": lambda w, p: qml.CRZ(p, wires=w),
    "isingxx": lambda w, p: qml.IsingXX(p, wires=w),
}


# ─── The reachable-rank law and the design rule it implies ────────────────


def reachable_rank_law(k: int, degree: int) -> int:
    """Maximum rank of the 2-D Fourier coefficient matrix reachable by a
    2-qubit re-uploading circuit of depth ``degree`` carrying ``k`` entangling
    gates, over the best placement of those gates.

    Verified exactly against simulation for ``k <= 3`` at degrees 4, 6, 8, 16,
    20 and 24. The ``k >= 4`` values are an extrapolation of the same pattern
    and are *not* confirmed: a search over 78 random placements at degree 56
    reached rank 93 against the extrapolated 108. The gap does not affect the
    design rule, because ``k = 3`` already saturates the ``2*degree+1`` cap at
    every degree below 18, so ``k = 4`` is never the recommended budget in the
    regime this repository operates in.
    """
    if k < 0 or degree < 0:
        raise ValueError("k and degree must be non-negative")
    unconstrained = 1 if k == 0 else 4 * 3 ** (k - 1)
    return int(min(2 * degree + 1, unconstrained))


def min_entanglers_for_rank(target_rank: float, degree: int) -> int:
    """Smallest entangler budget ``k`` whose reachable rank covers
    ``target_rank`` at the given depth, or ``-1`` if the depth caps out first.

    This is the entangling half of the design rule: feed it the SVD effective
    rank of the signal (:func:`code.baselines.svd_effective_rank`) and it
    returns how many entangling gates the ansatz needs. The depth half is the
    separate requirement ``degree >= omega_eff``.
    """
    for k in range(degree + 1):
        if reachable_rank_law(k, degree) >= target_rank:
            return k
    return -1


def omega_eff_2d(img: np.ndarray, energy_threshold: float = 0.95) -> int:
    """Smallest ``W`` whose ``(2W+1)^2`` box around DC holds ``energy_threshold``
    of a 2-D signal's spectral energy.

    The two-dimensional counterpart of the ``omega_eff`` the paper computes on
    1-D signals, and the same quantity in the same units, because a depth-``L``
    circuit reaches every frequency ``|omega| <= L`` on *both* axes at once. The
    box is therefore the right shape to ask about: taking a radial band instead
    would measure something the circuit's support does not have.

    The mean is removed first, so the DC coefficient never contributes and the
    threshold is a fraction of the varying part rather than of the offset the
    model carries as a free bias anyway.
    """
    n = img.shape[0]
    power = np.abs(np.fft.fft2(img - img.mean())) ** 2
    freqs = np.abs(np.fft.fftfreq(n, d=1.0 / n).astype(int))
    box = np.maximum(freqs[:, None], freqs[None, :])
    total = power.sum() + 1e-12
    for w in range(n):
        if power[box <= w].sum() / total >= energy_threshold:
            return w
    return n - 1


def choose_ansatz(img: np.ndarray, energy_threshold: float = 0.95) -> dict:
    """The design rule, end to end: read an ansatz off a signal without training.

    Depth answers to two masters rather than one. It has to reach the signal's
    bandwidth, ``L >= omega_eff``, and it has to lift the rank cap ``2L+1`` above
    the signal's effective rank, since no entangler budget can exceed a ceiling
    that depth alone fixes. Taking the larger of the two is what keeps
    :func:`min_entanglers_for_rank` from being handed a depth at which the rank
    it is asked for is unreachable, which on real images happens often enough to
    matter: a smooth fundus photograph at an 8-pixel grid has ``omega_eff = 1``
    and effective rank 3.1, and a depth-1 circuit caps at rank 3.

    With depth settled the entangler budget follows from the effective rank, and
    the placement from :func:`recommended_placement`. Everything here is read off
    the target by SVD and FFT, so the cost is negligible beside a single fit and
    nothing is trained to make the choice.

    Returns the chosen ``degree``, ``n_entanglers`` and ``ent_layers`` alongside
    the diagnostics that produced them and the ``dof_sufficient`` flag, which
    warns when the parameter budget cannot cover the target's degrees of freedom
    and the rank ceiling is therefore not the binding constraint.
    """
    from code.baselines import svd_effective_rank

    omega = omega_eff_2d(img, energy_threshold)
    eff_rank = svd_effective_rank(img)
    # 2*L+1 >= eff_rank is the depth the rank cap demands, on top of bandwidth.
    depth_for_rank = int(np.ceil((eff_rank - 1) / 2))
    degree = max(int(omega), depth_for_rank, 1)
    k = min_entanglers_for_rank(eff_rank, degree)
    return {
        "degree": degree,
        "n_entanglers": k,
        "ent_layers": recommended_placement(k, degree),
        "omega_eff": int(omega),
        "effective_rank": float(eff_rank),
        "depth_from_bandwidth": int(omega),
        "depth_from_rank": depth_for_rank,
        "dof_sufficient": rank_ceiling_is_attainable(
            int(round(eff_rank)), degree, k
        ),
    }


def fitted_model_rank(k: int, degree: int) -> int:
    """Rank reachable by the *fitted model* rather than by the bare correlator.

    :class:`EntanglerLadder2DQNN` returns ``scale * <Z0 Z1> + bias``, and on a
    coordinate grid the bias contributes an all-ones matrix of rank one, so the
    model reaches one rank above :func:`reachable_rank_law` (clipped at the box
    size). At ``k = 0`` this is the separable-plus-constant class of rank 2 that
    :func:`code.baselines.separable_plus_const_r2` already bounds, which is why
    that function rather than a plain rank-1 SVD is the right reference there.

    Using the bare law as a ceiling instead of this understates the model class
    by exactly one rank, which shows up as apparent ceiling violations at
    ``k = 0`` of order 0.001-0.005 in R^2.
    """
    return int(min(reachable_rank_law(k, degree) + 1, 2 * degree + 1))


def target_dof(rank: int, degree: int) -> int:
    """Real degrees of freedom of a rank-``rank`` matrix in the ``(2*degree+1)``
    coefficient box, ``r(2R - r)``.

    This is the quantity the parameter budget has to cover. The reachable-rank
    law says whether a target is *representable*, this says whether it is
    representable by a circuit of the size actually being trained. Where
    ``target_dof`` exceeds :func:`parameter_count` the rank ceiling is not
    attainable and reporting the shortfall as an entanglement effect would be a
    mistake.
    """
    box = 2 * degree + 1
    return int(rank * (2 * box - rank))


def parameter_count(degree: int, k: int = 0) -> int:
    """Trainable parameters of :class:`EntanglerLadder2DQNN`: three rotation
    angles on each of two wires per layer, plus scale and bias, plus one angle
    per parameterised entangler."""
    return 6 * (degree + 1) + 2 + k


def rank_ceiling_is_attainable(rank: int, degree: int, k: int) -> bool:
    """Whether the parameter budget can in principle cover a rank-``rank``
    target, i.e. whether it is worth reading a shortfall as an entangler issue.
    """
    return parameter_count(degree, k) >= target_dof(rank, degree)


def recommended_placement(k: int, degree: int) -> tuple[int, ...]:
    """Where to put ``k`` entanglers among ``degree`` re-upload layers.

    Placement matters at fixed ``k``: at degree 6 with ``k = 2`` an adjacent
    pair ``(0, 1)`` reaches rank 8 while a spread pair ``(1, 3)`` reaches the
    full cap of 12. Layer 0 is systematically the weakest slot, since only one
    upload has happened when its entangler fires. We therefore spread the
    entanglers evenly across the interior layers.
    """
    if k <= 0:
        return ()
    if k >= degree:
        return tuple(range(degree))
    interior = list(range(1, degree)) or [0]
    idx = np.linspace(0, len(interior) - 1, k)
    return tuple(sorted({interior[int(round(i))] for i in idx}))


def best_placement(k: int, degree: int, seeds: int = 3, n_grid: int | None = None,
                   max_patterns: int = 200, rng: np.random.Generator | None = None
                   ) -> tuple[tuple[int, ...], int]:
    """Search entangler placements and return the best ``(pattern, rank)``.

    Exhaustive when there are at most ``max_patterns`` combinations, random
    otherwise. Used to check :func:`recommended_placement` rather than in the
    training path, since it costs a few seconds per candidate.
    """
    rng = rng or np.random.default_rng(0)
    patterns = list(itertools.combinations(range(degree), k))
    if len(patterns) > max_patterns:
        pick = rng.choice(len(patterns), size=max_patterns, replace=False)
        patterns = [patterns[i] for i in pick]
    best, arg = -1, ()
    for pattern in patterns:
        r = measure_reachable_rank(degree, pattern, seeds=seeds, n_grid=n_grid, rng=rng)
        if r > best:
            best, arg = r, pattern
    return arg, best


# ─── Training-free probes ─────────────────────────────────────────────────


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(n) * 2 * np.pi / n
    return np.meshgrid(axis, axis, indexing="ij")


def _random_entangler_angles(rng: np.random.Generator, degree: int) -> np.ndarray:
    """Angles for parameterised entanglers, uniform on the full circle.

    Probing the *reachable* set means searching the entangler's own parameter
    as well as the rotations. A Gaussian draw concentrates near zero, where a
    CRZ or IsingXX is close to the identity and reaches a lower rank than it
    can, which is the same small-angle effect that makes a freshly initialised
    entangling circuit look separable.
    """
    return rng.uniform(0.0, 2 * np.pi, size=max(degree, 1))


def circuit_response(degree: int, ent_layers, rot_w: np.ndarray,
                     gate: str = "cnot", ent_w: np.ndarray | None = None,
                     n_grid: int | None = None) -> np.ndarray:
    """Evaluate ``<Z0 Z1>`` on an ``n_grid x n_grid`` grid over ``[0, 2*pi)^2``.

    The grid must resolve the circuit's own band-limit, so it defaults to
    comfortably more than the ``2*degree+1`` frequencies present.
    """
    n_grid = n_grid or max(32, 4 * degree + 8)
    ent_layers = set(ent_layers)
    dev = qml.device("default.qubit", wires=2)
    apply_gate = TWO_QUBIT_GATES[gate]

    @qml.qnode(dev)
    def circuit(x, y):
        qml.Rot(*rot_w[0, 0], wires=0)
        qml.Rot(*rot_w[0, 1], wires=1)
        for layer in range(degree):
            qml.RZ(x, wires=0)
            qml.RZ(y, wires=1)
            if layer in ent_layers:
                apply_gate([0, 1], None if ent_w is None else ent_w[layer])
            qml.Rot(*rot_w[layer + 1, 0], wires=0)
            qml.Rot(*rot_w[layer + 1, 1], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    X, Y = _grid(n_grid)
    return np.asarray(circuit(X.ravel(), Y.ravel())).reshape(n_grid, n_grid).real


def measure_reachable_rank(degree: int, ent_layers, gate: str = "cnot",
                           seeds: int = 3, n_grid: int | None = None,
                           rng: np.random.Generator | None = None,
                           rtol: float = 1e-7) -> int:
    """Numerical rank of the circuit's response matrix, maximised over random
    parameter draws.

    Because the 2-D DFT is unitary, the rank of the response sampled on a grid
    equals the rank of its Fourier coefficient matrix, so this measures the
    quantity the separability argument is about without touching a Fourier
    transform. The ranks are hard rather than soft: the singular values fall
    from order one to machine epsilon with nothing in between.
    """
    rng = rng or np.random.default_rng(0)
    best = 0
    for _ in range(seeds):
        rot_w = rng.standard_normal((degree + 1, 2, 3))
        ent_w = _random_entangler_angles(rng, degree)
        f = circuit_response(degree, ent_layers, rot_w, gate, ent_w, n_grid)
        s = np.linalg.svd(f, compute_uv=False)
        best = max(best, int(np.sum(s > rtol * s[0])))
    return best


def measure_frequency_support(degree: int, ent_layers, gate: str = "cnot",
                              seeds: int = 3, n_grid: int | None = None,
                              rng: np.random.Generator | None = None,
                              tol: float = 1e-8) -> dict:
    """Accessible 2-D Fourier support, to confirm that the entangler moves rank
    without moving bandwidth.

    Returns the number of non-negligible coefficients and the largest accessible
    frequency on each axis. For every entangler choice this should come back as
    ``(2*degree+1)**2`` coefficients spanning ``+/- degree`` on both axes, which
    is what licenses comparing entangling mechanisms at matched depth.
    """
    rng = rng or np.random.default_rng(0)
    n_grid = n_grid or max(32, 4 * degree + 8)
    acc = np.zeros((n_grid, n_grid))
    for _ in range(seeds):
        rot_w = rng.standard_normal((degree + 1, 2, 3))
        ent_w = _random_entangler_angles(rng, degree)
        f = circuit_response(degree, ent_layers, rot_w, gate, ent_w, n_grid)
        acc = np.maximum(acc, np.abs(np.fft.fft2(f) / f.size))
    freqs = np.abs(np.fft.fftfreq(n_grid, d=1.0 / n_grid).astype(int))
    mask = acc > tol
    return {
        "n_coefficients": int(mask.sum()),
        "omega_x_max": int((freqs[:, None] * mask).max()),
        "omega_y_max": int((freqs[None, :] * mask).max()),
    }


# ─── Controlled-rank synthetic targets ────────────────────────────────────


def bandlimited_basis(bandwidth: int, n_grid: int) -> np.ndarray:
    """Orthonormal basis of the band-limited subspace, as an
    ``n_grid x (2*bandwidth+1)`` matrix.

    The real trigonometric functions ``1, cos(mx), sin(mx)`` are mutually
    orthogonal when sampled on a uniform grid with ``n_grid > 2*bandwidth``, so
    normalising each column is all that is needed. Having an orthonormal basis
    is what lets :func:`controlled_rank_target` prescribe a singular spectrum
    exactly rather than inherit whatever the random draw produces.
    """
    if n_grid <= 2 * bandwidth:
        raise ValueError(
            f"n_grid {n_grid} must exceed 2*bandwidth = {2 * bandwidth} for the "
            f"trigonometric basis to stay orthogonal on the grid"
        )
    axis = np.arange(n_grid) * 2 * np.pi / n_grid
    cols = [np.ones_like(axis)]
    for m in range(1, bandwidth + 1):
        cols.append(np.cos(m * axis))
        cols.append(np.sin(m * axis))
    basis = np.stack(cols, axis=1)
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def controlled_rank_target(rank: int, bandwidth: int, n_grid: int = 16,
                           rng: np.random.Generator | None = None,
                           spectrum: str = "flat") -> np.ndarray:
    """A real 2-D signal with exact rank ``rank`` and Fourier support confined
    to the ``(2*bandwidth+1)`` box on each axis.

    Real images confound rank with bandwidth, smoothness and noise. This
    generator varies rank alone while holding the frequency support fixed at
    the circuit's own, so a fit that fails can only have failed on rank.

    The target is built as ``U diag(s) V^T`` with ``U`` and ``V`` random
    orthonormal frames drawn inside the band-limited subspace
    (:func:`bandlimited_basis`), so the singular values are exactly ``s`` and
    the algebraic rank is exactly ``rank``.

    ``spectrum`` decides how demanding the target is:

    ``"flat"``
        All singular values equal, so the SVD *effective* rank equals the
        algebraic rank. This is the default because effective rank is the
        diagnostic the rest of this work uses, and it is the only setting under
        which a rank-r target genuinely exercises rank r.
    ``"decay"``
        Singular values falling as ``1/(1+i)``, closer to a natural image but
        much easier than its nominal rank suggests. At bandwidth 4 an algebraic
        rank of 9 gives an effective rank of only 4.31, which flatters every
        low-budget ansatz and compresses the ladder. Kept for reference.

    Rank is capped at ``2*bandwidth+1``, the dimension of the band-limited
    subspace.
    """
    box = 2 * bandwidth + 1
    if rank > box:
        raise ValueError(
            f"rank {rank} exceeds the {box} dimensions available "
            f"at bandwidth {bandwidth}"
        )
    if spectrum not in ("flat", "decay"):
        raise ValueError(f"unknown spectrum {spectrum!r}; use 'flat' or 'decay'")
    rng = rng or np.random.default_rng(0)
    basis = bandlimited_basis(bandwidth, n_grid)

    def orthonormal_frame() -> np.ndarray:
        """A random ``rank``-dimensional orthonormal frame within the basis."""
        q, _ = np.linalg.qr(rng.standard_normal((box, rank)))
        return basis @ q

    s = (np.ones(rank) if spectrum == "flat"
         else 1.0 / (1.0 + np.arange(rank, dtype=float)))
    out = (orthonormal_frame() * s) @ orthonormal_frame().T
    # Scale but do not centre. Subtracting the mean would add a rank-one
    # constant matrix and silently make the target rank ``rank + 1``.
    return out / (out.std() + 1e-12)


def model_ceiling_r2(target: np.ndarray, k: int, degree: int) -> float:
    """Best R^2 attainable by the budget-``k`` model class on ``target``.

    The two budgets need different references, and using one for both is the
    mistake this repository has already made once. At ``k = 0`` the model is
    ``s g(x) h(y) + b``, a rank-one matrix plus a constant, which is a strict
    subset of the rank-two matrices. Its exact ceiling is the best
    separable-plus-constant fit (:func:`code.baselines.separable_plus_const_r2`)
    and a rank-two truncated SVD is far too generous, scoring 1.0 against an
    attainable 0.5 on a flat-spectrum rank-two target.

    At ``k >= 1`` no such closed form is available, so we fall back to the
    truncated SVD at the model's rank. That is a valid upper bound, since the
    fitted response really does have at most that rank, but a loose one: the
    reachable set is a low-dimensional manifold inside the rank-``r`` matrices,
    not all of them, and :func:`target_dof` measures how loose.
    """
    from code.baselines import separable_plus_const_r2

    if k == 0:
        return separable_plus_const_r2(target)
    return truncated_svd_r2(target, fitted_model_rank(k, degree))


def truncated_svd_r2(target: np.ndarray, rank: int) -> float:
    """R^2 of the best rank-``rank`` approximation to ``target``.

    This is the exact ceiling for any model whose reachable coefficient matrix
    has that rank, and so is the reference the fitted circuits are judged
    against. Unlike the separable-plus-constant bound used for the product
    ansatz it needs no special-casing, because the graded ladder makes the
    model's rank an explicit design parameter.
    """
    u, s, vt = np.linalg.svd(target, full_matrices=False)
    recon = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    ss_res = float(np.sum((target - recon) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2)) + 1e-12
    return float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))


# ─── The graded ansatz ────────────────────────────────────────────────────


class EntanglerLadder2DQNN(nn.Module):
    """``FourierSeries2DQNN`` with the entangler budget exposed as a parameter.

    ``ent_layers`` lists which of the ``degree`` re-upload layers carry a
    two-qubit gate. The empty set recovers the product ansatz and the full set
    recovers the fully entangled one, so the paper's original pair sits at the
    two ends of this ladder. Every point on it has the same parameter count
    apart from the trainable entanglers, and, more importantly, the same
    accessible frequency support, so a comparison along the ladder isolates
    rank.

    The default gate is ``crz`` rather than the ``cnot`` of the original model.
    A CRZ at angle zero is the identity, so each rung of the ladder contains the
    one below it and accuracy is monotone in the entangler budget. With a fixed
    CNOT the rungs are disjoint, and on a target the product ansatz fits exactly
    the k=1 circuit measures 0.83 against 1.00 purely because it cannot switch
    its entangler off. Pass ``gate="cnot"`` to reproduce the original
    non-nested behaviour.

    Unlike ``FourierSeries2DQNN`` the forward pass is broadcast over all input
    points in a single circuit evaluation, which is what makes sweeping the
    ladder affordable.
    """

    def __init__(self, degree: int, ent_layers=None, gate: str = "crz",
                 device_name: str = "default.qubit"):
        super().__init__()
        if gate not in TWO_QUBIT_GATES:
            raise ValueError(f"unknown gate {gate!r}; choose from {sorted(TWO_QUBIT_GATES)}")
        self.degree = degree
        self.gate = gate
        self.ent_layers = set(range(degree) if ent_layers is None else ent_layers)
        self.rot_weights = nn.Parameter(torch.randn(degree + 1, 2, 3, dtype=torch.float64))
        self.ent_weights = nn.Parameter(
            torch.randn(max(degree, 1), dtype=torch.float64)
        )
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        dev = qml.device(device_name, wires=2)
        ent_layers_set = self.ent_layers
        apply_gate = TWO_QUBIT_GATES[gate]

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, y, rot_w, ent_w):
            qml.Rot(*rot_w[0, 0], wires=0)
            qml.Rot(*rot_w[0, 1], wires=1)
            for layer in range(degree):
                qml.RZ(x, wires=0)
                qml.RZ(y, wires=1)
                if layer in ent_layers_set:
                    apply_gate([0, 1], ent_w[layer])
                qml.Rot(*rot_w[layer + 1, 0], wires=0)
                qml.Rot(*rot_w[layer + 1, 1], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self._circuit = circuit

    @property
    def n_entanglers(self) -> int:
        return len(self.ent_layers)

    @property
    def reachable_rank(self) -> int:
        """The law's prediction for this configuration."""
        return reachable_rank_law(self.n_entanglers, self.degree)

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        raw = self._circuit(xs, ys, self.rot_weights, self.ent_weights)
        return self.scale * raw + self.bias


# ─── Towards n qubits: rank across a bipartition ──────────────────────────
#
# With more than two coordinates the coefficient object is an order-n tensor and
# "the rank" is no longer well defined. The generalisation that matches the
# separability argument is the rank of the tensor matricised across a
# bipartition, which is the functional Schmidt rank across that cut and reduces
# to the 2-qubit coefficient matrix when n = 2.
#
# What we measure (3 qubits, <Z0 Z1 Z2>, depths 4 and 6) is that cut-locality is
# exact and the ladder transfers only as a lower bound. See
# :func:`crossing_count` for the caveat that makes the difference.


def circuit_response_nd(n_wires: int, degree: int, entanglers, rot_w: np.ndarray,
                        n_grid: int, gate: str = "cnot") -> np.ndarray:
    """Evaluate the full-weight correlator on an ``n_wires``-dimensional grid.

    ``entanglers`` is a sequence of ``(layer, (wire_a, wire_b))`` placements, so
    both which layers entangle and which pairs they act on are explicit. The
    returned array has shape ``(n_grid,) * n_wires``.
    """
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for layer, edge in entanglers:
        by_layer.setdefault(layer, []).append(tuple(edge))
    dev = qml.device("default.qubit", wires=n_wires)
    apply_gate = TWO_QUBIT_GATES[gate]

    @qml.qnode(dev)
    def circuit(*coords):
        for w in range(n_wires):
            qml.Rot(*rot_w[0, w], wires=w)
        for layer in range(degree):
            for w in range(n_wires):
                qml.RZ(coords[w], wires=w)
            for edge in by_layer.get(layer, []):
                apply_gate(list(edge), None)
            for w in range(n_wires):
                qml.Rot(*rot_w[layer + 1, w], wires=w)
        obs = qml.PauliZ(0)
        for w in range(1, n_wires):
            obs = obs @ qml.PauliZ(w)
        return qml.expval(obs)

    axis = np.arange(n_grid) * 2 * np.pi / n_grid
    mesh = np.meshgrid(*([axis] * n_wires), indexing="ij")
    flat = circuit(*[m.ravel() for m in mesh])
    return np.asarray(flat).reshape(*([n_grid] * n_wires)).real


def cut_rank(tensor: np.ndarray, subset, rtol: float = 1e-7) -> int:
    """Rank of ``tensor`` matricised with the modes in ``subset`` as rows.

    This is the functional Schmidt rank across the bipartition, the n-qubit
    stand-in for the coefficient-matrix rank of the two-qubit case.
    """
    n = tensor.ndim
    subset = list(subset)
    rest = [m for m in range(n) if m not in subset]
    size = tensor.shape[0]
    mat = np.transpose(tensor, subset + rest).reshape(
        size ** len(subset), size ** len(rest)
    )
    s = np.linalg.svd(mat, compute_uv=False)
    return int(np.sum(s > rtol * s[0]))


def crossing_count(entanglers, subset) -> int:
    """Number of entanglers whose two wires straddle the bipartition.

    Feeding this into :func:`reachable_rank_law` gives a *lower* bound on the
    cut's rank rather than its value. It is exact in most placements we tested,
    including the discriminating case at degree 6 where it correctly returns 12
    rather than the cap of 13. It undercounts when a non-crossing entangler sits
    *between* two crossing ones, because that gate scrambles the far side of the
    cut and the later crossing gate then carries a richer operator across it.
    Holding the crossing gates at layers 1 and 5, an interleaved ``(1,2)`` gate
    at layer 3 lifts the cut from 12 to 13 while the same gate placed at layer 0,
    before both, leaves it at 12. The effect cannot arise at ``n = 2``, where
    every entangler crosses the only cut, which is why the two-qubit law does not
    lift to n qubits unchanged.
    """
    subset = set(subset)
    return sum(1 for _, (a, b) in entanglers if (a in subset) != (b in subset))


def graph_cut_rank(entanglers, subset, n_wires: int) -> int:
    """Rank of the entangling graph's off-diagonal block across ``subset``.

    Build the 0/1 matrix whose rows are the wires in ``subset``, whose columns
    are the wires outside it, and whose entries mark which pairs carry an
    entangler. Its rank counts the independent channels the graph opens across
    the cut, as distinct from :func:`crossing_count`, which counts gates.

    The quantity is borrowed rather than derived. For quadratic phase states
    over a finite field the Rényi-2 entropy across a cut is exactly the rank
    of the corresponding off-diagonal block of the defining symmetric matrix,
    with the diagonal dropping out as local Clifford content (Dahbi and Zaïr,
    arXiv:2605.05167). That setting is discrete and stabiliser where ours is
    continuous, so the identity does not transfer, but the shape of the claim is
    testable here and :func:`bipartition_rank_law` already behaves as if it
    holds. Gates piled on one edge share a cell and give rank 1, edges sharing a
    qubit share a row or a column and also give rank 1, and disjoint edges give
    rank 2.

    Multiplicities are ignored, so this says how many channels are open and not
    how far each has climbed its own ladder.
    """
    subset = sorted(set(subset))
    rest = [w for w in range(n_wires) if w not in subset]
    row = {w: i for i, w in enumerate(subset)}
    col = {w: j for j, w in enumerate(rest)}
    block = np.zeros((len(subset), len(rest)))
    for _, (a, b) in entanglers:
        if a in row and b in col:
            block[row[a], col[b]] = 1.0
        elif b in row and a in col:
            block[row[b], col[a]] = 1.0
    if not block.any():
        return 0
    return int(np.linalg.matrix_rank(block))


def bipartition_rank_law(entanglers, subset, degree: int, n_wires: int) -> int | None:
    """Predicted rank across a bipartition when the crossing edges are pairwise
    disjoint, and ``None`` when they are not.

    The crossing *count* is not the quantity that governs a multipartition. At
    ``n = 4`` across the balanced ``{0,1}|{2,3}`` cut, two crossing entanglers
    reach rank 12 when both sit on the edge ``(0,2)``, 13 when they share a
    single qubit as ``(0,2)`` and ``(0,3)``, and 16 when they are disjoint as
    ``(0,2)`` and ``(1,3)``. Feeding "two crossings" into
    :func:`reachable_rank_law` returns one number for all three. The effect
    cannot arise at ``n = 2``, where there is only one edge to place a gate on,
    which is why the two-qubit law does not lift unchanged.

    Disjoint crossing edges behave as independent channels and their ranks
    multiply, each edge contributing the two-qubit ladder value for the number
    of gates placed on it, with the whole product capped at ``(2L+1)`` raised to
    the number of coordinates on the smaller side. Verified exactly at
    ``degree = 4``, where two gates on each of two disjoint edges reach the full
    ``9 * 9 = 81``. At ``degree = 6`` the same construction predicts 144 and the
    search reaches 142, climbing with the number of parameter draws, so the law
    is supported but not confirmed once the rank gets large.

    Edges that share a qubit form a star rather than a second channel, and the
    star climbs the ladder one rung faster than its gate count alone would say.
    Enumerating every crossing-edge multiset up to three gates on a ``2|2`` cut
    at degree 6 (:mod:`experiments.run_cut_graph_enumeration`) fixes both the
    rule and its scope. Writing each connected component of the crossing-edge
    graph as one channel carrying ``g`` gates over ``e`` distinct edges, the
    channel reaches ``reachable_rank_law(g + e - 1, degree)``, and the channels
    multiply. A single edge has ``e = 1`` and recovers the plain ladder, so
    nothing about the pairwise-disjoint case changes.

    The rule is exact on all 30 configurations whose crossing edges form a
    disjoint union of stars, which is precisely the condition that the number of
    connected components equals the matching number, and it is badly wrong on
    the 4 that do not. A path such as ``(0,2),(0,3),(1,2)`` is connected yet
    contains a perfect matching, and it measures 47 against a prediction of 13.
    Those return ``None``, as the shared-qubit case used to.
    """
    subset = set(subset)
    crossing = [tuple(sorted(edge)) for _, edge in entanglers
                if (edge[0] in subset) != (edge[1] in subset)]
    if not crossing:
        return 1

    parent = {w: w for edge in crossing for w in edge}

    def find(w: int) -> int:
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    for a, b in crossing:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    gates: dict[int, int] = {}
    edges: dict[int, set] = {}
    for edge in crossing:
        root = find(edge[0])
        gates[root] = gates.get(root, 0) + 1
        edges.setdefault(root, set()).add(edge)

    # A component is a star exactly when it has no two disjoint edges, and the
    # multiplicative law holds only for a disjoint union of stars.
    for root, comp_edges in edges.items():
        for e1, e2 in itertools.combinations(sorted(comp_edges), 2):
            if not (set(e1) & set(e2)):
                return None

    product = 1
    for root, count in gates.items():
        product *= reachable_rank_law(count + len(edges[root]) - 1, degree)
    n_side = min(len(subset), n_wires - len(subset))
    return int(min(product, (2 * degree + 1) ** n_side))


def measure_cut_rank(n_wires: int, degree: int, entanglers, subset,
                     seeds: int = 4, n_grid: int | None = None,
                     gate: str = "cnot",
                     rng: np.random.Generator | None = None) -> int:
    """Reachable rank across one bipartition, maximised over parameter draws."""
    rng = rng or np.random.default_rng(0)
    n_grid = n_grid or max(12, 2 * degree + 4)
    best = 0
    for _ in range(seeds):
        rot_w = rng.standard_normal((degree + 1, n_wires, 3))
        response = circuit_response_nd(n_wires, degree, entanglers, rot_w, n_grid, gate)
        best = max(best, cut_rank(response, subset))
    return best


# ─── Product read-out vs entangling ansatz across a cut ────────────────────
#
# The single-observable rank above answers "what does the fitted correlator
# reach". The read-out *model* is instead a linear combination of several
# observables, f = sum_i w_i <O_i>, so the object to rank is the whole family
# {<O_i>}, not one member. On a product state every <O_i> is a product of the
# single-coordinate marginals, so across a single-qubit cut A|B the only
# functions of the A-coordinate available are 1 and that qubit's own marginal:
# the family's rank across the cut is at most 2 no matter how high the
# correlator order climbs. An entangling gate makes <Z_A> itself a rich
# function of the A-coordinate and lifts the cut. This is the 3-qubit form of
# the rank-2 ceiling of eq. (separable-class); these two helpers measure it.


def _z_word(wires) -> "qml.operation.Observable":
    """The Pauli-Z product observable on ``wires`` (a tuple of wire indices)."""
    obs = qml.PauliZ(wires[0])
    for w in wires[1:]:
        obs = obs @ qml.PauliZ(w)
    return obs


def readout_responses(n_wires: int, degree: int, entanglers, rot_w: np.ndarray,
                      angle_of: dict, observables, n_grid: int,
                      gate: str = "crz") -> list[np.ndarray]:
    """Evaluate every observable in ``observables`` on the ``n_wires``-D grid for
    one parameter draw.

    ``observables`` is a sequence of wire-index tuples, e.g. ``(0,)`` for
    ``<Z_0>``, ``(0, 1)`` for the pairwise correlator and ``(0, 1, 2)`` for the
    triple. ``entanglers`` is a sequence of ``(layer, (a, b))`` placements and
    ``angle_of`` maps ``(layer, (a, b))`` to a rotation angle for parameterised
    gates (ignored for the parameter-free CZ/CNOT). Returns one
    ``(n_grid,) * n_wires`` array per observable, sharing the single circuit
    evaluation.
    """
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for layer, edge in entanglers:
        by_layer.setdefault(layer, []).append(tuple(edge))
    dev = qml.device("default.qubit", wires=n_wires)
    apply_gate = TWO_QUBIT_GATES[gate]
    param_free = gate in ("cz", "cnot")

    @qml.qnode(dev)
    def circuit(*coords):
        for w in range(n_wires):
            qml.Rot(*rot_w[0, w], wires=w)
        for layer in range(degree):
            for w in range(n_wires):
                qml.RZ(coords[w], wires=w)
            for edge in by_layer.get(layer, []):
                p = None if param_free else angle_of[(layer, edge)]
                apply_gate(list(edge), p)
            for w in range(n_wires):
                qml.Rot(*rot_w[layer + 1, w], wires=w)
        return [qml.expval(_z_word(o)) for o in observables]

    axis = np.arange(n_grid) * 2 * np.pi / n_grid
    mesh = np.meshgrid(*([axis] * n_wires), indexing="ij")
    flat = circuit(*[m.ravel() for m in mesh])
    return [np.asarray(f).reshape(*([n_grid] * n_wires)).real for f in flat]


def readout_span_cut_rank(n_wires: int, degree: int, entanglers, observables,
                          subset, seeds: int = 4, n_grid: int | None = None,
                          gate: str = "crz",
                          rng: np.random.Generator | None = None,
                          rtol: float = 1e-7) -> int:
    """Rank of the read-out family across the ``subset`` bipartition, maximised
    over random parameter draws.

    The reachable model is ``sum_i w_i <O_i>``, so the quantity that bounds it
    across the cut is the number of independent functions of the ``subset``
    coordinates the family can produce. We stack every observable's response,
    matricised with the ``subset`` modes as rows, side by side and take the rank
    of the result. For a product ansatz this comes back at 2 for every
    single-qubit cut whatever the observable order; an entangling ansatz lifts
    it toward the ``2*degree+1`` cap.
    """
    rng = rng or np.random.default_rng(0)
    n_grid = n_grid or max(12, 2 * degree + 4)
    subset = list(subset)
    rest = [w for w in range(n_wires) if w not in subset]
    param_free = gate in ("cz", "cnot")
    best = 0
    for _ in range(seeds):
        rot_w = rng.standard_normal((degree + 1, n_wires, 3))
        angle_of = {} if param_free else {
            (layer, tuple(edge)): float(rng.uniform(0.0, 2 * np.pi))
            for layer, edge in entanglers
        }
        resp = readout_responses(n_wires, degree, entanglers, rot_w, angle_of,
                                 observables, n_grid, gate)
        cols = [
            np.transpose(t, subset + rest).reshape(
                n_grid ** len(subset), n_grid ** len(rest))
            for t in resp
        ]
        phi = np.concatenate(cols, axis=1)
        s = np.linalg.svd(phi, compute_uv=False)
        r = int(np.sum(s > rtol * s[0])) if s.size else 0
        best = max(best, r)
    return best


class EntanglerReadoutClassifier(nn.Module):
    """Trainable classifier that mirrors :func:`readout_span_cut_rank`'s circuit,
    so its accuracy is the fitted companion to that reachable-rank measurement.

    One coordinate per wire, data re-uploading at depth ``degree``, an entangler
    ring on the layers named in ``entanglers`` and a linear head over the
    ``observables`` read-out family. The empty ``entanglers`` list is the product
    ansatz and the read-out ladder ``local -> +ZZ -> +ZZZ`` is chosen through
    ``observables``, exactly the two axes the rank study grades. Because the
    accessible bandwidth is the same for every entangler choice, all
    configurations share one input scaling and the comparison isolates rank.

    ``forward`` takes an already-scaled ``(batch, n_wires)`` tensor and returns
    class logits; the batch is broadcast through a single circuit evaluation.
    """

    def __init__(self, degree: int, entanglers, observables, n_wires: int = 3,
                 n_classes: int = 2, gate: str = "crz",
                 device_name: str = "default.qubit"):
        super().__init__()
        if gate not in TWO_QUBIT_GATES:
            raise ValueError(f"unknown gate {gate!r}")
        self.degree = degree
        self.entanglers = [(int(l), tuple(e)) for l, e in entanglers]
        self.observables = [tuple(o) for o in observables]
        self.n_wires = n_wires
        self.rot_weights = nn.Parameter(
            0.1 * torch.randn(degree + 1, n_wires, 3, dtype=torch.float64)
        )
        self.ent_weights = nn.Parameter(
            0.1 * torch.randn(max(len(self.entanglers), 1), dtype=torch.float64)
        )
        self.head = nn.Linear(len(self.observables), n_classes, dtype=torch.float64)

        apply_gate = TWO_QUBIT_GATES[gate]
        param_free = gate in ("cz", "cnot")
        by_layer: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for idx, (layer, edge) in enumerate(self.entanglers):
            by_layer.setdefault(layer, []).append((idx, edge))
        observables = self.observables
        dev = qml.device(device_name, wires=n_wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, rot_w, ent_w):
            for w in range(n_wires):
                qml.Rot(*rot_w[0, w], wires=w)
            for layer in range(degree):
                for w in range(n_wires):
                    qml.RZ(x[:, w], wires=w)
                for idx, edge in by_layer.get(layer, []):
                    apply_gate(list(edge), None if param_free else ent_w[idx])
                for w in range(n_wires):
                    qml.Rot(*rot_w[layer + 1, w], wires=w)
            return [qml.expval(_z_word(o)) for o in observables]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack(self._circuit(x, self.rot_weights, self.ent_weights), dim=1)
        return self.head(feats)
