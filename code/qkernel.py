"""
qkernel.py
──────────
Pauli feature-map quantum kernels of graded entangling capacity, used to
study the genuinely 3-D ``2c`` checkerboard dataset. This follows the
encoding conventions of the companion ``kernel-trainer`` project
(Havlicek et al., 2019 style feature maps; see
``kernel_trainer.kernels.quantum.pennylane_pauli_kernel``): single-qubit
rotation angle ``2*(pi - x_i)`` and two-body coupling angle
``2*(pi - x_i)*(pi - x_j)``. We reconstruct the feature map here (rather
than call the library function directly) only so that the initial
Hadamard superposition layer is applied in every case — the library
helper adds Hadamards only for its default single-Pauli ``['Z']`` map, so
a bare ``Z+ZZ`` map there is diagonal and trivial.

Entangling capacity is graded by the Pauli set and the two-body coupling
topology:

  * ``['Z']``              — product-state ZFeatureMap, no entanglement.
  * ``['Z','ZZ']`` linear  — nearest-neighbour ZZ couplings.
  * ``['Z','ZZ']`` full    — all-pairs ZZ couplings.

and quantified with the Meyer-Wallach measure (Sim et al., 2019), the same
entangling-capacity metric ``kernel_trainer.metrics.EntanglingCapacity``
computes.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pennylane as qml


def _two_body_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entanglement == "full":
        return list(combinations(range(n_qubits), 2))
    raise ValueError(f"unknown entanglement pattern: {entanglement}")


def pauli_feature_map(x, n_qubits: int, paulis: list[str], entanglement: str) -> None:
    """Havlicek-style Pauli feature map with an explicit Hadamard layer."""
    for i in range(n_qubits):
        qml.Hadamard(i)
        qml.RZ(2 * (np.pi - x[i]), wires=i)
    if "ZZ" in paulis:
        for i, j in _two_body_pairs(n_qubits, entanglement):
            qml.MultiRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])


def make_fidelity_kernel(n_qubits: int, paulis: list[str], entanglement: str, device_name="default.qubit"):
    """Return a scalar fidelity kernel k(x1, x2) = |<0|U(x2)^dag U(x1)|0>|^2."""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x1, x2):
        pauli_feature_map(x1, n_qubits, paulis, entanglement)
        qml.adjoint(pauli_feature_map)(x2, n_qubits, paulis, entanglement)
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2) -> float:
        return float(circuit(x1, x2)[0])

    return kernel


def kernel_matrix(kernel, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Dense kernel matrix; exploits symmetry when B is None (Gram matrix).

    Reference implementation: O(N^2) circuit evaluations. ``fast_kernel_matrix``
    computes the same numbers analytically and is what the experiments use.
    """
    if B is None:
        n = len(A)
        K = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = kernel(A[i], A[j])
        return K
    return np.array([[kernel(a, b) for b in B] for a in A])


def feature_states(
    X: np.ndarray,
    n_qubits: int,
    paulis: list[str],
    entanglement: str,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Statevectors of the Pauli feature map for a batch of inputs.

    Every gate after the Hadamard layer is diagonal (RZ, MultiRZ), so the
    prepared state is a uniform superposition carrying one phase per basis
    state. Writing ``u = pi - bandwidth*x`` and ``z_i = +1/-1`` for bit ``i``
    of basis state ``b``, PennyLane's ``RZ(phi) = exp(-i phi Z/2)`` with
    ``phi_i = 2 u_i`` and ``phi_ij = 2 u_i u_j`` gives

        <b|psi(x)> = 2^(-n/2) exp(-i [sum_i u_i z_i + sum_(i,j) u_i u_j z_i z_j]).

    ``bandwidth`` rescales the input before encoding (x -> lambda*x), the
    standard bandwidth convention for quantum kernels; lambda=1 recovers the
    plain Havlicek scaling used elsewhere in the paper.

    Returns an ``(len(X), 2**n_qubits)`` complex array.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    dim = 2**n_qubits
    # z[b, i] = +1 if bit i of b is 0 else -1, with qubit 0 the most significant
    # bit, matching PennyLane's wire ordering in qml.probs/qml.state.
    bits = ((np.arange(dim)[:, None] >> np.arange(n_qubits - 1, -1, -1)[None, :]) & 1)
    z = 1.0 - 2.0 * bits  # (dim, n_qubits)

    u = np.pi - bandwidth * X  # (N, n_qubits)
    phase = u @ z.T  # (N, dim)

    if "ZZ" in paulis:
        pairs = _two_body_pairs(n_qubits, entanglement)
        pi_, pj_ = np.array(pairs).T
        uu = u[:, pi_] * u[:, pj_]  # (N, |P|)
        zz = z[:, pi_] * z[:, pj_]  # (dim, |P|)
        phase = phase + uu @ zz.T

    return np.exp(-1j * phase) / np.sqrt(dim)


def fast_kernel_matrix(
    A: np.ndarray,
    B: np.ndarray | None,
    n_qubits: int,
    paulis: list[str],
    entanglement: str,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Fidelity Gram matrix via one inner-product matmul on the statevectors.

    Numerically identical to ``kernel_matrix(make_fidelity_kernel(...), A, B)``
    (see ``test_fast_kernel_matches_reference``) but O(N^2 * 2^n) flops instead
    of O(N^2) circuit simulations.
    """
    psi_a = feature_states(A, n_qubits, paulis, entanglement, bandwidth)
    psi_b = psi_a if B is None else feature_states(B, n_qubits, paulis,
                                                   entanglement, bandwidth)
    overlap = psi_a @ psi_b.conj().T
    return np.abs(overlap) ** 2


def meyer_wallach_capacity(
    n_qubits: int,
    paulis: list[str],
    entanglement: str,
    n_samples: int = 400,
    seed: int = 0,
    device_name: str = "default.qubit",
) -> float:
    """Meyer-Wallach entangling capacity of the feature map over random data.

    Q = 2 * (1 - (1/n) sum_k Tr(rho_k^2)), averaged over random inputs. This
    is the data-encoding analogue of Sim et al. (2019)'s parameterised-circuit
    measure (kernel_trainer.metrics.EntanglingCapacity), here averaging over
    random data points rather than random gate parameters.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def state(x):
        pauli_feature_map(x, n_qubits, paulis, entanglement)
        return qml.state()

    rng = np.random.default_rng(seed)
    qs = []
    for _ in range(n_samples):
        x = rng.random(n_qubits)
        psi = state(x)
        rho = np.outer(psi, psi.conj())
        purity_sum = 0.0
        for k in range(n_qubits):
            rho_k = qml.math.reduce_dm(rho, indices=[k])
            purity_sum += float(np.real(np.trace(rho_k @ rho_k)))
        qs.append(2 * (1 - purity_sum / n_qubits))
    return float(np.mean(qs))


# Named encodings of graded entangling capacity, used across experiments.
ENCODINGS = [
    ("Z (product)", ["Z"], "linear"),
    ("Z+ZZ linear", ["Z", "ZZ"], "linear"),
    ("Z+ZZ full", ["Z", "ZZ"], "full"),
]
