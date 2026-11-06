"""
qnn.py
──────
QNN models used to evidence that a fixed-frequency, data-re-uploading
quantum circuit is, exactly as predicted by the theory in
``docs/quantum-favorability.md`` (Schuld et al., 2021), a finite Fourier
series in the classical input — and that whether this representation
fits a given dataset is decidable from the dataset's own structure
(bandwidth, non-separability) before any quantum model is trained.

Model families:

``FourierSeriesQNN``
    Single-qubit data-re-uploading regressor.  With ``degree`` repetitions
    of the encoding gate the accessible spectrum is exactly
    ``{-degree, ..., degree}`` (Fourier degree = number of re-uploads).
    Used to *directly* fit one signal window, and to check that the
    achieved degree matches the classical FFT-based ``omega_max`` estimate
    already computed by :mod:`code.main`.

``AngleEncodingClassifier``
    Multi-qubit data-re-uploading classifier.  A length-``L`` input signal
    is first compressed to an ``n_qubits``-dim vector (average pooling —
    a cheap low-pass compression), then re-uploaded ``n_reuploads`` times
    with a trainable ``StronglyEntanglingLayers`` block between each
    upload. A classical linear read-out maps the ``n_qubits`` Z-expectation
    values to class logits.

``FourierSeries2DQNN``
    Two-qubit data-re-uploading regressor for 2-D signals (images): wire 0
    encodes the row coordinate, wire 1 the column coordinate, output is
    ``<Z0 Z1>``. With no entangling gate between the wires the two qubits
    stay a product state and ``<Z0 Z1> = <Z0><Z1>`` is *provably* a
    separable (rank-1) function ``g(x) h(y)``; a CNOT between the wires at
    each re-upload lifts this restriction. Used to test directly whether a
    non-separable 2-D signal needs entanglement to be well approximated.

``AmplitudeEncodingClassifier`` / ``AmplitudeReuploadClassifier``
    Amplitude-embedding classifiers ingesting the full, unpooled input on
    ``ceil(log2 d)`` qubits (the re-upload variant prepares ``k`` tensor
    copies on ``k`` registers). No pooling bottleneck, at the price of an
    exponentially deep state preparation on real hardware.

``HankelQSMClassifier`` (+ ``hankel_block_unitary``)
    Quantum Spectral Model (Wang, Parampalli & Myers, arXiv:2607.22516)
    for 1-D signals: each signal is lifted to its trajectory (Hankel)
    matrix and uploaded through the block-Hamiltonian unitary
    ``exp(-i H_block t / 2)``, so the candidate Fourier support is the
    sample's own Hankel singular gaps rather than a fixed grid. Ingests
    the full signal with no pooling, like the amplitude classifiers, but
    with a sample-conditioned spectrum.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn


# ─── Quantum Spectral Model (input-conditioned Hamiltonian encoding) ────
#
# Wang, Parampalli & Myers (arXiv:2607.22516) replace the coordinate-wise
# rotation gates of angle encoding by an upload unitary whose *generator*
# is built from the input matrix M, U(x) = exp(-i H(M) t / 2), so the
# candidate Fourier support is the sample's own set of eigengaps rather
# than the fixed grid {-D, ..., D}. We take their global block-Hamiltonian
# variant, their Eq. (15)-(16). Their construction needs a matrix input;
# our benchmarks are 1-D signals, so we lift each signal x of length d to
# its trajectory (Hankel) matrix H[i, j] = x[i + j], whose singular
# spectrum is the object singular spectrum analysis (SSA) uses for 1-D
# spectral structure. That lift is ours and is not in their paper. The
# block Hamiltonian
#       H_block = [[0, H], [H^T, 0]]
# is real symmetric with eigenvalues +-sigma_i(H), i.e. the upload's
# candidate support is the set of signed singular-gap combinations of the
# Hankel matrix.


def hankel_block_eigh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of the Hankel-lifted, padded block Hamiltonian.

    ``x`` is one signal of length ``d`` (or a batch ``(N, d)``), assumed
    z-scored per sample. The trajectory matrix uses window
    ``L = d // 2``, ``K = d - L + 1``, so ``H_block`` has size
    ``p = L + K = d + 1``; it is then promoted to ``2^m x 2^m`` by the
    direct-sum zero padding ``H -> H (+) 0`` of Wang et al., equivalently
    ``U -> U (+) I``, which preserves the nonzero spectrum and adds only
    zero modes, leaving the candidate gap set exactly that of the
    unpadded generator. Padding with an identity block instead would add
    eigenvalue-1 levels and with them carriers that are not Hankel
    singular gaps.

    Returns ``(eigvals, eigvecs)`` of shape ``(2^m,) / (2^m, 2^m)``, or
    batched with a leading ``N``. ``H_block`` is real symmetric, so the
    decomposition is exact and the upload unitary at any time ``t`` is
    ``eigvecs @ diag(exp(-i t eigvals / 2)) @ eigvecs.T``.
    """
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    n, d = x.shape
    window = d // 2
    k_cols = d - window + 1
    # Trajectory matrix via stride tricks: row i, column j holds x[i + j].
    stride = x.strides[1]
    traj = np.lib.stride_tricks.as_strided(
        x, shape=(n, window, k_cols), strides=(x.strides[0], stride, stride)
    )
    p = window + k_cols
    dim = int(2 ** np.ceil(np.log2(p)))

    # Zero padding: the trailing dim - p rows and columns stay empty.
    h_block = np.zeros((n, dim, dim), dtype=np.float64)
    h_block[:, :window, window:p] = traj
    h_block[:, window:p, :window] = np.swapaxes(traj, 1, 2)

    eigvals, eigvecs = np.linalg.eigh(h_block)
    return (eigvals[0], eigvecs[0]) if single else (eigvals, eigvecs)


def hankel_block_unitary(x: np.ndarray, t: float = 0.1) -> np.ndarray:
    """Upload unitary ``exp(-i H_block t / 2)`` at a fixed upload time.

    ``t`` is held fixed here, which is a deliberate departure from Wang
    et al.: their Eq. (26) makes the per-layer time ``t_l`` trainable
    (initialised to ``1/L``). Freezing it keeps the encoder free of
    trainable parameters, so the map from data to features is fixed once
    the sample is seen. :class:`TrainableTimeQSMClassifier` is the
    matching arm that restores their trainable time.

    Returns a complex128 array of shape ``(2^m, 2^m)`` or
    ``(N, 2^m, 2^m)``. The unitary is the same at every upload, so it
    should be precomputed once per dataset rather than inside the
    training loop.
    """
    eigvals, eigvecs = hankel_block_eigh(x)
    single = eigvals.ndim == 1
    if single:
        eigvals, eigvecs = eigvals[None], eigvecs[None]
    phases = np.exp(-0.5j * t * eigvals)
    u = eigvecs * phases[:, None, :]
    u = u @ np.swapaxes(eigvecs.conj(), 1, 2)
    return u[0] if single else u


class HankelQSMClassifier(nn.Module):
    """Quantum Spectral Model classifier over Hankel-lifted 1-D signals.

    Each layer applies the data-dependent upload ``U(x)`` (precomputed by
    :func:`hankel_block_unitary`, identical at every layer since it is a
    function of the input only) followed by a brick-wall mixer of
    nearest-neighbour ``qml.SpecialUnitary`` gates — a parameterised
    SU(4) with 15 parameters per neighbouring pair, ``15(n_qubits - 1)``
    per mixer layer, even bonds before odd, matching Wang et al.'s
    Eq. (147)-(150) exactly.

    Three deliberate departures from Wang et al., all of which keep this
    model comparable with the rest of the repo rather than with theirs:

    * the upload time is fixed, not the trainable ``t_l`` of their
      Eq. (26), so the encoder carries no trainable parameters (see
      :class:`TrainableTimeQSMClassifier` for the faithful arm);
    * the read-out is local ``<Z_i>`` expectations feeding a classical
      linear layer, not their class projector ``|j><j| (x) I``, so the
      comparison with :class:`AngleEncodingClassifier` isolates the
      encoding rather than the read-out;
    * the initial state is ``|0...0>``, this repo's convention, where
      they use ``|+>^{(x)n}``.

    The model ingests the *full, unpooled* signal (``d`` samples need
    only ``ceil(log2(d + 1))`` qubits), so it sidesteps the average-pool
    bottleneck that the paper's first axis blames for the angle
    encoding's failures. ``forward`` takes the precomputed upload
    unitaries, not the raw signals.
    """

    def __init__(
        self,
        n_qubits: int,
        n_uploads: int = 8,
        n_classes: int = 2,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_uploads = n_uploads
        self.dev = qml.device(device_name, wires=n_qubits)
        # 15 = 4^2 - 1 parameters per two-qubit SU(4); n_qubits - 1 pairs
        # in the brick wall (even bonds then odd bonds).
        self.weights = nn.Parameter(
            0.1 * torch.randn(n_uploads, n_qubits - 1, 15, dtype=torch.float64)
        )

        even_bonds = list(range(0, n_qubits - 1, 2))
        odd_bonds = list(range(1, n_qubits - 1, 2))
        bonds = even_bonds + odd_bonds

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(u, weights):
            for r in range(n_uploads):
                qml.QubitUnitary(u, wires=range(n_qubits))
                for pair, w in zip(bonds, weights[r]):
                    qml.SpecialUnitary(w, wires=[pair, pair + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits, n_classes, dtype=torch.float64)

    @staticmethod
    def n_qubits_for(seq_len: int) -> int:
        """Qubits needed for a length-``seq_len`` signal: ``ceil(log2(d + 1))``."""
        return int(np.ceil(np.log2(seq_len + 1)))

    def _q_features(self, u_batch: torch.Tensor) -> torch.Tensor:
        outs = [
            torch.stack(self._circuit(u_batch[i], self.weights)).reshape(-1)
            for i in range(u_batch.shape[0])
        ]
        return torch.stack(outs)

    def forward(self, u_batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self._q_features(u_batch))


class TrainableTimeQSMClassifier(nn.Module):
    """Hankel QSM with the trainable per-layer upload time of Wang et al.

    Their Eq. (26) is ``S_l(M) = exp(-i H_block(M) t_l / 2)`` with ``t_l``
    a trainable scalar per layer, initialised to ``1/L``, and their
    Section 4 treats the upload time as *the* Fourier variable: the
    realised frequency support is the generator's eigengaps scaled by
    ``t_l``, so training it fits the support to the task. That makes this
    arm strictly more adaptive than :class:`HankelQSMClassifier`, which
    freezes the time — the pair brackets what the trainable time buys.

    Because ``t_l`` is trainable the upload unitary cannot be precomputed,
    but the eigendecomposition of ``H_block`` can, since it depends on the
    sample alone. ``forward`` therefore takes the ``(eigvals, eigvecs)``
    pair from :func:`hankel_block_eigh` and rebuilds
    ``V diag(exp(-i t_l w / 2)) V^T`` per layer, which is differentiable
    in ``t_l``. Read-out and initial state follow
    :class:`HankelQSMClassifier`, so the two arms differ only in the time.
    """

    def __init__(
        self,
        n_qubits: int,
        n_uploads: int = 8,
        n_classes: int = 2,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_uploads = n_uploads
        self.dev = qml.device(device_name, wires=n_qubits)
        self.weights = nn.Parameter(
            0.1 * torch.randn(n_uploads, n_qubits - 1, 15, dtype=torch.float64)
        )
        # Their initialisation: t_l = 1/L for every layer.
        self.upload_times = nn.Parameter(
            torch.full((n_uploads,), 1.0 / n_uploads, dtype=torch.float64)
        )

        even_bonds = list(range(0, n_qubits - 1, 2))
        odd_bonds = list(range(1, n_qubits - 1, 2))
        bonds = even_bonds + odd_bonds

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(uploads, weights):
            for r in range(n_uploads):
                qml.QubitUnitary(uploads[r], wires=range(n_qubits))
                for pair, w in zip(bonds, weights[r]):
                    qml.SpecialUnitary(w, wires=[pair, pair + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits, n_classes, dtype=torch.float64)

    n_qubits_for = staticmethod(HankelQSMClassifier.n_qubits_for)

    def _uploads(self, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Stack of ``n_uploads`` unitaries for one sample's eigendecomposition."""
        # (n_uploads, dim) eigenangles, then V diag(phase) V^T per layer.
        phases = torch.exp(-0.5j * self.upload_times[:, None] * w[None, :])
        vc = v.to(torch.complex128)
        return (vc[None, :, :] * phases[:, None, :]) @ vc.T[None, :, :]

    def forward(self, w_batch: torch.Tensor, v_batch: torch.Tensor) -> torch.Tensor:
        outs = [
            torch.stack(self._circuit(self._uploads(w_batch[i], v_batch[i]), self.weights))
            .reshape(-1)
            for i in range(w_batch.shape[0])
        ]
        return self.readout(torch.stack(outs))


# ─── Single-qubit Fourier-series regressor ────────────────────────────────


class FourierSeriesQNN(nn.Module):
    """Single-qubit data-re-uploading Fourier model, degree = ``degree``."""

    def __init__(self, degree: int, device_name: str = "lightning.qubit"):
        super().__init__()
        self.degree = degree
        self.dev = qml.device(device_name, wires=1)
        # W_0 . S(x) . W_1 . S(x) . ... . S(x) . W_degree  ->  degree+1 trainable
        # blocks bracketing `degree` encoding gates (Schuld et al., 2021, Fig. 2).
        # Without the leading W_0, S(x)=RZ(x) acting on |0> (a Z-eigenstate) is
        # invariant up to global phase, silently wasting the first encoding gate.
        self.rot_weights = nn.Parameter(0.1 * torch.randn(degree + 1, 3))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, rot_weights):
            qml.Rot(*rot_weights[0], wires=0)
            for layer in range(degree):
                qml.RZ(x, wires=0)
                qml.Rot(*rot_weights[layer + 1], wires=0)
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def forward_scalar(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._circuit(x, self.rot_weights)
        return self.scale * raw + self.bias

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_scalar(x) for x in xs])


# ─── Multi-qubit angle-encoding classifier ────────────────────────────────


def _pool(x: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Legacy one-dimensional pool; image callers must supply explicit geometry."""
    from code.image_preprocessing import angle_input_features
    return angle_input_features(x, n_qubits)


class AngleEncodingClassifier(nn.Module):
    """Data-re-uploading angle-encoding classifier with two independent
    entanglement toggles, to isolate what entanglement contributes:

    ``use_entangling_ansatz``
        ``True`` (default): the trainable block between uploads is
        ``StronglyEntanglingLayers`` (single-qubit rotations + a CNOT ring).
        ``False``: the same per-qubit rotations with the CNOT ring removed,
        i.e. the ``n_qubits`` wires stay a product state throughout — an
        ensemble of independent single-qubit Fourier models sharing a
        classical read-out. Parameter count is identical either way.

    ``use_entangling_measurement``
        ``True``: append pairwise ``Z_i Z_{i+1}`` correlators (ring topology,
        matching the entangler's connectivity) to the local ``Z_i``
        expectation values fed to the read-out. ``False`` (default): local
        ``Z_i`` expectations only. Note that if the ansatz is a product
        state (``use_entangling_ansatz=False``), ``<Z_i Z_j> = <Z_i><Z_j>``
        identically, so this toggle can only add information when combined
        with an entangling ansatz — a useful internal consistency check.
    """

    def __init__(
        self,
        seq_len: int,
        n_qubits: int = 8,
        n_reuploads: int = 4,
        n_classes: int = 2,
        entangler_layers: int = 1,
        use_entangling_ansatz: bool = True,
        use_entangling_measurement: bool = False,
        device_name: str = "lightning.qubit",
        image_shape: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.image_shape = image_shape
        if image_shape is not None:
            from code.image_preprocessing import image_conv_grid
            if image_shape[0] * image_shape[1] != seq_len:
                raise ValueError("image_shape must match seq_len")
            image_conv_grid(image_shape, n_qubits)
        self.n_qubits = n_qubits
        self.n_reuploads = n_reuploads
        self.use_entangling_ansatz = use_entangling_ansatz
        self.use_entangling_measurement = use_entangling_measurement
        self.dev = qml.device(device_name, wires=n_qubits)

        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=entangler_layers, n_wires=n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(n_reuploads, *weight_shape))

        def variational_block(w):
            # w: (entangler_layers, n_qubits, 3), identical shape/param count
            # regardless of ansatz so the two conditions are directly comparable.
            if use_entangling_ansatz:
                qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
            else:
                for layer in range(w.shape[0]):
                    for wire in range(n_qubits):
                        qml.Rot(*w[layer, wire], wires=wire)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(features, weights):
            for r in range(n_reuploads):
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
                variational_block(weights[r])
            obs = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            if use_entangling_measurement:
                obs += [
                    qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
                    for i in range(n_qubits)
                ]
            return obs

        self._circuit = circuit
        n_out_features = n_qubits * (2 if use_entangling_measurement else 1)
        self.readout = nn.Linear(n_out_features, n_classes)

    def _q_features(self, x: torch.Tensor) -> torch.Tensor:
        from code.image_preprocessing import angle_input_features
        pooled = angle_input_features(x, self.n_qubits, self.image_shape)
        outs = [
            torch.stack(self._circuit(pooled[i], self.weights)).reshape(-1)
            for i in range(pooled.shape[0])
        ]
        return torch.stack(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_feats = self._q_features(x)
        return self.readout(q_feats)


# ─── Two-qubit Fourier-series regressor for 2-D signals (images) ─────────


class FourierSeries2DQNN(nn.Module):
    """Two-qubit data-re-uploading Fourier model over (x, y), degree = ``degree`` per axis.

    ``entangled=True``: a CNOT(0,1) is applied between the two wires at
    every re-upload, so ``<Z0 Z1>`` can represent a general (non-separable)
    2-D truncated Fourier series with independent coefficients per (m, n).

    ``entangled=False``: the two wires never interact, so the state stays a
    product state and ``<Z0 Z1> = <Z0><Z1>`` is provably a *separable*
    function ``g(x) h(y)`` -- the 2-D analogue of a rank-1 approximation.
    Both variants have exactly the same parameter count.
    """

    def __init__(self, degree: int, entangled: bool = True, device_name: str = "lightning.qubit"):
        super().__init__()
        self.degree = degree
        self.entangled = entangled
        self.dev = qml.device(device_name, wires=2)
        self.rot_weights = nn.Parameter(0.1 * torch.randn(degree + 1, 2, 3))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, y, rot_weights):
            qml.Rot(*rot_weights[0, 0], wires=0)
            qml.Rot(*rot_weights[0, 1], wires=1)
            for layer in range(degree):
                qml.RZ(x, wires=0)
                qml.RZ(y, wires=1)
                if entangled:
                    qml.CNOT(wires=[0, 1])
                qml.Rot(*rot_weights[layer + 1, 0], wires=0)
                qml.Rot(*rot_weights[layer + 1, 1], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self._circuit = circuit

    def forward_scalar(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw = self._circuit(x, y, self.rot_weights)
        return self.scale * raw + self.bias

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.forward_scalar(x, y) for x, y in zip(xs, ys)]
        )


# ─── Amplitude-encoding classifier (no pooling needed) ────────────────────


def to_nonneg_domain(x: np.ndarray) -> np.ndarray:
    """Per-sample min-max scale to [0, 1] before amplitude embedding.

    Kang, Sevior & Usman (arXiv:2606.14206) prove that a *symmetric* domain
    (e.g. zero-mean features, as used elsewhere in this repo) makes the
    zero-frequency Fourier coefficient of an amplitude-embedded circuit
    vanish identically; a non-negative domain does not have this pathology.
    We therefore always rescale to a non-negative range before amplitude
    embedding, regardless of how the same feature was normalised for the
    angle-encoding experiments.
    """
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    return (x - lo) / (hi - lo + 1e-8)


class AmplitudeEncodingClassifier(nn.Module):
    """Classifier via amplitude embedding of the *full, unpooled* input.

    Unlike :class:`AngleEncodingClassifier`, which needs one qubit per
    encoded feature and therefore forces average-pooling to fit a small
    qubit budget, amplitude embedding needs only
    ``n_qubits = ceil(log2(input_dim))`` qubits for any input dimension --
    e.g. 10 qubits for a 784-pixel image, vs. the 8-qubit *pooled* budget
    used elsewhere in this repo. This lets the classifier see the entire
    input directly, at the cost of an amplitude-embedding state preparation
    that would be expensive on real hardware (cheap here since we simulate
    classically -- see paper limitations).
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        n_classes: int = 2,
        device_name: str = "lightning.qubit",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = max(1, int(np.ceil(np.log2(input_dim))))
        self.dev = qml.device(device_name, wires=self.n_qubits)

        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=self.n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(*weight_shape))

        n_qubits = self.n_qubits

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, weights):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits, n_classes)

    def _q_features(self, x: torch.Tensor) -> torch.Tensor:
        outs = [
            torch.stack(self._circuit(x[i], self.weights)).reshape(-1)
            for i in range(x.shape[0])
        ]
        return torch.stack(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self._q_features(x))


class AmplitudeReuploadClassifier(nn.Module):
    """Tensor-re-embedding amplitude classifier (Route 1 "amplitude re-uploading").

    Amplitude embedding is a state *preparation*, not a data-dependent
    evolution, so it cannot be literally re-applied mid-circuit the way an
    angle rotation can. The tractable analogue of re-uploading is to prepare
    ``k`` tensor copies ``|x>^{\\otimes k}`` on ``k`` separate registers of
    ``ceil(log2 d)`` qubits each and entangle them with a single variational
    block. The amplitudes of ``|x>^{\\otimes k}`` are degree-``k`` monomials
    in the features, so a measured observable realises a degree-``2k``
    polynomial in ``x`` -- the amplitude-embedding analogue of angle
    re-uploading, which instead grows the *Fourier* degree. ``k=1`` recovers
    :class:`AmplitudeEncodingClassifier`.
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 2,
        n_layers: int = 3,
        n_classes: int = 2,
        device_name: str = "lightning.qubit",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.n_per = max(1, int(np.ceil(np.log2(input_dim))))
        self.n_qubits = self.k * self.n_per
        self.dev = qml.device(device_name, wires=self.n_qubits)

        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=self.n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(*weight_shape))

        n_per, k, n_qubits = self.n_per, self.k, self.n_qubits

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, weights):
            for c in range(k):
                wires = range(c * n_per, (c + 1) * n_per)
                qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits, n_classes)

    def _q_features(self, x: torch.Tensor) -> torch.Tensor:
        outs = [
            torch.stack(self._circuit(x[i], self.weights)).reshape(-1)
            for i in range(x.shape[0])
        ]
        return torch.stack(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self._q_features(x))
