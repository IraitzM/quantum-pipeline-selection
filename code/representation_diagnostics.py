"""Training-free diagnostics with explicit Fourier and compression conventions."""
from __future__ import annotations

import numpy as np


def signal_spectrum(x: np.ndarray) -> dict[str, np.ndarray]:
    """Centred real signals, unweighted selector power and Parseval power.

    The last axis is the signal coordinate. Constant signals are rejected because
    relative reconstruction error and a 95% variance bandwidth are undefined.
    """
    x = np.asarray(x, dtype=float)
    z = x - x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("Relative spectral diagnostics require nonconstant signals")
    z = z / norm * np.sqrt(x.shape[-1])
    power = np.abs(np.fft.rfft(z, axis=-1)) ** 2
    weights = np.full(power.shape[-1], 2.0)
    weights[0] = 1.0
    if x.shape[-1] % 2 == 0:
        weights[-1] = 1.0
    energy = power * weights
    return {"z": z, "selector_power": power / power.sum(axis=-1, keepdims=True),
            "energy": energy / energy.sum(axis=-1, keepdims=True)}


def bandwidth(power: np.ndarray, fraction: float = 0.95) -> np.ndarray:
    """Smallest prefix retaining a specified fraction of normalised power."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    cdf = np.cumsum(power, axis=-1)
    # Pin the final value to one to avoid roundoff at fraction=1.
    cdf[..., -1] = 1.0
    return np.argmax(cdf >= fraction, axis=-1)


def pool_projection_loss(x: np.ndarray, n: int = 8) -> np.ndarray:
    """Exact energy outside the row space of the implemented mean pool.

    The circuit pools n blocks of floor(d/n) samples and drops any remainder.
    The orthogonal projection repeats each block mean and fills the discarded
    remainder with zeros. This is a squared-error diagnostic, not Bayes error.
    """
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    if not 1 <= n <= d:
        raise ValueError("n must lie between 1 and the input dimension")
    width = d // n
    means = x[..., : n * width].reshape(*x.shape[:-1], n, width).mean(axis=-1)
    restored = np.zeros_like(x)
    restored[..., : n * width] = np.repeat(means, width, axis=-1)
    denominator = np.sum(x**2, axis=-1)
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError("Relative projection loss is undefined for zero-norm inputs")
    return np.sum((x - restored) ** 2, axis=-1) / denominator


def hankel_singular_values(x: np.ndarray) -> np.ndarray:
    """Same d//2 by (d-d//2+1) trajectory matrix as the QSM upload."""
    x = np.asarray(x, dtype=float)
    width = x.shape[-1] // 2
    if width < 1:
        raise ValueError("Hankel lifting needs at least two coordinates")
    trajectory = np.lib.stride_tricks.sliding_window_view(x, x.shape[-1] - width + 1, axis=-1)
    return np.linalg.svd(trajectory, compute_uv=False)


def native_spectrum(x: np.ndarray) -> np.ndarray:
    """Multidimensional Fourier energy grouped by max absolute frequency index.

    x has shape (samples, spatial_axis_1, ...). Prefix w is a square/cube in
    frequency space, not a radial band or a Fourier transform of a raster scan.
    """
    x = np.asarray(x, dtype=float)
    axes = tuple(range(1, x.ndim))
    centred = x - x.mean(axis=axes, keepdims=True)
    power = np.abs(np.fft.fftn(centred, axes=axes)) ** 2
    freq = [np.rint(np.abs(np.fft.fftfreq(d) * d)).astype(int) for d in x.shape[1:]]
    grids = np.meshgrid(*freq, indexing="ij")
    shell = np.maximum.reduce(grids).ravel()
    out = np.stack([np.bincount(shell, weights=p.ravel()) for p in power])
    if np.any(out.sum(axis=-1) == 0):
        raise ValueError("Native spectra require nonconstant inputs")
    return out / out.sum(axis=-1, keepdims=True)


def amplitude_witness(n_points: int = 256) -> dict[str, np.ndarray | float]:
    """Exact observable witness on the unit circle in embedding coordinates.

    A single real amplitude copy restricts any affine observable score to
    span{1, cos(2u), sin(2u)}. On two copies, Z tensor Z - X tensor X gives
    cos(4u). This uses an affine combination of observables, as a linear
    classifier read-out may do; it does not assert a particular ansatz fits it.
    """
    u = 2 * np.pi * np.arange(n_points) / n_points
    states = np.column_stack([np.cos(u), np.sin(u)])
    target = np.cos(4 * u)
    basis = np.column_stack([np.ones_like(u), np.cos(2 * u), np.sin(2 * u)])
    single = basis @ np.linalg.lstsq(basis, target, rcond=None)[0]
    z = np.diag([1.0, -1.0])
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    observable = np.kron(z, z) - np.kron(pauli_x, pauli_x)
    copies = np.einsum("ni,nj->nij", states, states).reshape(n_points, 4)
    double = np.einsum("ni,ij,nj->n", copies, observable, copies)
    return {"u": u, "target": target, "single": single, "double": double,
            "single_mse": float(np.mean((single - target) ** 2)),
            "double_mse": float(np.mean((double - target) ** 2))}


def spatial_image_diagnostics(x: np.ndarray, image_shape=(28, 28), n_features=8,
                              fraction=0.95) -> dict:
    """Native-2D Fourier proxy and exact fixed-convolution projection loss.

    The spectral proxy is energy outside |ky|<=gh//2 AND |kx|<=gw//2,
    for the output grid gh x gw. It is not the error of box convolution.
    Bandwidths summarize max(|ky|, |kx|) in native pixel coordinates.
    """
    from code.image_preprocessing import image_conv_grid
    images = np.asarray(x, dtype=float).reshape(-1, *image_shape)
    z = images - images.mean(axis=(1, 2), keepdims=True)
    total = np.sum(z*z, axis=(1, 2))
    if np.any(total <= 0):
        raise ValueError('Spectral diagnostics require nonconstant images')
    gh, gw = image_conv_grid(image_shape, n_features)
    kh, kw = image_shape[0] // gh, image_shape[1] // gw
    features = z.reshape(-1, gh, kh, gw, kw).mean(axis=(2, 4))
    projection = features.repeat(kh, axis=1).repeat(kw, axis=2)
    loss = ((z-projection)**2).sum(axis=(1, 2)) / total
    power = abs(np.fft.fft2(z))**2
    power /= power.sum(axis=(1, 2), keepdims=True)
    fy, fx = [np.rint(np.fft.fftfreq(d)*d).astype(int) for d in image_shape]
    ky, kx = np.meshgrid(fy, fx, indexing='ij')
    inside = (abs(ky) <= gh//2) & (abs(kx) <= gw//2)
    delta = power[:, ~inside].sum(axis=1)
    shell = np.maximum(abs(ky), abs(kx)).ravel()
    effective, maximum = [], []
    for p in power:
        band = np.bincount(shell, weights=p.ravel())
        effective.append(int(np.searchsorted(np.cumsum(band), fraction)))
        order = np.argsort(p.ravel(), kind='stable')[::-1]
        end = int(np.searchsorted(np.cumsum(p.ravel()[order]), fraction)) + 1
        maximum.append(int(shell[order[:end]].max()))
    return dict(omega_max=int(np.median(maximum)),
                omega_eff=int(np.median(effective)), delta_e=float(delta.mean()),
                pool_projection_loss=float(loss.mean()),
                per_input_delta_e=delta, per_input_pool_loss=loss,
                output_grid=(gh, gw), kernel_shape=(kh, kw),
                spectrum_geometry='native_2d', preprocessing='spatial_box_conv2d')
