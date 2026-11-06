"""
datasets.py
───────────
Loaders for the signal-based benchmarks used to evidence that angle-encoded
QNNs (finite Fourier series in the input, Schuld et al. 2021) approximate
real periodic/quasi-periodic signals well:

  * MNIST1D    — Greydanus (2024) 1-D projection of MNIST-like digit strokes,
                 length-40 sequences, 10 classes.
  * ECG beats  — MIT-BIH Arrhythmia Database (PhysioNet), length-40 windows
                 centred on the annotated R-peak, Normal vs. Ventricular
                 ectopic (PVC) beats.
  * EEG seizure — UCI Epileptic Seizure Recognition (id 388; reshaped from
                 Andrzejak et al. 2001's Bonn single-channel EEG data),
                 length-178 windows, seizure vs. non-seizure.

MNIST1D and ECG are cached as fixed-length (L=40) numeric sequences so the
same angle-encoding circuit architecture can be reused across them; EEG
seizure windows are natively length 178.
"""

from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEQ_LEN = 40

MNIST1D_URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
MNIST1D_PKL = DATA_DIR / "mnist1d_data.pkl"

# UCI Epileptic Seizure Recognition (id 388); fetched once via Hugging Face
# Hub mirror since the UCI static-file endpoints for this legacy dataset
# return 403/404. Cached locally at DATA_DIR / EEG_SEIZURE_CSV.
EEG_SEIZURE_CSV = DATA_DIR / "eeg_seizure_uci388.csv"
EEG_SEIZURE_URL = (
    "https://huggingface.co/datasets/ankislyakov/EpilepticSeizureRecognition/"
    "resolve/main/Epileptic%20Seizure%20Recognition.csv"
)

# Records with a healthy mix of Normal (N) and PVC (V) beats.
ECG_TRAIN_RECORDS = ["106", "119", "200", "208"]
ECG_TEST_RECORDS = ["203", "223"]
ECG_HALF_WINDOW = SEQ_LEN // 2  # 20 samples either side of the R-peak


def load_mnist1d() -> dict[str, np.ndarray]:
    """Download (once) and load the official MNIST1D benchmark."""
    if not MNIST1D_PKL.exists():
        urllib.request.urlretrieve(MNIST1D_URL, MNIST1D_PKL)
    with open(MNIST1D_PKL, "rb") as f:
        raw = pickle.load(f)
    return {
        "x_train": np.asarray(raw["x"], dtype=np.float64),
        "y_train": np.asarray(raw["y"], dtype=np.int64),
        "x_test": np.asarray(raw["x_test"], dtype=np.float64),
        "y_test": np.asarray(raw["y_test"], dtype=np.int64),
        "t": np.asarray(raw["t"], dtype=np.float64),
    }


def _extract_beats(record_id: str, max_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length beat windows (label 0=Normal, 1=PVC) from one MIT-BIH record."""
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    signal = record.p_signal[:, 0]  # lead MLII

    beats, labels = [], []
    n_count = v_count = 0
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in ("N", "V"):
            continue
        lo, hi = sample - ECG_HALF_WINDOW, sample + ECG_HALF_WINDOW
        if lo < 0 or hi > len(signal):
            continue
        label = 0 if symbol == "N" else 1
        if max_per_class is not None:
            if label == 0 and n_count >= max_per_class:
                continue
            if label == 1 and v_count >= max_per_class:
                continue
        window = signal[lo:hi].astype(np.float64)
        window = (window - window.mean()) / (window.std() + 1e-8)
        beats.append(window)
        labels.append(label)
        if label == 0:
            n_count += 1
        else:
            v_count += 1
    return np.asarray(beats), np.asarray(labels, dtype=np.int64)


def load_ecg_beats(
    max_per_class_train: int = 400,
    max_per_class_test: int = 100,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Build a patient-disjoint Normal-vs-PVC beat classification set from MIT-BIH."""
    cache_path = DATA_DIR / "ecg_beats.npz"
    if cache and cache_path.exists():
        npz = np.load(cache_path)
        return {k: npz[k] for k in npz.files}

    x_train_parts, y_train_parts = [], []
    for rid in ECG_TRAIN_RECORDS:
        x, y = _extract_beats(rid, max_per_class=max_per_class_train)
        x_train_parts.append(x)
        y_train_parts.append(y)

    x_test_parts, y_test_parts = [], []
    for rid in ECG_TEST_RECORDS:
        x, y = _extract_beats(rid, max_per_class=max_per_class_test)
        x_test_parts.append(x)
        y_test_parts.append(y)

    data = {
        "x_train": np.concatenate(x_train_parts, axis=0),
        "y_train": np.concatenate(y_train_parts, axis=0),
        "x_test": np.concatenate(x_test_parts, axis=0),
        "y_test": np.concatenate(y_test_parts, axis=0),
    }

    if cache:
        np.savez_compressed(cache_path, **data)
    return data


# ─── Synthetic overlap-controlled benchmark (kernel-trainer '2c' template) ──
#
# Independent of the periodic-signal story above: a 4x4 checkerboard of 3-D
# ellipsoid clusters (kernel_trainer.dataset.DataGenerator, template '2c').
# The checkerboard label pattern is inherently non-linear (XOR-like) at any
# overlap, and the ``overlap`` parameter continuously slides cluster centres
# towards the cube centre, letting us measure how classification difficulty
# scales with class overlap -- independent of any spectral/Fourier argument,
# this checks how the models compare "in the first place" on a controllable,
# low-dimensional, densely-labelled synthetic benchmark.


def load_2c_overlap(
    overlap: float,
    samples_train: int = 900,
    samples_test: int = 300,
    imbalance_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate train/test splits of the '2c' checkerboard template at a given overlap."""
    from kernel_trainer.dataset import DataGenerator

    gen_train = DataGenerator(samples=samples_train, imbalance_ratio=imbalance_ratio, seed=seed)
    gen_test = DataGenerator(
        samples=samples_test, imbalance_ratio=imbalance_ratio, seed=seed + 1
    )
    df_train = gen_train.generate_dataset("2c", overlap=overlap)
    df_test = gen_test.generate_dataset("2c", overlap=overlap)

    feature_cols = [c for c in df_train.columns if c != "y"]
    return {
        "x_train": df_train[feature_cols].to_numpy(dtype=np.float64),
        "y_train": df_train["y"].to_numpy(dtype=np.int64),
        "x_test": df_test[feature_cols].to_numpy(dtype=np.float64),
        "y_test": df_test["y"].to_numpy(dtype=np.int64),
    }


# ─── UCI Epileptic Seizure Recognition (real EEG, id 388) ───────────────────


def load_eeg_seizure(
    n_per_class_train: int = 800,
    n_per_class_test: int = 200,
    seed: int = 42,
    split: str = "random",
) -> dict[str, np.ndarray]:
    """Binary seizure-vs-non-seizure task from raw, length-178 EEG windows.

    Label 1 in the original 5-class scheme is seizure activity; labels
    2-5 are pooled into the non-seizure class.

    ``split`` selects how train and test are separated.

    ``"random"`` is the historical default and reproduces every EEG table in
    the paper. It draws windows at random, stratified by label, which is how
    this dataset is normally used in the literature and which leaks badly. The
    public CSV holds 11,500 rows built from 500 recordings cut into 23
    consecutive windows each, so a random draw puts sibling windows of one
    recording on both sides. At the sizes these experiments use, 59% of test
    windows share a recording with the training set, and neighbouring windows
    of one 23.6-second recording are near duplicates.

    ``"recording"`` splits on the recording first and draws windows only
    afterwards, so no recording contributes to both sides. The identifier is
    the ``Unnamed`` column, whose entries read ``X21.V1.791`` for window 21 of
    recording 791. An earlier version of this docstring asserted that no such
    identifier existed, which was wrong, and the paper repeated the claim.
    """
    if split not in ("random", "recording"):
        raise ValueError(f"split must be 'random' or 'recording', got {split!r}")

    if not EEG_SEIZURE_CSV.exists():
        urllib.request.urlretrieve(EEG_SEIZURE_URL, EEG_SEIZURE_CSV)

    df = pd.read_csv(EEG_SEIZURE_CSV)
    feature_cols = [c for c in df.columns if c.startswith("X")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = (df["y"].to_numpy(dtype=np.int64) == 1).astype(np.int64)  # 1=seizure, 0=other

    rng = np.random.default_rng(seed)
    n_train = n_per_class_train
    n_test = n_per_class_test

    if split == "recording":
        # ``X21.V1.791`` -> ``V1.791``; every recording carries one label and
        # exactly 23 windows, so grouping is exact rather than approximate.
        rec = df["Unnamed"].astype(str).str.replace(r"^X\d+\.", "", regex=True).to_numpy()
        train_parts, test_parts = [], []
        for cls in (1, 0):
            rows = np.where(y == cls)[0]
            recs = np.unique(rec[rows])
            rng.shuffle(recs)
            # Allocate recordings in proportion to the window demand, then draw
            # windows inside each pool. Both pools keep at least one recording.
            frac = n_train / (n_train + n_test)
            cut = int(round(frac * len(recs)))
            cut = min(max(cut, 1), len(recs) - 1)
            tr_recs, te_recs = set(recs[:cut]), set(recs[cut:])
            tr_pool = rows[np.isin(rec[rows], list(tr_recs))]
            te_pool = rows[np.isin(rec[rows], list(te_recs))]
            rng.shuffle(tr_pool)
            rng.shuffle(te_pool)
            if len(tr_pool) < n_train or len(te_pool) < n_test:
                raise ValueError(
                    f"recording split cannot supply {n_train}/{n_test} windows for "
                    f"class {cls}: pools hold {len(tr_pool)}/{len(te_pool)}"
                )
            train_parts.append(tr_pool[:n_train])
            test_parts.append(te_pool[:n_test])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
    else:
        idx_pos = rng.permutation(np.where(y == 1)[0])
        idx_neg = rng.permutation(np.where(y == 0)[0])
        train_idx = np.concatenate([idx_pos[:n_train], idx_neg[:n_train]])
        test_idx = np.concatenate(
            [idx_pos[n_train : n_train + n_test], idx_neg[n_train : n_train + n_test]]
        )
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    def _znorm(a: np.ndarray) -> np.ndarray:
        return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)

    return {
        "x_train": _znorm(X[train_idx]),
        "y_train": y[train_idx],
        "x_test": _znorm(X[test_idx]),
        "y_test": y[test_idx],
    }


# ─── MedMNIST (2-D image benchmark) ─────────────────────────────────────────
#
# PneumoniaMNIST: 28x28 grayscale pediatric chest X-rays, binary
# normal-vs-pneumonia. The 2-D analogue of the 1-D signal datasets above:
# used to test whether genuinely two-dimensional (non-separable) spatial
# structure is where entanglement earns its keep, rather than just
# spectral bandwidth.


def load_pneumoniamnist(
    n_per_class_train: int = 300,
    n_per_class_test: int = 150,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Class-balanced PneumoniaMNIST subset, images returned as (N, 28, 28) float64 in [0, 1]."""
    from medmnist import PneumoniaMNIST

    train_ds = PneumoniaMNIST(split="train", download=True, size=28)
    test_ds = PneumoniaMNIST(split="test", download=True, size=28)

    def _to_arrays(ds):
        x = np.stack([np.asarray(ds[i][0], dtype=np.float64) for i in range(len(ds))])
        y = np.array([ds[i][1][0] for i in range(len(ds))], dtype=np.int64)
        return x / 255.0, y

    x_train_all, y_train_all = _to_arrays(train_ds)
    x_test_all, y_test_all = _to_arrays(test_ds)

    rng = np.random.default_rng(seed)

    def _balanced_sample(x, y, n_per_class):
        idx_all = []
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            idx_all.append(idx_c[:n_per_class])
        idx = np.concatenate(idx_all)
        rng.shuffle(idx)
        return x[idx], y[idx]

    x_train, y_train = _balanced_sample(x_train_all, y_train_all, n_per_class_train)
    x_test, y_test = _balanced_sample(x_test_all, y_test_all, n_per_class_test)

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


# ─── HLS-CMDS heart-vs-lung sounds (oscillatory audio, low data) ─────────────
#
# Heart and Lung Sounds Dataset (Torabi, Shirani & Reilly, Zenodo 15376628):
# 4 kHz, 15 s clinical-manikin stethoscope recordings, 100 heart (HS) and 50
# lung (LS). We window each recording and resample every window to a fixed
# length so the same encoders used on ECG/EEG apply, and split by RECORDING so
# no two windows of one recording straddle the train/test boundary. Heart takes
# fewer windows and lung more, to balance the 100/50 recording counts.

HLS_DIR = DATA_DIR / "hls_cmds"


def load_hls_cmds(
    target_len: int = 178,
    win_seconds: float = 1.0,
    heart_windows: int = 2,
    lung_windows: int = 4,
    test_frac: float = 0.3,
    seed: int = 42,
    representation: str = "waveform",
) -> dict[str, np.ndarray]:
    """Heart-vs-lung (0/1) fixed-length windows, split at the recording level.

    ``representation`` selects the feature form. ``"waveform"`` resamples the raw
    time-domain window (the form the paper's Fourier-encoding framework assumes,
    on which unaligned audio is not discriminative). ``"spectral"`` returns the
    log-power-spectral envelope of the window, a classical transform under which
    heart and lung separate.
    """
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    if representation not in ("waveform", "spectral"):
        raise ValueError(f"unknown representation {representation!r}")
    rng = np.random.default_rng(seed)

    def recordings(cat: str) -> list[str]:
        out = []
        for root, _, files in os.walk(HLS_DIR / cat):
            if "__MACOSX" in str(root):
                continue
            for f in files:
                if f.lower().endswith(".wav") and not f.startswith("._"):
                    out.append(os.path.join(root, f))
        return sorted(out)

    def windows(path: str, n_win: int) -> np.ndarray:
        x, sr = sf.read(path)
        if x.ndim > 1:
            x = x.mean(axis=1)
        w = int(win_seconds * sr)
        if len(x) < w:
            x = np.pad(x, (0, w - len(x)))
        starts = np.linspace(0, len(x) - w, n_win).astype(int)
        out = []
        for s in starts:
            seg = x[s:s + w]
            if representation == "spectral":
                mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                feat = np.log1p(np.maximum(resample_poly(mag, target_len, len(mag)), 0.0))
            else:
                feat = resample_poly(seg, target_len, w)  # anti-aliased -> target_len
            feat = (feat[:target_len] if len(feat) >= target_len
                    else np.pad(feat, (0, target_len - len(feat))))
            out.append(feat.astype(np.float64))
        return np.asarray(out)

    split: dict[str, list] = {"train": [], "test": []}
    for cat, label, n_win in [("HS", 0, heart_windows), ("LS", 1, lung_windows)]:
        recs = recordings(cat)
        rng.shuffle(recs)
        n_test = int(round(len(recs) * test_frac))
        for part, paths in (("test", recs[:n_test]), ("train", recs[n_test:])):
            for p in paths:
                for seg in windows(p, n_win):
                    split[part].append((seg, label))

    def pack(pairs: list) -> tuple[np.ndarray, np.ndarray]:
        order = rng.permutation(len(pairs))
        x = np.stack([pairs[i][0] for i in order])
        y = np.array([pairs[i][1] for i in order], dtype=np.int64)
        return x, y

    x_train, y_train = pack(split["train"])
    x_test, y_test = pack(split["test"])
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


# ─── MedMNIST 3-D (real volumetric benchmark) ───────────────────────────────
#
# NoduleMNIST3D: 28x28x28 low-dose chest-CT lung-nodule volumes, binary
# benign-vs-malignant. The real, genuinely 3-D counterpart to the synthetic
# 3-D ``2c`` checkerboard: used to test whether feature-map entangling
# capacity helps on real 3-D medical data, not only on the synthetic case.


def _pool_grid_3d(volumes: np.ndarray, grid: int = 2) -> np.ndarray:
    """Block-average a batch of (N, D, H, W) volumes to a (N, grid**3) vector."""
    n, d, h, w = volumes.shape
    bd, bh, bw = d // grid, h // grid, w // grid
    trimmed = volumes[:, : bd * grid, : bh * grid, : bw * grid]
    pooled = trimmed.reshape(n, grid, bd, grid, bh, grid, bw).mean(axis=(2, 4, 6))
    return pooled.reshape(n, grid**3)


def load_nodulemnist3d(
    grid: int = 2,
    n_per_class_train: int = 150,
    n_per_class_test: int = 60,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Class-balanced NoduleMNIST3D subset, pooled to ``grid**3`` features in [0, 1].

    ``grid=2`` gives 8 features (a 2x2x2 voxel grid), matching an 8-qubit
    feature map; the flattened, full-volume arrays are also returned for the
    classical raw-voxel reference baseline.
    """
    from medmnist import NoduleMNIST3D

    train_ds = NoduleMNIST3D(split="train", download=True)
    test_ds = NoduleMNIST3D(split="test", download=True)

    x_train_v = train_ds.imgs.astype(np.float64) / 255.0
    y_train_all = train_ds.labels.ravel().astype(np.int64)
    x_test_v = test_ds.imgs.astype(np.float64) / 255.0
    y_test_all = test_ds.labels.ravel().astype(np.int64)

    rng = np.random.default_rng(seed)

    def _balanced_idx(y, n_per_class):
        idx_all = []
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            idx_all.append(idx_c[:n_per_class])
        idx = np.concatenate(idx_all)
        rng.shuffle(idx)
        return idx

    tr_idx = _balanced_idx(y_train_all, n_per_class_train)
    te_idx = _balanced_idx(y_test_all, n_per_class_test)

    return {
        "x_train": _pool_grid_3d(x_train_v[tr_idx], grid),
        "y_train": y_train_all[tr_idx],
        "x_test": _pool_grid_3d(x_test_v[te_idx], grid),
        "y_test": y_test_all[te_idx],
        "x_train_full": x_train_v[tr_idx].reshape(len(tr_idx), -1),
        "x_test_full": x_test_v[te_idx].reshape(len(te_idx), -1),
    }


if __name__ == "__main__":
    m = load_mnist1d()
    print("MNIST1D:", {k: v.shape for k, v in m.items()})
    e = load_ecg_beats()
    print("ECG beats:", {k: v.shape for k, v in e.items()})
    print("ECG train label balance:", np.bincount(e["y_train"]))
    print("ECG test label balance:", np.bincount(e["y_test"]))
    s = load_eeg_seizure()
    print("EEG seizure:", {k: v.shape for k, v in s.items()})
    print("EEG seizure train label balance:", np.bincount(s["y_train"]))
    print("EEG seizure test label balance:", np.bincount(s["y_test"]))
    p = load_pneumoniamnist()
    print("PneumoniaMNIST:", {k: v.shape for k, v in p.items()})
    print("PneumoniaMNIST train label balance:", np.bincount(p["y_train"]))
    print("PneumoniaMNIST test label balance:", np.bincount(p["y_test"]))
