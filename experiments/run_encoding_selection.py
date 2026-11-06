"""
run_encoding_selection.py
─────────────────────────
End-to-end test of the encoding-selection decision procedure on the paper's
classification benchmarks.

The procedure is: compute two training-free diagnostics (effective bandwidth
and pooled energy loss), predict which of three encoding families — amplitude
embedding, angle encoding with data re-uploading, or Hankel-lifted QSM —
should win, and then verify the prediction by training all three and checking.

Previous experiments ran each encoding in isolation with different seeds and
protocols, making the cross-encoding comparison semi-paired at best.  This
script runs all three under a *single shared protocol* (same seeds, same
train/test draws, same optimiser settings where applicable), so the only
difference between a row's three numbers is the encoding family.

The decision rule (from the paper's diagnostics) is:

  1.  Compute omega_eff and omega_max on the raw signal.
  2.  Compute DeltaE, the fraction of spectral energy the n_qubit-level
      average pool discards (Eq. pool-out-energy in the paper).
  3.  If DeltaE is small (pool is faithful), angle encoding is viable;
      otherwise the pool is the bottleneck and QSM or amplitude should
      be considered.
  4.  If the difficulty is spectral (high omega_max) and QSM is feasible
      (signal dimension manageable), QSM is predicted to win.
  5.  If omega_max is low or QSM is infeasible (high-dimensional signals
      where Hankel lifting is too expensive), amplitude embedding is
      predicted to win: it avoids the pool bottleneck while remaining
      computationally tractable.

Output:
  results/qnn-encoding-selection/table_encoding_selection.csv   (per-seed)
  results/qnn-encoding-selection/table_encoding_summary.csv     (mean +/- std)
  results/qnn-encoding-selection/figure_encoding_selection.png

Usage:
  uv run experiments/run_encoding_selection.py
  uv run experiments/run_encoding_selection.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)

from code.datasets import (  # noqa: E402
    load_ecg_beats,
    load_eeg_seizure,
    load_mnist1d,
    load_pneumoniamnist,
)
from code.qnn import (  # noqa: E402
    AmplitudeEncodingClassifier,
    AngleEncodingClassifier,
    HankelQSMClassifier,
    hankel_block_unitary,
)

OUT_DIR = ROOT / "results" / "qnn-image-convolution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = OUT_DIR / "table_encoding_selection.csv"
SUMMARY_CSV = OUT_DIR / "table_encoding_summary.csv"

# ── Training hyperparameters ──────────────────────────────────────────────
# Angle and amplitude use the repo-standard settings from
# run_classification_seeds.py.  QSM uses the tuned LR=0.005 and EPOCHS=20
# from run_hankel_qsm.py (LR=0.05 never leaves chance for this model class).

N_QUBITS_ANGLE = 8           # fixed qubit budget for angle encoding
N_REUPLOADS_ANGLE = 4
ANGLE_EPOCHS = 10
ANGLE_LR = 0.05
ANGLE_BATCH = 32

AMPLITUDE_LAYERS = 3
AMPLITUDE_EPOCHS = 10
AMPLITUDE_LR = 0.05
AMPLITUDE_BATCH = 32

QSM_N_UPLOADS = 8
QSM_UPLOAD_T = 0.1
QSM_EPOCHS = 20
QSM_LR = 0.005
QSM_BATCH = 16

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
MNIST1D_CLASSES = [0, 1, 2, 3]
ENERGY_THRESHOLD = 0.05       # epsilon for omega_max / omega_eff
QSM_DIM_LIMIT = 256           # d above which Hankel QSM is too slow to run


# ── Utility ────────────────────────────────────────────────────────────────

def _znorm(a: np.ndarray) -> np.ndarray:
    """Per-sample z-score."""
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)


def subsample(x, y, n_per_class, rng):
    idx_all = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        idx_all.append(idx_c[:n_per_class])
    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return x[idx], y[idx]


# ── FFT diagnostics ───────────────────────────────────────────────────────

def fft_diagnostics(signal_1d: np.ndarray, eps: float = ENERGY_THRESHOLD):
    """Compute omega_max, omega_eff, and per-frequency energy for a single
    1-D signal after z-scoring.

    Returns dict with omega_max, omega_eff, total_power, and the per-frequency
    power array (for computing pooled energy loss later).
    """
    x = (signal_1d - signal_1d.mean()) / (signal_1d.std() + 1e-8)
    coeffs = np.fft.rfft(x)
    power = np.abs(coeffs) ** 2
    total = power.sum() + 1e-12

    # omega_max: highest frequency in the minimal set reaching (1-eps) energy
    sorted_idx = np.argsort(power[1:])[::-1] + 1  # skip DC
    cumulative = 0.0
    selected = set()
    for idx in sorted_idx:
        cumulative += power[idx] / total
        selected.add(idx)
        if cumulative >= 1.0 - eps:
            break
    omega_max = max(selected) if selected else 0

    # omega_eff: narrowest symmetric prefix band holding (1-eps) energy
    prefix = 0.0
    omega_eff = 0
    for w in range(len(power)):
        prefix += power[w] / total
        if prefix >= 1.0 - eps:
            omega_eff = w
            break

    return dict(omega_max=int(omega_max), omega_eff=int(omega_eff),
                total_power=float(total), power=power)


def pooled_energy_loss(signal_1d: np.ndarray, n_pooled: int):
    """Fraction of spectral energy discarded when average-pooling a signal
    from length d to n_pooled bins.

    Computes DeltaE of Eq. pool-out-energy: the fraction of the raw signal's
    DFT energy sitting at frequency indices above the Nyquist of the pooled
    representation. For a representation with n_pooled values the Nyquist is
    floor(n_pooled / 2), so any DFT coefficient at index k > n_pooled/2 in the
    original signal corresponds to structure that cannot survive the pool.
    """
    d = len(signal_1d)
    if d <= n_pooled:
        return 0.0
    x = (signal_1d - signal_1d.mean()) / (signal_1d.std() + 1e-8)
    power = np.abs(np.fft.rfft(x)) ** 2
    total = power.sum() + 1e-12
    nyq = n_pooled // 2
    energy_above = power[nyq + 1:].sum()
    return float(energy_above / total)


def dataset_diagnostics(x: np.ndarray, n_pooled: int, image_shape=None):
    """Compute diagnostics over a representative sample of signals.

    Returns median omega_max, median omega_eff, and mean pooled energy loss.
    """
    if image_shape is not None:
        from code.representation_diagnostics import spatial_image_diagnostics
        return spatial_image_diagnostics(x[:50], image_shape, n_pooled)
    omega_maxes, omega_effs, delta_es = [], [], []
    for i in range(min(len(x), 50)):  # up to 50 samples for speed
        d = fft_diagnostics(x[i])
        omega_maxes.append(d["omega_max"])
        omega_effs.append(d["omega_eff"])
        delta_es.append(pooled_energy_loss(x[i], n_pooled))
    return dict(
        omega_max=int(np.median(omega_maxes)),
        omega_eff=int(np.median(omega_effs)),
        delta_e=float(np.mean(delta_es)),
    )


def predict_encoding(omega_max: int, omega_eff: int, delta_e: float,
                     n_qubits: int, signal_dim: int) -> str:
    """The decision rule. Returns 'amplitude', 'angle', or 'qsm'.

    Logic:
      - If the pool is faithful (delta_e small), angle encoding works.
      - If the pool is lossy:
          - If omega_max is high (spectral difficulty) AND QSM is feasible
            (signal_dim <= QSM_DIM_LIMIT), QSM is predicted to win.
          - Otherwise (omega_max low, or QSM infeasible at this dimension),
            amplitude embedding is predicted to win: it avoids the pool
            bottleneck while remaining computationally tractable.
    """
    pool_lossy = delta_e >= 0.05
    spectral_difficulty = omega_max > n_qubits
    qsm_feasible = signal_dim <= QSM_DIM_LIMIT

    if not pool_lossy:
        return "angle"
    else:
        if spectral_difficulty and qsm_feasible:
            return "qsm"
        else:
            return "amplitude"


# ── Training functions ─────────────────────────────────────────────────────

def train_angle(x_train, y_train, x_test, y_test, n_classes, seed, image_shape=None):
    """Angle encoding using the entangled ansatz + entangled measurement
    variant (the one used in the main classification experiments).  The
    Images use a fixed spatial convolution; signals use the 1D block pool."""
    torch.manual_seed(seed)
    seq_len = x_train.shape[1]
    model = AngleEncodingClassifier(
        seq_len=seq_len, n_qubits=N_QUBITS_ANGLE, n_reuploads=N_REUPLOADS_ANGLE,
        n_classes=n_classes, use_entangling_ansatz=True,
        use_entangling_measurement=True, image_shape=image_shape,
    )
    opt = torch.optim.Adam(model.parameters(), lr=ANGLE_LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, dtype=torch.float64)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = xt.shape[0]
    t0 = time.time()
    for _ in range(ANGLE_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, ANGLE_BATCH):
            idx = perm[start:start + ANGLE_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float64)).argmax(1).numpy()
    return dict(test_bacc=balanced_accuracy_score(y_test, pred),
                train_time_s=round(time.time() - t0, 1))


def train_amplitude(x_train, y_train, x_test, y_test, n_classes, seed):
    """Amplitude embedding on the full, unpooled signal."""
    torch.manual_seed(seed)
    d = x_train.shape[1]
    model = AmplitudeEncodingClassifier(
        input_dim=d, n_layers=AMPLITUDE_LAYERS, n_classes=n_classes,
    )
    opt = torch.optim.Adam(model.parameters(), lr=AMPLITUDE_LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, dtype=torch.float64)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = xt.shape[0]
    t0 = time.time()
    for _ in range(AMPLITUDE_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, AMPLITUDE_BATCH):
            idx = perm[start:start + AMPLITUDE_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float64)).argmax(1).numpy()
    return dict(test_bacc=balanced_accuracy_score(y_test, pred),
                train_time_s=round(time.time() - t0, 1))


def train_qsm(x_train, y_train, x_test, y_test, n_classes, seed):
    """Hankel-lifted QSM on the full, unpooled signal."""
    torch.manual_seed(seed)
    n_qubits = HankelQSMClassifier.n_qubits_for(x_train.shape[1])
    t0 = time.time()
    u_train = torch.tensor(hankel_block_unitary(_znorm(x_train), t=QSM_UPLOAD_T),
                           dtype=torch.complex128)
    u_test = torch.tensor(hankel_block_unitary(_znorm(x_test), t=QSM_UPLOAD_T),
                          dtype=torch.complex128)
    precompute_s = time.time() - t0

    model = HankelQSMClassifier(n_qubits=n_qubits, n_uploads=QSM_N_UPLOADS,
                                n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=QSM_LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.tensor(y_train, dtype=torch.long)
    n = u_train.shape[0]
    t0 = time.time()
    for _ in range(QSM_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, QSM_BATCH):
            idx = perm[start:start + QSM_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(u_train[idx]), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(u_test).argmax(1).numpy()
    return dict(test_bacc=balanced_accuracy_score(y_test, pred),
                n_qubits=n_qubits, precompute_s=round(precompute_s, 1),
                train_time_s=round(time.time() - t0, 1))


def train_rbf(x_train, y_train, x_test, y_test, seed=0):
    """RBF-SVM on standardised features, bandwidth selected on validation.

    Shares ``code.baselines.gamma_grid`` with the classification track, so the
    classical reference searches the same 11 bandwidths everywhere. Selection
    uses a stratified quarter of the training draw and never sees the test set.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from code.baselines import gamma_grid

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_te = scaler.transform(x_test)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_tr, y_train, test_size=0.25, random_state=seed, stratify=y_train)
    best_gamma, best_val = None, -np.inf
    for g in gamma_grid(x_tr):
        val = balanced_accuracy_score(
            y_val, SVC(kernel="rbf", gamma=g).fit(x_fit, y_fit).predict(x_val))
        if val > best_val:
            best_gamma, best_val = float(g), float(val)

    clf = SVC(kernel="rbf", gamma=best_gamma)
    clf.fit(x_tr, y_train)
    pred = clf.predict(x_te)
    return dict(test_bacc=balanced_accuracy_score(y_test, pred), gamma=best_gamma)


# ── Dataset builders (shared test sets) ────────────────────────────────────

def image_shape_for_dataset(name):
    """Explicit geometry: never infer image status from vector length."""
    return (28, 28) if name == "PneumoniaMNIST" else None


def build_datasets(names=None):
    """Yield (name, n_classes, x_tr_full, y_tr_full, n_per_class,
    x_test, y_test, qsm_feasible).

    Images are transported as flat arrays with explicit geometry supplied to
    the native-2D diagnostics and spatial-convolution angle input map.  QSM is skipped when signal_dim > QSM_DIM_LIMIT
    because the Hankel lifting becomes impractical.
    """
    if names is None or "MNIST1D" in names:
        m = load_mnist1d()
        keep = MNIST1D_CLASSES
        trm, tem = np.isin(m["y_train"], keep), np.isin(m["y_test"], keep)
        rng42 = np.random.default_rng(42)
        x_test, y_test = subsample(m["x_test"][tem], m["y_test"][tem], 40, rng42)
        yield ("MNIST1D", 4, m["x_train"][trm], m["y_train"][trm], 150,
               x_test, y_test, True)

    if names is None or "ECG" in names:
        e = load_ecg_beats()
        yield ("ECG", 2, e["x_train"], e["y_train"], 300,
               e["x_test"], e["y_test"], True)

    if names is None or "EEG" in names:
        s = load_eeg_seizure(n_per_class_train=200, n_per_class_test=100, seed=42)
        yield ("EEG", 2, s["x_train"], s["y_train"], 200,
               s["x_test"], s["y_test"], True)

    # Images are transported flat, with explicit geometry passed to the model.
    if names is None or "PneumoniaMNIST" in names:
        p = load_pneumoniamnist()
        x_tr_flat = p["x_train"].reshape(p["x_train"].shape[0], -1)
        x_te_flat = p["x_test"].reshape(p["x_test"].shape[0], -1)
        yield ("PneumoniaMNIST", 2, x_tr_flat, p["y_train"], 300,
               x_te_flat, p["y_test"], False)  # d=784, QSM infeasible



# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global OUT_DIR, RUNS_CSV, SUMMARY_CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--datasets", nargs="+", choices=["ECG", "MNIST1D", "EEG", "PneumoniaMNIST"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--image-pooling", choices=["spatial-conv2d", "raster"], default="spatial-conv2d")
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_CSV = OUT_DIR / "table_encoding_selection.csv"
    SUMMARY_CSV = OUT_DIR / "table_encoding_summary.csv"
    manifest = dict(protocol="spatial-conv2d-v1" if args.image_pooling == "spatial-conv2d" else "legacy-raster",
                    seeds=args.seeds, datasets=args.datasets, image_pooling=args.image_pooling,
                    kernel=[14, 7], stride=[14, 7], padding="valid", output_grid=[2, 4],
                    kernel_trainable=False, normalization="per-sample z-score before convolution",
                    angle_epochs=ANGLE_EPOCHS, angle_lr=ANGLE_LR, angle_batch=ANGLE_BATCH,
                    angle_qubits=N_QUBITS_ANGLE, angle_reuploads=N_REUPLOADS_ANGLE,
                    amplitude_epochs=AMPLITUDE_EPOCHS, amplitude_layers=AMPLITUDE_LAYERS,
                    torch_version=torch.__version__, source_sha256={})
    for path in [Path(__file__), ROOT / "code/qnn.py", ROOT / "code/image_preprocessing.py", ROOT / "code/representation_diagnostics.py", ROOT / "code/datasets.py"]:
        manifest["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT_DIR / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for (name, n_classes, x_tr_full, y_tr_full, n_per,
         x_test, y_test, qsm_feasible) in build_datasets(args.datasets):
        signal_dim = x_tr_full.shape[1]
        image_shape = image_shape_for_dataset(name) if args.image_pooling == "spatial-conv2d" else None
        x_test_z = _znorm(x_test)

        # Diagnostics on the raw (unpooled) signals
        diag = dataset_diagnostics(x_tr_full, N_QUBITS_ANGLE, image_shape)
        prediction = predict_encoding(
            diag["omega_max"], diag["omega_eff"], diag["delta_e"],
            N_QUBITS_ANGLE, signal_dim,
        )

        print(f"\n{'='*60}")
        print(f"  {name}: d={signal_dim}, omega_max={diag['omega_max']}, "
              f"omega_eff={diag['omega_eff']}, "
              f"DeltaE(pool={N_QUBITS_ANGLE})={diag['delta_e']:.3f}")
        print(f"  QSM feasible: {qsm_feasible}")
        print(f"  Predicted best encoding: {prediction}")
        print(f"{'='*60}", flush=True)

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            x_tr, y_tr = subsample(x_tr_full, y_tr_full, n_per, rng)
            x_tr_z = _znorm(x_tr)

            r = dict(dataset=name, seed=seed,
                     omega_max=diag["omega_max"], omega_eff=diag["omega_eff"],
                     delta_e=round(diag["delta_e"], 4),
                     signal_dim=signal_dim, qsm_feasible=qsm_feasible,
                     predicted=prediction,
                     preprocessing="spatial_box_conv2d" if image_shape else "raster_block_mean_1d",
                     spectrum_geometry="native_2d" if image_shape else "signal_coordinate_1d",
                     pool_projection_loss=diag.get("pool_projection_loss", float("nan")),
                     train_input_sha256=hashlib.sha256(np.ascontiguousarray(x_tr_z).tobytes()).hexdigest(),
                     test_input_sha256=hashlib.sha256(np.ascontiguousarray(x_test_z).tobytes()).hexdigest())

            # Angle encoding (model applies internal pool)
            ar = train_angle(x_tr_z, y_tr, x_test_z, y_test, n_classes, seed, image_shape)
            r["angle_bacc"] = round(ar["test_bacc"], 4)
            r["angle_time"] = ar["train_time_s"]
            print(f"  [{seed}] angle: {ar['test_bacc']:.4f} ({ar['train_time_s']:.0f}s)",
                  flush=True)

            # Amplitude embedding (unpooled, non-negative domain)
            from code.qnn import to_nonneg_domain
            x_tr_amp = to_nonneg_domain(x_tr_z)
            x_te_amp = to_nonneg_domain(x_test_z)
            mr = train_amplitude(x_tr_amp, y_tr, x_te_amp, y_test, n_classes, seed)
            r["amplitude_bacc"] = round(mr["test_bacc"], 4)
            r["amplitude_time"] = mr["train_time_s"]
            print(f"  [{seed}] amplitude: {mr['test_bacc']:.4f} ({mr['train_time_s']:.0f}s)",
                  flush=True)

            # QSM (unpooled, Hankel-lifted) — skip if infeasible
            if qsm_feasible:
                qr = train_qsm(x_tr_z, y_tr, x_test_z, y_test, n_classes, seed)
                r["qsm_bacc"] = round(qr["test_bacc"], 4)
                r["qsm_time"] = qr["train_time_s"]
                r["qsm_qubits"] = qr.get("n_qubits", "")
                print(f"  [{seed}] QSM: {qr['test_bacc']:.4f} ({qr['train_time_s']:.0f}s)",
                      flush=True)
            else:
                r["qsm_bacc"] = float("nan")
                r["qsm_time"] = float("nan")
                r["qsm_qubits"] = ""
                print(f"  [{seed}] QSM: skipped (d={signal_dim} > {QSM_DIM_LIMIT})",
                      flush=True)

            # RBF-SVM raw (unpooled) and pooled
            rbfr = train_rbf(x_tr_z, y_tr, x_test_z, y_test, seed=seed)
            r["rbf_raw_bacc"] = round(rbfr["test_bacc"], 4)

            from code.image_preprocessing import angle_input_features
            xt_t = torch.tensor(x_tr_z, dtype=torch.float64)
            xte_t = torch.tensor(x_test_z, dtype=torch.float64)
            x_tr_pooled = angle_input_features(xt_t, N_QUBITS_ANGLE, image_shape).numpy()
            x_te_pooled = angle_input_features(xte_t, N_QUBITS_ANGLE, image_shape).numpy()
            rbfp = train_rbf(x_tr_pooled, y_tr, x_te_pooled, y_test, seed=seed)
            r["rbf_pooled_bacc"] = round(rbfp["test_bacc"], 4)

            # Determine actual winner (among available encodings)
            candidates = [("angle", r["angle_bacc"]),
                          ("amplitude", r["amplitude_bacc"])]
            if qsm_feasible:
                candidates.append(("qsm", r["qsm_bacc"]))
            best = max(candidates, key=lambda t: t[1])
            r["actual_best"] = best[0]
            r["prediction_correct"] = prediction == best[0]

            rows.append(r)
            pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)

    # Summary
    df = pd.DataFrame(rows)
    _summarize(df)
    _plot(df)


def _summarize(df: pd.DataFrame):
    """Per-dataset summary with prediction accuracy."""
    rows = []
    for dataset, sub in df.groupby("dataset"):
        r = dict(dataset=dataset)
        r["omega_max"] = sub["omega_max"].iloc[0]
        r["omega_eff"] = sub["omega_eff"].iloc[0]
        r["delta_e"] = sub["delta_e"].iloc[0]
        r["qsm_feasible"] = sub["qsm_feasible"].iloc[0]
        r["predicted"] = sub["predicted"].iloc[0]
        r["n_seeds"] = len(sub)
        for enc in ["angle", "amplitude", "qsm"]:
            vals = sub[f"{enc}_bacc"].dropna()
            if len(vals) > 0:
                r[f"{enc}_mean"] = round(vals.mean(), 4)
                r[f"{enc}_std"] = round(vals.std(ddof=1), 4) if len(vals) > 1 else 0.0
            else:
                r[f"{enc}_mean"] = float("nan")
                r[f"{enc}_std"] = float("nan")
        r["rbf_raw_mean"] = round(sub["rbf_raw_bacc"].mean(), 4)
        r["rbf_pooled_mean"] = round(sub["rbf_pooled_bacc"].mean(), 4)

        # Which encoding wins on average?
        means = {}
        for enc in ["angle", "amplitude", "qsm"]:
            v = r.get(f"{enc}_mean")
            if v is not None and not np.isnan(v):
                means[enc] = v
        r["actual_best_mean"] = max(means, key=means.get) if means else ""
        r["prediction_correct_mean"] = r["predicted"] == r["actual_best_mean"]

        # Paired advantage of the predicted winner over the runner-up
        pred = r["predicted"]
        others = [e for e in ["angle", "amplitude", "qsm"]
                  if e != pred and e in means]
        if others:
            best_other = max(others, key=lambda e: means[e])
            paired = sub[f"{pred}_bacc"].values - sub[f"{best_other}_bacc"].values
            r["paired_gain"] = round(float(np.nanmean(paired)), 4)
            valid = paired[~np.isnan(paired)]
            if len(valid) > 1:
                sem = valid.std(ddof=1) / np.sqrt(len(valid))
                r["paired_gain_sem"] = round(float(sem), 4)
                r["paired_gain_sigma"] = round(float(np.nanmean(paired)) / sem, 1) if sem > 0 else float("nan")
            else:
                r["paired_gain_sem"] = float("nan")
                r["paired_gain_sigma"] = float("nan")
        else:
            r["paired_gain"] = float("nan")
            r["paired_gain_sem"] = float("nan")
            r["paired_gain_sigma"] = float("nan")

        rows.append(r)

    summ = pd.DataFrame(rows)
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}")
    print(summ.to_string(index=False))

    # Overall prediction rate
    overall = df["prediction_correct"].mean()
    print(f"\nPer-seed prediction accuracy: {overall:.0%} ({df['prediction_correct'].sum()}/{len(df)})")
    by_dataset = df.groupby("dataset")["prediction_correct"].mean()
    for ds, acc in by_dataset.items():
        print(f"  {ds}: {acc:.0%}")


def _plot(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    colours = {"angle": "#2c7fb8", "amplitude": "#d95f02", "qsm": "#1b9e77"}

    for ax, dataset in zip(axes, datasets):
        sub = df[df.dataset == dataset]
        pred = sub["predicted"].iloc[0]
        qsm_avail = sub["qsm_feasible"].iloc[0]

        encodings = ["angle", "amplitude"]
        if qsm_avail:
            encodings.append("qsm")

        positions, labels, data = [], [], []
        for pos, enc in enumerate(encodings):
            vals = sub[f"{enc}_bacc"].dropna().values
            positions.append(pos)
            labels.append(enc.upper() if enc == "qsm" else enc.capitalize())
            data.append(vals)

        # RBF references
        rbf_raw = sub["rbf_raw_bacc"].mean()
        rbf_pool = sub["rbf_pooled_bacc"].mean()

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                       markeredgecolor="black", markersize=6))
        for i, enc in enumerate(encodings):
            bp["boxes"][i].set_facecolor(colours[enc])
            bp["boxes"][i].set_alpha(0.7 if enc == pred else 0.3)
            bp["medians"][i].set_color("black")

        ax.axhline(rbf_raw, color="red", ls="--", lw=1.5, label=f"RBF raw ({rbf_raw:.3f})")
        ax.axhline(rbf_pool, color="red", ls=":", lw=1, alpha=0.5,
                   label=f"RBF pooled ({rbf_pool:.3f})")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        title = (f"{dataset}\n"
                 f"$\\omega_{{\\max}}$={sub['omega_max'].iloc[0]}, "
                 f"$\\Delta E$={sub['delta_e'].iloc[0]:.2f}")
        if not qsm_avail:
            title += "\n(QSM infeasible at $d$=784)"
        title += f"\nPredicted: {pred}"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Balanced accuracy")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="y")

        # Annotate the predicted winner
        if pred in encodings:
            pred_pos = encodings.index(pred)
            pred_data = data[pred_pos]
            if len(pred_data) > 0:
                ax.annotate("predicted", xy=(pred_pos, pred_data.max()),
                            xytext=(pred_pos + 0.3, pred_data.max() + 0.02),
                            fontsize=8, color=colours[pred], fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=colours[pred], lw=1.5))

    fig.suptitle("Encoding selection: diagnostics predict the winner", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_encoding_selection.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'figure_encoding_selection.png'}")


if __name__ == "__main__":
    main()
