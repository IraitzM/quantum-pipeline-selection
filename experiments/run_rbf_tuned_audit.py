"""Refit every RBF-SVM cell the paper reports with the 11-point gamma grid.

The paper states that the RBF-SVM baseline searches gamma over an 11-point
geometric grid centred on scikit-learn's ``scale`` heuristic
(``code.baselines.gamma_grid``). The ECG and EEG classification tables were
produced that way, but several other cells came from scripts that call
``SVC(kernel="rbf", gamma="scale")`` directly, or were run before
``rbf_svm_scores`` gained its grid. This script rebuilds each of those data draws
with the original script's own loaders and refits only the classical arm, twice:
once with ``gamma="scale"`` to confirm the draw reproduces the archived number,
and once with the grid. No quantum model is retrained, and nothing under an
existing results directory is overwritten.

Cells covered, with the script that produced the archived value:

  * mnist1d_seeds     ``run_classification_seeds.py --dataset mnist1d``
  * mnist1d_overtrain ``run_mnist1d_overtrain.py`` (StandardScaler, 5 seeds)
  * hls_<rep>         ``run_hls_cmds.py`` (waveform and spectral, 5 seeds)
  * pneumonia_sim     ``run_pneumonia_qsm_sim.py`` (200/75 per class, 5 seeds)
  * eeg_robustness    ``run_eeg_qsm_robustness.py`` (z-scored, recording split)
  * eeg_amp_<feat>    ``run_eeg_amplitude_control.py`` (raw window, log std)
  * nodule_<feat>     ``run_3d_entangling_capacity_seeds.py`` (pooled, raw volume)

Output: ``results/qnn-rbf-audit/table_rbf_audit.csv`` (per seed) and
``table_rbf_audit_summary.csv``.

    uv run experiments/run_rbf_tuned_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)  # local package shadows stdlib ``code``

from code.baselines import rbf_svm_scores  # noqa: E402
from code.datasets import (  # noqa: E402
    load_eeg_seizure,
    load_hls_cmds,
    load_mnist1d,
    load_nodulemnist3d,
    load_pneumoniamnist,
)

OUT_DIR = ROOT / "results" / "qnn-rbf-audit"
SEEDS5 = [42, 43, 44, 45, 46]


def _pair(x_tr, y_tr, x_te, y_te, seed):
    """Test balanced accuracy and selected gamma, untuned then tuned."""
    u = rbf_svm_scores(x_tr, y_tr, x_te, y_te, tune=False)
    t = rbf_svm_scores(x_tr, y_tr, x_te, y_te, tune=True, seed=seed)
    return dict(scale_bacc=u["test_bacc"], tuned_bacc=t["test_bacc"],
                scale_gamma=u["gamma"], tuned_gamma=t["gamma"])


def cells():
    from experiments.run_classification_seeds import build_mnist1d
    for seed in [42, 43, 44]:
        x_tr, y_tr, x_te, y_te, _ = build_mnist1d(seed)
        yield "mnist1d_seeds", seed, _pair(x_tr, y_tr, x_te, y_te, seed)

    from experiments import run_mnist1d_overtrain as ot
    m = load_mnist1d()
    trm, tem = np.isin(m["y_train"], ot.KEEP_CLASSES), np.isin(m["y_test"], ot.KEEP_CLASSES)
    x_te, y_te = ot.subsample(m["x_test"][tem], m["y_test"][tem], ot.N_PER_CLASS_TEST, seed=42)
    for seed in SEEDS5:
        x_all, y_all = ot.subsample(m["x_train"][trm], m["y_train"][trm],
                                    ot.N_PER_CLASS_TRAIN, seed)
        x_tr, y_tr = x_all[len(y_all) // 5:], y_all[len(y_all) // 5:]
        sc = StandardScaler().fit(x_tr)
        yield "mnist1d_overtrain", seed, _pair(sc.transform(x_tr), y_tr,
                                               sc.transform(x_te), y_te, seed)

    from experiments.run_hls_cmds import D
    for rep in ["waveform", "spectral"]:
        for seed in SEEDS5:
            d = load_hls_cmds(target_len=D, representation=rep, seed=seed)
            sc = StandardScaler().fit(d["x_train"])
            yield f"hls_{rep}", seed, _pair(sc.transform(d["x_train"]), d["y_train"],
                                            sc.transform(d["x_test"]), d["y_test"], seed)

    for seed in SEEDS5:
        d = load_pneumoniamnist(n_per_class_train=200, n_per_class_test=75, seed=seed)
        x_tr = d["x_train"].reshape(len(d["x_train"]), -1)
        x_te = d["x_test"].reshape(len(d["x_test"]), -1)
        yield "pneumonia_sim", seed, _pair(x_tr, d["y_train"], x_te, d["y_test"], seed)

    for seed in SEEDS5:
        d = load_eeg_seizure(n_per_class_train=200, n_per_class_test=100,
                             seed=seed, split="recording")
        yield "eeg_robustness", seed, _pair(d["x_train"], d["y_train"],
                                            d["x_test"], d["y_test"], seed)

    from experiments.run_eeg_amplitude_control import raw_recording_split
    for seed in SEEDS5:
        xtr, ytr, xte, yte = raw_recording_split(seed)
        yield "eeg_amp_raw", seed, _pair(xtr, ytr, xte, yte, seed)
        ltr = np.log(xtr.std(axis=1, keepdims=True) + 1e-8)
        lte = np.log(xte.std(axis=1, keepdims=True) + 1e-8)
        yield "eeg_amp_logstd", seed, _pair(ltr, ytr, lte, yte, seed)

    for seed in SEEDS5:
        d = load_nodulemnist3d(grid=2, n_per_class_train=150, n_per_class_test=60, seed=seed)
        yield "nodule_pooled", seed, _pair(d["x_train"], d["y_train"],
                                           d["x_test"], d["y_test"], seed)
        yield "nodule_raw", seed, _pair(d["x_train_full"], d["y_train"],
                                        d["x_test_full"], d["y_test"], seed)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell, seed, res in cells():
        rows.append(dict(cell=cell, seed=seed, **res))
        print(f"  {cell:18s} seed {seed}: scale {res['scale_bacc']:.4f}  "
              f"tuned {res['tuned_bacc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "table_rbf_audit.csv", index=False)

    df = pd.DataFrame(rows)
    df["delta"] = df["tuned_bacc"] - df["scale_bacc"]
    g = df.groupby("cell", sort=False)
    summ = pd.DataFrame({
        "n_seeds": g.size(),
        "scale_mean": g["scale_bacc"].mean(), "scale_std": g["scale_bacc"].std(),
        "tuned_mean": g["tuned_bacc"].mean(), "tuned_std": g["tuned_bacc"].std(),
        "delta_mean": g["delta"].mean(),
        "delta_sem": g["delta"].std() / np.sqrt(g.size()),
    }).round(4)
    summ.to_csv(OUT_DIR / "table_rbf_audit_summary.csv")
    print("\n" + summ.to_string())


if __name__ == "__main__":
    main()
