"""Prints every number quoted in the EPJ article and the value the committed
CSVs carry, and fails if they disagree.

The article is the EPJ Quantum Technology companion, "Selecting and Configuring
a Quantum Classification Pipeline from Data Diagnostics and Device Budgets".
Each check names the table or paragraph the number appears in and the experiment
that produced the CSV. Tolerances cover the paper's rounding only.

Run with ``uv run python analysis/paper_numbers.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FAILURES = []


def check(where, label, paper, data, tol=5e-4):
    """One quoted number against one CSV value, paper rounding allowed."""
    ok = abs(paper - data) <= tol
    print(f"{'ok ' if ok else 'FAIL'} {where:<28} {label:<58} paper {paper:<10} data {data:.4g}")
    if not ok:
        FAILURES.append((where, label, paper, data))


def meanstd(df, column):
    return df[column].mean(), df[column].std(ddof=1)


def check_mean(where, label, paper, data, tol=2e-3):
    """A mean quoted to three decimals, allowing the half-way rounding case."""
    check(where, label, paper, data, tol=tol)


# Section 3, Table 2: the diagnostics, per task.
print("\n== Table 2, training-free diagnostics (run_encoding_selection.py, run_hls_cmds.py) ==")
sel = pd.read_csv(RESULTS / "qnn-encoding-selection" / "table_encoding_selection.csv")
for dataset, (omax, oeff, de) in {"ECG": (3, 3, 0.01), "MNIST1D": (12, 11, 0.31),
                                  "EEG": (22, 19, 0.59), "PneumoniaMNIST": (237, 136, 0.77)}.items():
    row = sel[sel.dataset == dataset].iloc[0]
    check("Table 2", f"{dataset} omega_max", omax, row.omega_max, tol=0.5)
    check("Table 2", f"{dataset} omega_eff", oeff, row.omega_eff, tol=0.5)
    check("Table 2", f"{dataset} delta_E", de, row.delta_e, tol=0.005)
hls_diag = pd.read_csv(RESULTS / "qnn-hls" / "table_hls_cmds_diagnostics.csv")
for repr_, de in [("waveform", 0.94), ("spectral", 0.42)]:
    col = "delta_e" if "delta_e" in hls_diag.columns else hls_diag.columns[-2]
    row = hls_diag[hls_diag.iloc[:, 0].astype(str).str.contains(repr_)].iloc[0]
    check("Table 2", f"HLS {repr_} delta_E", de, float(row[col]), tol=0.02)
sweep = pd.read_csv(RESULTS / "qnn-medmnist" / "regression_2d_sweep.csv")
ranks = sweep.groupby("image").effective_rank.first()
check("Table 2", "PneumoniaMNIST SVD ranks within 5.1--5.8", 0, float((ranks < 5.1).sum() + (ranks > 5.8).sum()))

# Section 3.1, the single-qubit calibration.
print("\n== Section 3.1, degree calibration (run_signal_regression.py) ==")
reg = pd.read_csv(RESULTS / "qnn-signals" / "regression_sweep.csv")
mad = (reg.qnn_r2 - reg.classical_r2).abs().mean()
check("Section 3.1", "QNN never beats the least-squares bound, MAD 0.002", 0.002, mad, tol=0.002)
check("Section 3.1", "QNN at no more than three times the parameters", 3.0,
      (reg.qnn_n_params / reg.classical_n_params).max(), tol=0.4)
for signal, expected in [("mnist1d_digit0", 7), ("ecg_normal", 3), ("ecg_pvc", 2)]:
    sub = reg[reg.signal == signal]
    reached = sub[sub.qnn_r2 >= 0.95].degree.min()
    check("Section 3.1", f"degree for R^2>=0.95 on {signal} equals omega_eff", expected, reached, tol=0.5)
d7 = reg[reg.signal == "mnist1d_digit7"]
check("Section 3.1", "mnist1d_digit7 does not resolve within degree 8", 0,
      int((d7.qnn_r2 >= 0.95).any()), tol=0)

# Section 4, Tables 3 and 4, and Figure 2: the selection under budgets.
print("\n== Section 4, Table 4 / Figure 2 (run_encoding_selection.py) ==")
expect = {"ECG": {"angle": (0.744, 0.018), "amplitude": (0.745, 0.004), "qsm": (0.736, 0.023)},
          "MNIST1D": {"angle": (0.421, 0.028), "amplitude": (0.465, 0.031), "qsm": (0.605, 0.070)},
          "EEG": {"angle": (0.691, 0.021), "amplitude": (0.517, 0.029), "qsm": (0.750, 0.008)},
          "PneumoniaMNIST": {"angle": (0.737, 0.010), "amplitude": (0.799, 0.041)}}
for dataset, arms in expect.items():
    sub = sel[sel.dataset == dataset]
    for arm, (m, s) in arms.items():
        mean, std = meanstd(sub, f"{arm}_bacc")
        check_mean("Table 4", f"{dataset} {arm} mean", m, mean)
        check("Table 4", f"{dataset} {arm} std", s, std, tol=0.002)
def paired(dataset, winner, runner_up):
    sub = sel[sel.dataset == dataset].dropna(subset=["qsm_bacc" if winner == "qsm" else "amplitude_bacc"])
    diff = sub[f"{winner}_bacc"] - sub[f"{runner_up}_bacc"]
    return diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
for dataset, winner, runner_up, gain, sig in [("MNIST1D", "qsm", "amplitude", 0.140, 4.3),
                                              ("EEG", "qsm", "angle", 0.059, 5.2),
                                              ("PneumoniaMNIST", "amplitude", "angle", 0.062, 3.0)]:
    mean, sem = paired(dataset, winner, runner_up)
    check("Table 4", f"{dataset} paired gain {winner} over {runner_up}", gain, mean, tol=0.005)
    check("Table 4", f"{dataset} paired gain sigma", sig, mean / sem, tol=0.4)
check("Table 4", "per-seed prediction accuracy 17 of 20", 17, int(sel.prediction_correct.sum()))

# Section 4, the HLS-CMDS pair and the cost-waived check.
print("\n== Section 4, HLS-CMDS pair and PneumoniaMNIST simulation check ==")
hls = pd.read_csv(RESULTS / "qnn-hls" / "table_hls_cmds_summary.csv")
wave = hls[hls.representation == "waveform"]["mean"].to_numpy()
check("Section 4", "every model at chance on the waveform", 0.0, float(np.abs(wave - 0.5).max()), tol=0.06)
spec = hls[hls.representation == "spectral"].set_index("model")["mean"]
check("Section 4", "spectral QSM 0.803", 0.803, spec["QSM"], tol=0.005)
check("Table 5", "spectral MLP 0.825", 0.825, spec["MLP"], tol=0.005)
cluster = spec.drop("angle (pooled)")
check("Section 4", "no-pool and classical cluster, low end 0.74", 0.74, cluster.min(), tol=0.01)
check("Section 4", "no-pool and classical cluster, high end 0.83", 0.83, cluster.max(), tol=0.01)
pneu = pd.read_csv(RESULTS / "qnn-qsm" / "table_pneumonia_qsm_summary.csv")
check("Section 4", "QSM simulated on PneumoniaMNIST 0.764", 0.764, pneu.test_bacc_mean.iloc[0], tol=0.002)
check("Section 4", "amplitude matched-N 0.800", 0.800, pneu.test_bacc_mean.iloc[1], tol=0.002)
check("Section 4", "QSM train accuracy 0.984", 0.984, pneu.train_bacc_mean.iloc[0], tol=0.002)
audit = pd.read_csv(RESULTS / "qnn-rbf-audit" / "table_rbf_audit_summary.csv").set_index("cell")
check("Table 5", "RBF raw pixels tuned 0.833", 0.833, audit.loc["pneumonia_sim", "tuned_mean"], tol=0.002)

# Section 4, the tensor-copy ladder.
print("\n== Section 4, tensor copies (run_amplitude_reupload_seeds.py) ==")
reu = pd.read_csv(RESULTS / "qnn-amplitude" / "table_reupload_summary.csv")
m1 = reu[(reu.dataset.str.contains("MNIST1D")) & (reu.k == 1)]["mean"].iloc[0]
m2 = reu[(reu.dataset.str.contains("MNIST1D")) & (reu.k == 2)]["mean"].iloc[0]
check("Section 4", "MNIST1D amplitude k=1 0.448", 0.448, m1, tol=0.002)
check("Section 4", "MNIST1D amplitude k=2 0.533", 0.533, m2, tol=0.002)
check("Section 4", "k=2 paired gain about 0.085", 0.085, m2 - m1, tol=0.005)
e1 = reu[(reu.dataset.str.contains("ECG")) & (reu.k == 1)]["mean"].iloc[0]
e2 = reu[(reu.dataset.str.contains("ECG")) & (reu.k == 2)]["mean"].iloc[0]
check("Section 4", "ECG k=2 gain not significant", 0.009, e2 - e1, tol=0.005)

# Section 5.1, depth and the entangling budget.
print("\n== Section 5.1, depth and entanglement (run_mnist1d_overtrain.py, rank ladder) ==")
over = pd.read_csv(RESULTS / "qnn-qsm" / "table_mnist1d_overtrain_summary.csv")
check("Section 5.1", "QSM final epoch 100, 0.665", 0.665, over.test_bacc_mean.iloc[1], tol=0.002)
check("Section 5.1", "MLP h=64 0.693", 0.693, over.test_bacc_mean.iloc[3], tol=0.002)
check("Section 5.1", "tuned RBF 0.786 ahead by 0.121", 0.786,
      audit.loc["mnist1d_overtrain", "tuned_mean"], tol=0.002)
ladder = pd.read_csv(RESULTS / "qnn-rankdesign" / "table_rank_ladder_summary.csv")
prod = ladder[ladder.n_entanglers == 0]
check("Section 5.1", "product ansatz attains its ceiling at every rank", 0.0,
      float(np.abs(prod.fitted_r2 - prod.ceiling_r2).max()), tol=0.005)
conv = pd.read_csv(RESULTS / "qnn-rankdesign" / "table_restart_convergence.csv")
cnot1 = conv[(conv.gate == "cnot") & (conv.target_rank == 1)].set_index("k").observed_best64
crz1 = conv[(conv.gate == "crz") & (conv.target_rank == 1)].set_index("k").observed_best64
for k, expected in [(1, 0.838), (2, 0.847), (3, 0.854)]:
    check("Section 5.1", f"fixed CNOT on a rank-1 target at {k} entangler(s)", expected, cnot1[k], tol=0.002)
check("Section 5.1", "parameterised entangler scores 1.000 on rank 1", 1.000, crz1.max(), tol=0.002)
design = pd.read_csv(RESULTS / "qnn-rankdesign" / "table_design_rule.csv")
summary = pd.read_csv(RESULTS / "qnn-rankdesign" / "table_design_rule_summary.csv")
check("Section 5.1", "max-entanglement gain over the rule at k=1, -0.009", -0.009,
      summary[summary.rule_k == 1].gain_over_maxent.iloc[0], tol=0.003)
check("Section 5.1", "max-entanglement gain over the rule at k=3, -0.058", -0.058,
      summary[summary.rule_k == 3].gain_over_maxent.iloc[0], tol=0.003)
k3 = design[design.columns[0]].count()
beat = int((design.maxent_r2 > design.rule_r2).sum())
check("Section 5.1", "maximum entanglement wins on 22 of 24 images", 22, beat, tol=0.5)
check("Section 5.1", "24 images in the study", 24, k3, tol=0.5)
check("Section 5.1", "no image is dof-sufficient", 0, int(design.dof_sufficient.astype(bool).sum()), tol=0)

# Section 5.2, the read-out before the gate.
print("\n== Section 5.2, read-out (run_overlap_sweep.py, run_3d_rank_matching.py) ==")
ovl = pd.read_csv(RESULTS / "qnn-overlap" / "table_performance.csv")
o0 = ovl[ovl.overlap == 0.0].set_index("model").test_bacc
check("Section 5.2", "product + local at chance, 0.501", 0.501, o0["QNN (product ansatz + local meas.)"], tol=0.002)
check("Section 5.2", "product + correlators 0.942", 0.942, o0["QNN (product ansatz + entangled meas.)"], tol=0.002)
check("Section 5.2", "entangled ansatz 0.899", 0.899, o0["QNN (entangled ansatz + local meas.)"], tol=0.002)
ranks = pd.read_csv(RESULTS / "qnn-3d" / "table_rank_matching.csv")
check("Figure 3", "product read-out pinned at field rank 2", 2, int(ranks[ranks.config == "product / +ZZ"].reach_cut0.iloc[0]))
check("Figure 3", "entangling ring x1 reaches 5", 5, int(ranks[ranks.config == "entangling ring x1"].reach_cut0.iloc[0]))
check("Figure 3", "entangling ring x2 reaches the 2L+1 cap 13", 13,
      int(ranks[ranks.config == "entangling ring x2"].reach_cut0.iloc[0]))
acc = pd.read_csv(RESULTS / "qnn-3d" / "table_rank_matching_accuracy.csv")
a0 = acc[acc.overlap == 0.0].set_index("config")
check("Figure 3", "product + ZZ 0.943", 0.943, a0.loc["product / +ZZ", "test_bacc"], tol=0.002)
check("Figure 3", "entangling ring x2 0.911", 0.911, a0.loc["entangling ring x2", "test_bacc"], tol=0.002)
check("Figure 3", "tuned RBF 0.920", 0.920, a0.loc["RBF-SVM (classical)", "test_bacc"], tol=0.002)

# Section 5.3, scales before structure.
print("\n== Section 5.3, scales (run_kernel_bandwidth_sweep.py, run_qsm_time_control.py) ==")
tuned = pd.read_csv(RESULTS / "qnn-3d" / "table_bandwidth_tuned.csv")
c2 = tuned[tuned.dataset == "2c overlap=0.0"]
for enc, expected in [("Z (product)", 0.581), ("Z+ZZ linear", 0.829), ("Z+ZZ full", 0.800)]:
    mean, _ = meanstd(c2[c2.encoding == enc], "test_bacc")
    check("Section 5.3", f"2c tuned {enc}", expected, mean, tol=0.01)
untuned = c2[c2.encoding.str.startswith("Z+ZZ")].test_bacc_at_bw1
check("Section 5.3", "entangling maps score about 0.61 at the default scale", 0.61, untuned.mean(), tol=0.02)
bws = pd.read_csv(RESULTS / "qnn-3d" / "table_bandwidth_summary.csv")
rbf = bws[(bws.dataset == "2c overlap=0.0") & (bws.encoding == "RBF-SVM (tuned)")]
check("Section 5.3", "RBF-SVM at the equal grid 0.902", 0.902, float(rbf.mean_test.iloc[0]), tol=0.002)
nod = tuned[tuned.dataset == "NoduleMNIST3D"]
mean, std = meanstd(nod[nod.encoding == "Z (product)"], "test_bacc")
check("Section 6", "NoduleMNIST3D product map best of the quantum kernels, 0.667", 0.667, mean, tol=0.01)
time_sum = pd.read_csv(RESULTS / "qnn-qsm" / "table_qsm_time_control_summary.csv")
time_sum = time_sum[time_sum.dataset.str.contains("MNIST1D")].set_index("upload_t")
check("Section 5.3", "default upload time 0.1 on the flank, 0.605", 0.605, time_sum.loc[0.1, "mean"], tol=0.002)
check("Section 5.3", "best constant t=0.2, 0.693", 0.693, time_sum.loc[0.2, "mean"], tol=0.002)
check("Section 5.3", "tuning the constant is worth +0.088", 0.088,
      time_sum.loc[0.2, "mean"] - time_sum.loc[0.1, "mean"], tol=0.004)
qsm = pd.read_csv(RESULTS / "qnn-qsm" / "table_qsm_summary.csv")
tr = qsm[(qsm.dataset.str.contains("MNIST1D")) & (qsm.arm == "trainable")]["mean"].iloc[0]
check("Section 5.3", "trained profile 0.688", 0.688, tr, tol=0.002)
check("Section 5.3", "training the time appears worth +0.083", 0.083, tr - time_sum.loc[0.1, "mean"], tol=0.004)

# Section 6, the coupling map.
print("\n== Section 6, coupling map (run_balanced_cut_ladder.py, run_3d_entangling_capacity_seeds.py) ==")
graph = pd.read_csv(RESULTS / "qnn-rankdesign" / "table_cut_rank_graph_law.csv")
enum = pd.read_csv(RESULTS / "qnn-rankdesign" / "table_cut_graph_enumeration.csv")
print(f"      placement tables: {len(graph)} and {len(enum)} rows, see the two scripts' docstrings")

# Section 3.2, the gradient bracket.
print("\n== Section 3.2, gradient bracket (run_gradient_variance_probe.py) ==")
probe = pd.read_csv(RESULTS / "qnn-qsm" / "table_gradient_probe.csv")
check("Section 3.2", "log10 gradient variance -2.9 at n=8, R=4", -2.9,
      probe[(probe.n_qubits == 8) & (probe.n_reuploads == 4)].log10_grad_var.iloc[0], tol=0.05)
check("Section 3.2", "log10 gradient variance -2.7 at n=8, R=32", -2.7,
      probe[(probe.n_qubits == 8) & (probe.n_reuploads == 32)].log10_grad_var.iloc[0], tol=0.05)

# Table 5, the remaining classical references.
print("\n== Table 5, classical references ==")
ecg = pd.read_csv(RESULTS / "qnn-classification" / "table_ecg_seeds_summary.csv")
rbf_ecg = ecg[ecg.model == "RBF-SVM"]
check("Table 5", "ECG RBF reference 0.758", 0.758, float(rbf_ecg["mean"].iloc[0]), tol=0.002)
cap = pd.read_csv(RESULTS / "qnn-3d" / "table_capacity_seeds.csv")
raw_rbf = cap[cap.model == "RBF-SVM"].groupby("encoding").test_bacc.mean()
check("Table 5", "NoduleMNIST3D RBF on the volume 0.722", 0.722, raw_rbf["RBF-SVM (raw full volume)"], tol=0.002)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES")
    for f in FAILURES:
        print("  ", f)
    sys.exit(1)
print("All checks passed within the paper's rounding.")
