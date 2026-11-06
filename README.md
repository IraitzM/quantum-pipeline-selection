# Selecting and Configuring a Quantum Classification Pipeline from Data Diagnostics and Device Budgets: code

This repository reproduces every number of the article, from the training-free
diagnostics to the coupling-map placements. It simulates angle encoding with
data re-uploading, amplitude embedding and the quantum spectral model (QSM) on
signal, image and volume tasks, plus the synthetic 2c checkerboard, and it
computes the two measurements the article's procedure rests on, one FFT and one
SVD. Every simulation is noiseless and uses exact expectation values.

It is the companion repository of the Quantum Machine Intelligence article
"How Quantum Data Encoding Constrains Representation Learning", whose code
reproduces the accessibility ladder, the pooling screens, the feature probes and
the regime map that this article cites. The two repositories are deliberately
disjoint in their experiment scripts: what the companion article reports is
reproduced there, not here.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/IraitzM/quantum-pipeline-selection
cd quantum-pipeline-selection
uv sync
```

`uv.lock` pins the library versions that produced the results in `results/`
(PyTorch 2.12.1, PennyLane 0.45.0, scikit-learn 1.9.0, NumPy 2.4.6). Everything
runs on a CPU. The slowest scripts are the QSM experiments on EEG, which take
tens of minutes per seed on a laptop.

## Verifying the article's numbers

The fastest check does not rerun anything. The committed CSVs under `results/`
are the ones the article quotes, and

```bash
uv run python analysis/paper_numbers.py
```

prints every quoted number next to the value the CSV carries and fails if any
disagrees beyond the paper's rounding. To recompute a bundle from the data, run
its script under `experiments/`; each script's docstring names the numbers it
produces and the folder it writes. The scripts write into `results/` and will
overwrite the committed CSVs, so copy them aside first if you want to compare.

## Data

Every dataset is downloaded on first use into `data/` and checked where the
source provides a checksum.

| Task | Source |
|---|---|
| ECG | MIT-BIH Arrhythmia Database 1.0.0, PhysioNet, read with `wfdb` |
| EEG | Bonn recordings in the UCI Epileptic Seizure Recognition release (Hugging Face mirror) |
| MNIST1D | official release (Greydanus and Kobak, 2024) |
| PneumoniaMNIST, NoduleMNIST3D | MedMNIST (28-pixel archives) |
| HLS-CMDS | heart and lung recordings, Zenodo record 15376628 |
| 2c checkerboard | generated on the fly by `kernel_trainer.dataset.DataGenerator`, template `2c` |

PhysioNet can be slow on first use. The 2c generator comes from the companion
package `kernel-trainer` (github.com/IraitzM/kernel-trainer, tag v0.1.9), which
`uv sync` installs from git.

## Layout

```
code/           the library
  datasets.py     downloads and per-experiment data draws, including the 2c generator
  qnn.py          the three encodings as PennyLane classifiers, the QSM uploads, the 2-D regressors
  baselines.py    tuned RBF-SVM (11-point gamma grid), MLPs, Fourier fits, SVD effective rank
  rankdesign.py   the entangler ladder, cut ranks, placement search, controlled-rank targets
  qkernel.py      graded Pauli feature maps, the analytic fidelity Gram matrix, Meyer--Wallach capacity
  image_preprocessing.py  pooling grids for the image and volume tasks
  representation_diagnostics.py  spectral and spatial diagnostics behind Table 2
  dft.py          one-sided DFT helpers for the single-qubit calibration
experiments/    one script per result bundle, with the article's settings fixed inside
analysis/       paper_numbers.py, every quoted number against the committed CSVs
results/        the CSVs the article quotes, one folder per bundle
```

The package is named `code`, which shadows the standard-library module. That is
deliberate, the scripts insert the repository root at the front of `sys.path`
and the working repository this is trimmed from uses the same name.

## Which script produces which number

| Article item | Experiment script(s) | Results folder |
|---|---|---|
| Table 2 diagnostics, Table 4 selection, Figure 2 | `run_encoding_selection.py` | `qnn-encoding-selection` |
| Table 2 SVD ranks, the 2-D regression ceilings | `run_medmnist_regression.py`, `run_medmnist_regression_seeds.py` | `qnn-medmnist` |
| Section 3.1 degree calibration (7, 3, 2) | `run_signal_regression.py` | `qnn-signals` |
| Section 3.2 gradient bracket | `run_gradient_variance_probe.py` | `qnn-qsm` |
| Section 4 HLS-CMDS pair | `run_hls_cmds.py` | `qnn-hls` |
| Section 4 cost-waived PneumoniaMNIST check | `run_pneumonia_qsm_sim.py` | `qnn-qsm` |
| Section 4 tensor copies | `run_amplitude_reupload_seeds.py` | `qnn-amplitude` |
| Section 5.1 prolonged training | `run_mnist1d_overtrain.py` | `qnn-qsm` |
| Section 5.1 rank ladder and ceilings | `run_rank_ladder_validation.py` | `qnn-rankdesign` |
| Section 5.1 CNOT versus parameterised entangler | `run_restart_convergence.py`, `run_cnot_violation_probe.py` | `qnn-rankdesign` |
| Section 5.1 the 24-image budget caveat | `run_design_rule_end_to_end.py` | `qnn-rankdesign` |
| Section 5.2 checkerboard stress test | `run_overlap_sweep.py` | `qnn-overlap` |
| Section 5.2 and Figure 3, three-qubit rank matching | `run_3d_rank_matching.py` | `qnn-3d` |
| Section 5.3 input-scale sweep, tuned RBF | `run_kernel_bandwidth_sweep.py` | `qnn-3d` |
| Section 5.3 upload-time sweep and trained arm | `run_qsm_time_control.py`, `run_hankel_qsm.py` | `qnn-qsm` |
| Section 6 linear versus all-to-all, NoduleMNIST3D | `run_3d_entangling_capacity_seeds.py` | `qnn-3d` |
| Section 6 cut-locality and placement | `run_balanced_cut_ladder.py`, `run_cut_rank_graph_law.py`, `run_cut_graph_enumeration.py` | `qnn-rankdesign` |
| Table 5 ECG classical reference | `run_classification_seeds.py` | `qnn-classification` |
| Table 5 remaining tuned RBF cells | `run_rbf_tuned_audit.py` | `qnn-rbf-audit` |

Two details are documented in the runners rather than in a results table. The
learning-rate screening for the QSM (0.05 oscillates, 0.005 converges) is
recorded in `run_hankel_qsm.py`, and the restart shares behind "no entangled
configuration above rank one converged from more than four per cent of starts"
are computed inside `run_restart_convergence.py`.

## Differences from the working repository

This is a trimmed standalone copy of the MLST-era code behind the archived
manuscript in `archive/paper/`, not of the QMI companion `qenc` library. The
three DFT helpers of `code/dft.py` are extracted verbatim from the working
repository's `code/main.py`, and `run_signal_regression.py` imports them from
there. `code/__init__.py` is emptied so importing the package does not pull in
modules this repository does not ship. No experiment script, model or protocol
is otherwise modified.
