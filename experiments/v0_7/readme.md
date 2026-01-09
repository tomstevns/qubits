# v0_7 experiments (REAL vs spectrum-matched NULL)

This folder contains the experimental artifacts for **v0_7** of the project:
**REAL vs spectrum-matched NULL** (logical retention under open-system Lindblad noise, `n_qubits=4`).

Primary purpose:
- provide a **reproducible audit trail** from manuscript tables/claims to concrete run output files, and
- enable independent verification via **SHA256 manifests**.

---

## Contents (recommended structure)

- `scripts/`
  - Python scripts used to generate the results (e.g. `v0_7_convergence_families_v23*.py`)
- `outputs/`
  - run outputs referenced in the manuscript (`v23_*.txt`)
- `logs/`
  - console logs for the same runs (`v23_*.log`)
- `manifests/`
  - file manifests and cross-reference tables
  - `run_manifest_sha256.csv`
  - `run_crossref_mapping*.csv` / `.xlsx`

If your repository is kept flat (no subfolders), the same filenames still apply.

---

## Key result files referenced by the manuscript (headline set)

These files are referenced by **original output filename** (for direct lookup on GitHub):

### Amplitude damping (rand64 manifold probing; subset qubit = 3)
- `v23_ampd_g10_subset3_noise_diag_p400_rand64.txt`
- `v23_ampd_g10_subset3_eigen_p400_rand64.txt`
- `v23_ampd_g05_subset3_noise_diag_p400_rand64.txt`
- `v23_ampd_g02_subset3_noise_diag_p400_rand64.txt`

### Dephasing-only (rand64; subset qubit = 3)
- `v23_deph_gphi10_subset3_noise_diag_p400_rand64.txt`
- `v23_deph_gphi20_subset3_eigen_p400_rand64.txt`

### Mixed noise (“both”; rand64; subset qubit = 3)
- `v23_both_gphi10_g02_subset3_eigen_p400_rand64.txt`

### Diagnostic baseline (ZX probes)
- `v23_ampd_g10_subset3_noise_diag_p400.txt`  *(states=zx; screening/diagnostic)*

---

## Probe-state regimes (ZX vs rand64)

Two probe-state regimes are used:

- `states=zx`  
  Four canonical probe states (fast screening/diagnostic; can miss direction-dependent effects).

- `states=rand64`  
  Manifold probing using many more directions (≈66 probes total), used for **headline results**.

Unless otherwise stated in the manuscript, **main tables/claims are based on `rand64` runs**; ZX runs are provided as diagnostics.

---

## Reproducibility (Windows / PowerShell)

Typical run settings (as used in the manuscript):
- `--n_qubits 4`
- `--seeds_per_batch 5000`
- `
::contentReference[oaicite:0]{index=0}
readme.md
