# AI-Driven Discovery of Physics-Native Quantum Information Units

This repository contains code, logs, and manuscript material for the ongoing whitepaper / preprint series:

**AI-Driven Discovery of Physics-Native Quantum Information Units**  
**Status:** up to **v0.7** (Jan 2026)  
**Author:** Tom Stevns

---

## Motivation

This project explores whether **information-preserving structures** (e.g., logical 2D subspaces) can emerge *naturally* from the symmetry, degeneracy, and low-energy structure of physical Hamiltonians—rather than being purely engineered abstractions.

A central hypothesis is that large physical-to-logical overheads in error correction may partly reflect a **mismatch between human-designed encodings** and the **information structures the underlying physics naturally supports**.

The work is designed as a modular, reproducible exploration of that hypothesis, with emphasis on:
- **Matched controls** (REAL vs spectrum-matched NULL)
- **Leakage-aware evaluation** (avoid “coherent-but-leaky” false positives)
- **Audit-ready provenance** (run registry, command registry, logs)

---

## What’s new in v0.7

v0.7 focuses on **open-system retention** and “loophole-resistant” benchmarking:

- **Open-system dynamics (Lindblad/GKLS)** with configurable noise models (e.g., dephasing, amplitude damping)
- **Matched-control baselines:** REAL is evaluated against a **spectrum-matched NULL** to reduce confounders
- **Joint metrics (reported together):**
  - `F_uncond(t)` unconditional retention (primary KPI)
  - `F_cond(t)`   conditional retention (diagnostic KPI)
  - `L(t)`        leakage / out-of-subspace probability (failure-channel KPI)
- **Probe coverage:** canonical probe sets (e.g., ZX) are treated as diagnostic; headline results use **manifold coverage sampling** (e.g., `rand64`) to reduce direction-dependent blind spots
- **Replicated evidence criteria:** effects are reported across batches under an explicit decision rule (e.g., sign agreement ≥ 2/3 and a practical effect floor)

This reduces two common pitfalls:
1) “coherent-but-leaky” false positives  
2) probe-set blind spots

---

## Repository structure

- `experiments/`  
  Versioned, reproducible experiment units (code + outputs + logs + reports).

- `documentation/`  
  Draft manuscripts and supplementary materials (e.g., *OpenSystem_Retention_2D_Subspace_REAL_vs_NULL.pdf*).

Additional supporting documents may also appear at repo root (executive summaries, “need-to-know” notes, appendices).

---

## How to run (local)

Typical requirements:
- Python 3.10+
- `numpy`, `scipy`
- optional: `qiskit`, `qiskit-aer` (experiment-dependent)

Example:
```bash

Version timeline (v0.3 → v0.7)
v0.3 — Seed scan + near-degenerate pairs (proof of concept)

Builds random sparse Pauli-sum Hamiltonians

Exact diagonalization (small n)

Detects degenerate / near-degenerate eigenpairs

Extracts dominant computational-basis amplitudes

Outputs candidate subspaces as structured eigenvector manifolds

v0.4 — Report-grade packaging (reproducible experiment unit)

Standardizes experiment packaging: code + output + report

Consolidates interpretation: eigenvector structure is not arbitrary noise; it can be systematically extracted and summarized

v0.5 — Dynamic stability test (static spectrum → time evolution / leakage proxy)

Selects candidates from spectral findings

Adds lightweight time-evolution testing (Trotterized where appropriate)

Compares local simulation vs backend-style behavior where possible

Introduces “stability under dynamics” as a scoring signal

v0.6 — Systematic signature detection (classification step)

Scans large seed sets

Converts each candidate into a discretized signature key (e.g., spectral proximity, dominant basis-state count, amplitude entropy bins, leakage proxy bins)

Aggregates recurring signature families across the population

v0.6.1 — Evidence layer (robustness + enrichment + hold-out replication)

Robustness sweeps across discretization settings

Enrichment: stable subset vs baseline frequency (with uncertainty)

Hold-out replication: top signatures must survive a data split

v0.7 — Open-system retention (Lindblad/GKLS) with matched controls

Shifts headline evaluation to open-system retention

Uses REAL vs spectrum-matched NULL to reduce confounders

Reports retention jointly with leakage to reduce “loophole wins”

Notes on interpretation

This repository intentionally separates:

Spectral coincidences (degeneracy happens)
from

Structured candidate subspaces (eigenvectors show low-entropy basis structure)
from

Stability-correlated structure (low leakage proxy + repeatable signature families)

The aim is not to claim that a final “new qubit” has been discovered.
The aim is to produce a reproducible, progressively stronger basis for asking:

What information units does the physics itself prefer?

Suggested entry points

If you are new to the repo, recommended entry points:

whitepaper_v1_1.pdf
Overall concept + framing

experiments/v0_4/
Clean early “experiment unit” packaging

experiments/v0_5/
Dynamic stability motivation

experiments/v0_6/ and experiments/v0_6_1/
Signature families + evidence layer

experiments/v0_7/
Open-system retention results and provenance artifacts

Citation

If you reference this work, please cite it as:

Stevns, T. (2025). AI-Driven Discovery of Physics-Native Quantum Information Units.
GitHub repository: https://github.com/tomstevns/qubits

Status

This is an exploratory research project. The intention is to contribute constructively to discussions on:

quantum information encoding

error correction overhead vs physical structure

Hamiltonian-native computation

stability of low-dimensional invariant manifolds

Future updates may include:

curated Hamiltonian sets (beyond random Pauli sums)

more explicit baselines for stability scoring

more principled “interestingness” scoring

tighter statistical controls on signature enrichment

optional backend validation when available

Contact

tomstevns@gmail.com

For discussion or collaboration, please reach out via GitHub or LinkedIn.
cd experiments/v0_7
python -X utf8 -u v0_7_convergence_families_v23.py --help
