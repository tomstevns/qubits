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
- **matched controls** (REAL vs spectrum-matched NULL),
- **leakage-aware evaluation** (avoid “coherent-but-leaky” false positives),
- **audit-ready provenance** (run registry, command registry, logs).

---

## What’s new in v0.7

v0.7 focuses on **open-system retention** and “loophole-resistant” benchmarking:

- **Open-system dynamics (Lindblad/GKLS)** with configurable noise models (e.g., dephasing, amplitude damping).
- **Matched-control baselines:** REAL is evaluated against a **spectrum-matched NULL** to reduce confounders.
- **Joint metrics (reported together):**
  - `F_uncond(t)` unconditional retention (primary KPI),
  - `F_cond(t)`   conditional retention (diagnostic KPI),
  - `L(t)`        leakage / out-of-subspace probability (failure-channel KPI).
- **Probe coverage:** canonical probe sets (e.g., ZX) are treated as diagnostic; headline results use **manifold coverage sampling** (e.g., `rand64`) to reduce direction-dependent blind spots.
- **Replicated evidence criteria:** effects are reported across batches under an explicit decision rule (e.g., sign agreement ≥ 2/3 and a practical effect floor).

This reduces two common pitfalls:
1) “coherent-but-leaky” false positives,
2) probe-set blind spots.

---

## Repository structure

- `experiments/`  
  Versioned, reproducible experiment units (code + outputs + logs + reports).  
  v0.7 contains open-system retention runs and provenance artifacts.

- `documentation/`
  Draft manuscripts(OpenSystem_Retention_2D_Subspace_REAL_vs_NULL.pdf) and supplementary materials.

---

## How to run (local)

Typical requirements:
- Python 3.10+
- `numpy`, `scipy`
- optional: `qiskit`, `qiskit-aer` (experiment-dependent)

Example:
```bash
cd experiments/v0_7
python -X utf8 -u v0_7_convergence_families_v23.py --help

Repository/Experiment Status: up to v0.6.1 (December 2025)
Author: Tom Stevns


Important: v0.3–v0.6.1 focus on building and validating the early “discovery + scoring” components in small, controlled prototypes.

What’s New in v0.6 and v0.6.1 (Why this repository is now more than a toy scan)

Earlier prototypes (v0.3–v0.5) focused on:

Finding degeneracies / near-degeneracies in Hamiltonian spectra

Verifying structure in eigenvectors (dominant basis-state patterns)

Testing dynamic stability via leakage-style proxies (local and backend/simulator where available)

v0.6 adds a systematic layer:

It aggregates many candidate near-degenerate pairs across large seed scans

It reduces each candidate into a discrete “signature family” (binned structural descriptors)

It measures how often signature families recur across the scan population

v0.6.1 adds an evidence layer designed to separate repeatable, stability-correlated structure from one-off spectral coincidences:

A small robustness sweep over discretization / thresholds

An enrichment test (stable subset vs overall frequency, with uncertainty)

A hold-out replication split across seed halves (train top signatures on one half, validate on the other)

Together, v0.6–v0.6.1 are intended to convert “interesting examples” into “repeatable classes” under controlled criteria, without changing the underlying discovery logic.

Repository Structure

High-level layout (may expand over time):

whitepaper_v1_1.pdf
The main conceptual whitepaper (motivation + architecture + discussion).

experiments/
Each experiment version is packaged as a small, reproducible unit:

Python source code (.py)

captured console output (*_output.txt)

a report (.pdf or .docx) describing method + results + interpretation

Example structure:

experiments/v0_4/

physics_native_subspace_v0_4.py

v0_4_output.txt

experiment_v0_4_full_report.pdf

experiments/v0_5/

local and backend-oriented variants (if applicable)

output logs

report documenting local-vs-backend consistency

experiments/v0_6/

signature detection across many seeds

fast scan variants / sensitivity variants

output logs showing signature counts

experiments/v0_6_1/

evidence layer: robustness, enrichment, hold-out replication

outputs intended to be scientifically readable (summary-first)

Other supporting documents may appear at repo root (executive summaries, “need-to-know” notes, appendices).

Experiments (Short Timeline)
v0.3 — Seed Scan + Near-Degenerate Pairs (Proof of Concept)

Builds random sparse Pauli-sum Hamiltonians

Exact diagonalization (small n)

Detects degenerate / near-degenerate eigenpairs

Extracts dominant computational-basis amplitudes

Outputs candidate subspaces as structured eigenvector manifolds

v0.4 — Report-Grade Packaging (Reproducible Experiment Unit)

Standardizes experiment packaging:

code + output + report (8-section structure)

Consolidates interpretation: eigenvector structure is not arbitrary noise; it can be systematically extracted and summarized

v0.5 — Dynamic Stability Test (Static Spectrum → Time Evolution / Leakage Proxy)

Selects candidates from spectral findings

Adds lightweight time-evolution testing (Trotterized where appropriate)

Compares local simulation vs backend-style behavior where possible

Introduces the key idea: “stability under dynamics” as a scoring signal

v0.6 — Systematic Signature Detection (Classification Step)

Scans large seed sets

Converts each candidate into a discretized signature key:

spectral proximity

dominant basis-state count

amplitude entropy bins

leakage proxy bins

Aggregates recurring signature families across population

v0.6.1 — Evidence Layer (Robustness + Enrichment + Hold-Out Replication)

Robustness sweep across discretization settings

Enrichment: stable subset vs baseline frequency (with uncertainty)

Hold-out replication: top signatures must survive data split

How to Run (Local)

Typical requirements (varies per experiment):

Python 3.10+ recommended

numpy, scipy

qiskit and optionally qiskit-aer for simulator-based runs

Example (run one experiment locally):

cd experiments/v0_6
python v0_6.py


Outputs are typically written to:

console

and/or *_output.txt in the same experiment folder

Notes on Interpretation

This repository intentionally separates:

Spectral coincidences (degeneracy happens)
from

structured candidate subspaces (eigenvectors show low-entropy basis structure)
from

stability-correlated structure (low leakage proxy + repeatable signature families)

The aim is not to claim that a final “new qubit” has been discovered.
The aim is to produce a reproducible, progressively stronger basis for asking:

What information units does the physics itself prefer?

Files of Interest (Starting Points)

If you are new to the repo, recommended entry points:

whitepaper_v1_1.pdf
Overall concept + framing.

experiments/v0_4/
Clean early “experiment unit” packaging.

experiments/v0_5/
Dynamic stability motivation.

experiments/v0_6/ and experiments/v0_6_1/
Signature families + evidence layer.

Citation

If you reference this work, please cite it as:

Stevns, T. (2025). AI-Driven Discovery of Physics-Native Quantum Information Units.
GitHub repository: https://github.com/tomstevns/qubits

Status

This is an exploratory research project.
The intention is to contribute quietly and constructively to discussions on:

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
