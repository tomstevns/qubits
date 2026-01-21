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
cd experiments/v0_7
python -X utf8 -u v0_7_convergence_families_v23.py --help
