# AI-Driven Discovery of Physics-Native Quantum Information Units

# Soft Spaces

## Exploring Hidden Structures in Low-Qubit Hilbert Spaces

This repository accompanies the **Phase 1** publication of the *Soft Spaces* project.

The work investigates whether hidden structures can be identified in low-qubit Hilbert spaces through software-based analysis of open quantum systems.

The study focuses on:

- REAL–NULL subspace separation
- Dynamical subspace stability
- Lindblad-based evolution
- Statistical reproducibility across repeated simulations

## Publication

The official archived publication is available on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.21819815

## Repository contents

- Research paper (PDF)
- Supporting documentation
- Project material

## Status

**Phase 1 completed**

Future work will investigate:

- Scaling to larger Hilbert spaces (9–12 qubits)
- Comparison with quantum-computing implementations (Qiskit)
- Further characterization of Soft Spaces

---

Author: **Tom Stevns**

ORCID: https://orcid.org/0009-0000-5306-7776



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



---
