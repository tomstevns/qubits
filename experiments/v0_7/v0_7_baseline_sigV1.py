#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_baseline_sigV1.py
Baseline / null-model calibration for the v0_6–v0_6.1 "systematic signature detection" pipeline.

Goal
----
Characterize what "chance structure" looks like under the same pipeline, so that
v0_6.1-style evidence (enrichment + hold-out replication) can be interpreted against a null.

Key idea
--------
We keep the same spectrum E (and thus the same near-degenerate pair selection) but vary the eigenbasis:
  - REAL: use the true eigenvectors of the generated Hamiltonian.
  - BASIS_PERMUTE: apply a fixed random permutation of the computational basis to the eigenbasis.
  - HAAR_BASIS: replace the eigenbasis by a random unitary (Haar-like) basis, keeping the same E.

For each near-degenerate pair (i, j), we build an integer signature key from:
  - dominant basis-state count (union across the two eigenvectors)
  - binned amplitude entropies of each eigenvector
  - binned leakage proxy (a simple, consistent proxy)

Then we compute:
  - enrichment of signature families inside a "stable set" (lowest leakage quantiles)
  - bootstrap confidence intervals for enrichment factors
  - hold-out replication: top families from half A also appear in half B

Designed for local execution (PyCharm, etc.) and writes console output to UTF-8 text file.
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp
except Exception:
    print("ERROR: qiskit is required for this script (qiskit.quantum_info.SparsePauliOp).")
    print("Install/upgrade with: pip install qiskit")
    raise


# -----------------------------
# I/O helpers (tee to file)
# -----------------------------

class Tee:
    """
    Tee stdout to both console and a UTF-8 text file.
    Avoids Windows cp1252 encoding issues by forcing UTF-8.
    """
    def __init__(self, path: str):
        self.path = path
        self._console = sys.stdout
        self._file = open(path, "w", encoding="utf-8", errors="replace", buffering=1)

    def write(self, obj):
        s = str(obj)
        self._console.write(s)
        self._file.write(s)

    def flush(self):
        try:
            self._console.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


# -----------------------------
# Core math utilities
# -----------------------------

def shannon_entropy_bits(p: np.ndarray, eps: float = 1e-15) -> float:
    """Shannon entropy in bits for a probability vector p."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def dominant_count_union(vec_i: np.ndarray, vec_j: np.ndarray, p_thresh: float) -> int:
    """Count basis states whose probability exceeds threshold in either of two vectors."""
    pi = np.abs(vec_i) ** 2
    pj = np.abs(vec_j) ** 2
    mask = (pi >= p_thresh) | (pj >= p_thresh)
    return int(np.sum(mask))


def normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        return psi
    return psi / nrm


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-like random unitary via QR of random complex Gaussian matrix.
    """
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph
    return q


def basis_permutation_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(dim)
    p = np.zeros((dim, dim), dtype=complex)
    for i, j in enumerate(perm):
        p[j, i] = 1.0
    return p


# -----------------------------
# Hamiltonian generator
# -----------------------------

PAULI_CHARS = ["I", "X", "Y", "Z"]

def random_pauli_string(n_qubits: int, rng: np.random.Generator) -> str:
    while True:
        s = "".join(rng.choice(PAULI_CHARS) for _ in range(n_qubits))
        if any(c != "I" for c in s):
            return s


def build_random_hamiltonian_sparse(
    n_qubits: int,
    num_terms: int,
    seed: int,
    coeff_scale: float = 1.0,
) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    paulis = []
    coeffs = []
    for _ in range(num_terms):
        p = random_pauli_string(n_qubits, rng)
        c = coeff_scale * rng.uniform(-1.0, 1.0)
        paulis.append(p)
        coeffs.append(c)
    return SparsePauliOp(paulis, coeffs=np.asarray(coeffs, dtype=complex))


# -----------------------------
# Eigen analysis & pipeline
# -----------------------------

def eigensystem_from_sparsepauliop(H: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (eigenvalues sorted ascending, eigenvectors as columns aligned with eigenvalues).
    """
    mat = np.asarray(H.to_matrix(), dtype=complex)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals.real)
    vals = vals[order].real
    vecs = vecs[:, order]
    return vals, vecs


def neighbor_near_degenerate_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    """Check adjacent eigenvalues only (fast)."""
    pairs = []
    for i in range(len(evals) - 1):
        if abs(evals[i + 1] - evals[i]) < eps:
            pairs.append((i, i + 1))
    return pairs


def leakage_proxy(evecs: np.ndarray, pair: Tuple[int, int]) -> float:
    """
    Consistent leakage proxy for a near-degenerate pair:
      - define psi0 = (phi_i + phi_j)/sqrt(2)
      - leakage = 1 - (|<phi_i|psi0>|^2 + |<phi_j|psi0>|^2)
    For an orthonormal pair, this is exactly 0. Here the value is always 0 by construction,
    so we instead measure how much psi0 lies in the *computational basis support* of the pair:
      - compute dominant-basis union mask (using a fixed threshold)
      - leakage_support = 1 - sum_{x in dominant_union} |psi0_x|^2
    This stays within the spirit of v0_6/v0_6.1: a lightweight stability/structure proxy,
    but now interpretable under basis-randomized nulls.
    """
    i, j = pair
    phi_i = evecs[:, i]
    phi_j = evecs[:, j]
    psi0 = normalize_state(phi_i + phi_j)
    # leakage_support computed later because it depends on dominant mask
    return float(np.nan), psi0


@dataclass(frozen=True)
class SignatureKey:
    dom_count: int
    ent_bin_i: int
    ent_bin_j: int
    leak_bin: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.dom_count, self.ent_bin_i, self.ent_bin_j, self.leak_bin)


@dataclass
class Candidate:
    seed: int
    model: str
    pair: Tuple[int, int]
    delta_E: float
    dom_count: int
    ent_i: float
    ent_j: float
    leakage: float
    sig: SignatureKey


def build_signature(
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    psi0: np.ndarray,
    p_dom_thresh: float,
    ent_bin_width: float,
    leak_bin_width: float,
) -> Tuple[int, float, float, float, SignatureKey]:
    pi = np.abs(vec_i) ** 2
    pj = np.abs(vec_j) ** 2

    dom = dominant_count_union(vec_i, vec_j, p_dom_thresh)
    ent_i = shannon_entropy_bits(pi)
    ent_j = shannon_entropy_bits(pj)

    # leakage as "probability mass of psi0 outside dominant_union"
    mask = (pi >= p_dom_thresh) | (pj >= p_dom_thresh)
    p_in = float(np.sum(np.abs(psi0[mask]) ** 2))
    leakage = max(0.0, 1.0 - p_in)

    ent_bin_i = int(np.floor(ent_i / ent_bin_width + 1e-12))
    ent_bin_j = int(np.floor(ent_j / ent_bin_width + 1e-12))
    # symmetry in i/j
    if ent_bin_j < ent_bin_i:
        ent_bin_i, ent_bin_j = ent_bin_j, ent_bin_i
        ent_i, ent_j = ent_j, ent_i

    leak_bin = int(np.floor(leakage / leak_bin_width + 1e-12))
    key = SignatureKey(int(dom), int(ent_bin_i), int(ent_bin_j), int(leak_bin))
    return int(dom), float(ent_i), float(ent_j), float(leakage), key


# -----------------------------
# Evidence layer (enrichment + bootstrap + hold-out)
# -----------------------------

def define_stable_set(cands: List[Candidate], stable_quantile: float) -> List[Candidate]:
    if not cands:
        return []
    leaks = np.array([c.leakage for c in cands], dtype=float)
    thr = float(np.quantile(leaks, stable_quantile))
    return [c for c in cands if c.leakage <= thr]


def counts_by_signature(cands: List[Candidate]) -> Dict[SignatureKey, int]:
    d: Dict[SignatureKey, int] = {}
    for c in cands:
        d[c.sig] = d.get(c.sig, 0) + 1
    return d


def enrichment_factors(
    cands_all: List[Candidate],
    cands_stable: List[Candidate],
    min_count_overall: int,
    top_k: int,
) -> List[Tuple[SignatureKey, float, int, int]]:
    N_all = len(cands_all)
    N_stable = len(cands_stable)
    if N_all == 0 or N_stable == 0:
        return []

    cnt_all = counts_by_signature(cands_all)
    cnt_st = counts_by_signature(cands_stable)

    rows = []
    for sig, ca in cnt_all.items():
        if ca < min_count_overall:
            continue
        cs = cnt_st.get(sig, 0)
        p_all = ca / N_all
        p_st = cs / N_stable
        ef = (p_st / p_all) if p_all > 0 else float("nan")
        rows.append((sig, float(ef), int(ca), int(cs)))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def bootstrap_enrichment_ci(
    cands_all: List[Candidate],
    stable_quantile: float,
    sig: SignatureKey,
    n_boot: int,
    rng_seed: int = 1234,
) -> Tuple[float, float]:
    if not cands_all:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(rng_seed)
    N = len(cands_all)
    efs = []

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        sample = [cands_all[int(i)] for i in idx]
        stable = define_stable_set(sample, stable_quantile)

        N_all = len(sample)
        N_st = len(stable)
        if N_all == 0 or N_st == 0:
            continue

        ca = sum(1 for c in sample if c.sig == sig)
        if ca == 0:
            continue
        cs = sum(1 for c in stable if c.sig == sig)

        p_all = ca / N_all
        p_st = cs / N_st
        efs.append(p_st / p_all)

    if len(efs) < 30:
        return (float("nan"), float("nan"))
    return (float(np.quantile(efs, 0.025)), float(np.quantile(efs, 0.975)))


def holdout_replication_score(
    cands: List[Candidate],
    stable_quantile: float,
    top_k: int,
    min_count_overall: int,
) -> float:
    if not cands:
        return float("nan")
    seeds = sorted({c.seed for c in cands})
    if len(seeds) < 20:
        return float("nan")

    mid = len(seeds) // 2
    setA = set(seeds[:mid])
    setB = set(seeds[mid:])

    cA = [c for c in cands if c.seed in setA]
    cB = [c for c in cands if c.seed in setB]

    stA = define_stable_set(cA, stable_quantile)
    stB = define_stable_set(cB, stable_quantile)

    topA = enrichment_factors(cA, stA, min_count_overall=min_count_overall, top_k=top_k)
    topB = enrichment_factors(cB, stB, min_count_overall=min_count_overall, top_k=top_k)

    sigA = {row[0] for row in topA}
    sigB = {row[0] for row in topB}
    if not sigA:
        return 0.0
    return float(len(sigA.intersection(sigB)) / len(sigA))


# -----------------------------
# Runner
# -----------------------------

def format_sig(sig: SignatureKey) -> str:
    return f"(dom={sig.dom_count}, ent_bins=({sig.ent_bin_i},{sig.ent_bin_j}), leak_bin={sig.leak_bin})"


def run_model_scan(
    model_name: str,
    seeds: Iterable[int],
    n_qubits: int,
    num_terms: int,
    eps: float,
    p_dom_thresh: float,
    ent_bin_width: float,
    leak_bin_width: float,
    P: Optional[np.ndarray],
    rng_basis: np.random.Generator,
) -> List[Candidate]:
    dim = 2 ** n_qubits
    cands: List[Candidate] = []

    for seed in seeds:
        H = build_random_hamiltonian_sparse(n_qubits=n_qubits, num_terms=num_terms, seed=seed)
        evals, evecs = eigensystem_from_sparsepauliop(H)

        if model_name == "REAL":
            V = evecs
        elif model_name == "BASIS_PERMUTE":
            if P is None:
                raise ValueError("Permutation matrix P required.")
            V = P @ evecs
        elif model_name == "HAAR_BASIS":
            U = random_unitary(dim, rng_basis)
            V = U @ evecs
        else:
            raise ValueError("Unknown model.")

        pairs = neighbor_near_degenerate_pairs(evals, eps)
        if not pairs:
            continue

        for (i, j) in pairs:
            dE = float(abs(evals[j] - evals[i]))
            vec_i = V[:, i]
            vec_j = V[:, j]

            _, psi0 = leakage_proxy(V, (i, j))
            dom, ent_i, ent_j, leakage, sig = build_signature(
                vec_i, vec_j, psi0,
                p_dom_thresh=p_dom_thresh,
                ent_bin_width=ent_bin_width,
                leak_bin_width=leak_bin_width,
            )

            cands.append(Candidate(
                seed=int(seed),
                model=model_name,
                pair=(int(i), int(j)),
                delta_E=dE,
                dom_count=dom,
                ent_i=ent_i,
                ent_j=ent_j,
                leakage=leakage,
                sig=sig,
            ))
    return cands


def main():
    CFG = {
        "n_qubits": 3,
        "num_terms": 6,
        "seeds_scanned": 5000,
        "seed_start": 0,

        "eps": 0.05,

        "p_dom_thresh": 0.05,
        "ent_bin_width": 0.25,
        "leak_bin_width": 0.01,

        "stable_quantile": 0.05,
        "min_count_overall": 20,
        "top_k_signatures": 10,
        "bootstrap_samples": 400,
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_txt = os.path.join(out_dir, "v0_7_baseline_sigV1_output.txt")

    tee = Tee(out_txt)
    sys.stdout = tee

    try:
        print("=== v0_7: Baseline / Null-Model Calibration ===")
        print("Purpose: quantify what 'chance structure' looks like under the same pipeline.")
        print()
        print("Config:")
        print(json.dumps(CFG, indent=2))
        print()

        n_qubits = int(CFG["n_qubits"])
        dim = 2 ** n_qubits
        seeds = list(range(int(CFG["seed_start"]), int(CFG["seed_start"]) + int(CFG["seeds_scanned"])))

        rng_perm = np.random.default_rng(202512)
        P = basis_permutation_matrix(dim, rng_perm)

        rng_basis = np.random.default_rng(9001)
        models = ["REAL", "BASIS_PERMUTE", "HAAR_BASIS"]

        t0 = time.time()

        for model in models:
            print("----------------------------------------------")
            print(f"Model: {model}")
            t_model = time.time()

            cands = run_model_scan(
                model_name=model,
                seeds=seeds,
                n_qubits=n_qubits,
                num_terms=int(CFG["num_terms"]),
                eps=float(CFG["eps"]),
                p_dom_thresh=float(CFG["p_dom_thresh"]),
                ent_bin_width=float(CFG["ent_bin_width"]),
                leak_bin_width=float(CFG["leak_bin_width"]),
                P=P if model == "BASIS_PERMUTE" else None,
                rng_basis=rng_basis,
            )

            print(f"Candidates found (near-degenerate neighbor pairs): {len(cands)}")
            if not cands:
                print("No candidates. Consider increasing eps or seeds_scanned.")
                continue

            stable = define_stable_set(cands, float(CFG["stable_quantile"]))
            print(f"Stable set size (quantile={CFG['stable_quantile']}): {len(stable)}")

            rep = holdout_replication_score(
                cands,
                stable_quantile=float(CFG["stable_quantile"]),
                top_k=int(CFG["top_k_signatures"]),
                min_count_overall=int(CFG["min_count_overall"]),
            )
            print(f"Hold-out replication (overlap fraction): {rep:.3f}")
            print()

            top = enrichment_factors(
                cands_all=cands,
                cands_stable=stable,
                min_count_overall=int(CFG["min_count_overall"]),
                top_k=int(CFG["top_k_signatures"]),
            )

            print("Top enriched signature families:")
            if not top:
                print("  (none passed min_count_overall)")
            else:
                for sig, ef, ca, cs in top:
                    lo, hi = bootstrap_enrichment_ci(
                        cands_all=cands,
                        stable_quantile=float(CFG["stable_quantile"]),
                        sig=sig,
                        n_boot=int(CFG["bootstrap_samples"]),
                        rng_seed=1234,
                    )
                    ci_txt = "CI: n/a" if (math.isnan(lo) or math.isnan(hi)) else f"CI95% [{lo:.2f}, {hi:.2f}]"
                    print(f"  {format_sig(sig)}  EF={ef:.2f}  overall={ca}  stable={cs}  {ci_txt}")

            dt = time.time() - t_model
            print(f"Model runtime: {dt:.1f} s")
            print()

        dt_total = time.time() - t0
        print("----------------------------------------------")
        print(f"Total runtime: {dt_total:.1f} s")
        print()
        print("Interpretation guide:")
        print("  - REAL should show stronger and more consistent enrichment than HAAR_BASIS if structure is non-chance.")
        print("  - BASIS_PERMUTE tests label-invariance (results should be similar to REAL in aggregate).")
        print()
        print("=== End of v0_7 ===")

    finally:
        sys.stdout = tee._console
        tee.close()


if __name__ == "__main__":
    main()
