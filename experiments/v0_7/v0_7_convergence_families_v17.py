#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v17.py

Purpose
-------
Generalization step from v15 to n_qubits=4 (d=16) as a stronger evidence test:
- Keeps the fully reproducible, statistical–numerical protocol (no model training).
- Retains REAL vs spectrum-matched Haar NULL baseline.
- Keeps fixed-fraction "stable" selection per model (v15), optionally with leakage constraints.
- Retains fine + coarse quantile signatures and paired reporting (with optional NN fallback).

Key refinements vs v15
----------------------
1) Default n_qubits=4 (can be overridden).
2) Precomputes all n-qubit Pauli operators (excluding all-I) once and reuses them.
   This avoids repeated kronecker products inside the tight loop and helps keep runtime reasonable.
3) Makes coarse dom-binning dimension-aware (uses dom/d ratio bins), so it scales with d.

Dependencies
------------
Only numpy (no Qiskit, no SciPy).

Notes on scaling
----------------
- For n_qubits=4, entropy ranges up to log2(d)=4 bits, so absolute medians shift.
  Interpretations rely on REAL–NULL deltas/effect sizes and batch convergence, not raw levels.
"""

from __future__ import annotations

import argparse
import math
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional
from collections import Counter
from itertools import product

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def stable_hash_int(s: str) -> int:
    """Deterministic int hash independent of Python's hash randomization."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def bin_index(x: float, step: float) -> int:
    if step <= 0:
        raise ValueError("step must be > 0")
    return int(math.floor(float(x) / float(step) + 1e-12))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def make_quantile_edges(values: np.ndarray, q_bins: int) -> np.ndarray:
    """
    Quantile edges for q_bins categories.
    Returns length q_bins+1, with edges[0]=-inf and edges[-1]=+inf.
    """
    if q_bins < 2:
        raise ValueError("q_bins must be >= 2")
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        edges = np.linspace(0.0, 1.0, q_bins + 1)
    else:
        qs = np.linspace(0.0, 1.0, q_bins + 1)
        try:
            edges = np.quantile(v, qs, method="linear")
        except TypeError:
            edges = np.quantile(v, qs)
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.maximum.accumulate(edges)
    return edges


def quantile_bin(x: float, edges: np.ndarray, q_bins: int) -> int:
    idx = int(np.searchsorted(edges, float(x), side="right") - 1)
    return clamp_int(idx, 0, q_bins - 1)


def safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.median(x))


# ----------------------------
# Exact binomial tail (no SciPy)
# ----------------------------

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    p = float(p)
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return _log_choose(n, k) + k * math.log(p) + (n - k) * math.log(1.0 - p)


def _logsumexp(log_terms: List[float]) -> float:
    m = max(log_terms)
    if m == -math.inf:
        return -math.inf
    s = sum(math.exp(t - m) for t in log_terms)
    return m + math.log(s)


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """Exact tail probability P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_terms = [_log_binom_pmf(x, n, p) for x in range(k, n + 1)]
    return float(math.exp(_logsumexp(log_terms)))


# ----------------------------
# Pauli operator cache (dimension dependent)
# ----------------------------

PAULIS = ["I", "X", "Y", "Z"]
PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def kron_n(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def precompute_pauli_ops(n_qubits: int) -> Tuple[List[str], List[np.ndarray]]:
    """
    Precompute all n-qubit Pauli operators (excluding all-I) in a fixed order.
    Returned labels are strings like "IXYZ". Operators are dense (d x d) complex arrays.
    """
    labels: List[str] = []
    ops: List[np.ndarray] = []
    for tup in product(PAULIS, repeat=n_qubits):
        if all(p == "I" for p in tup):
            continue
        lab = "".join(tup)
        labels.append(lab)
        ops.append(kron_n([PAULI_MATS[p] for p in tup]))
    return labels, ops


@dataclass(frozen=True)
class PauliCache:
    n_qubits: int
    d: int
    labels: List[str]
    ops: List[np.ndarray]

    @staticmethod
    def build(n_qubits: int) -> "PauliCache":
        d = 2 ** int(n_qubits)
        labels, ops = precompute_pauli_ops(int(n_qubits))
        return PauliCache(n_qubits=int(n_qubits), d=d, labels=labels, ops=ops)


def build_random_pauli_hamiltonian_cached(cache: PauliCache, n_terms: int, seed: int) -> np.ndarray:
    rng = rng_from_seed(seed)
    d = cache.d
    H = np.zeros((d, d), dtype=complex)

    m = len(cache.ops)
    # Sample with replacement; coefficients uniform in [-1, 1]
    idxs = rng.integers(0, m, size=int(n_terms))
    coeffs = rng.uniform(-1.0, 1.0, size=int(n_terms)).astype(float)
    for k in range(int(n_terms)):
        H = H + float(coeffs[k]) * cache.ops[int(idxs[k])]

    H = 0.5 * (H + H.conj().T)
    return H


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph
    return Q


def null_haar_basis_hamiltonian(evals: np.ndarray, seed: int) -> np.ndarray:
    rng = rng_from_seed(stable_hash_int(f"NULL_HAAR_BASIS|{seed}"))
    d = evals.size
    U = haar_random_unitary(d, rng)
    H = U @ np.diag(evals) @ U.conj().T
    H = 0.5 * (H + H.conj().T)
    return H



def null_haar_basis_eigs(evals: np.ndarray, seed: int, tag: str = "NULL_HAAR_BASIS") -> Tuple[np.ndarray, np.ndarray]:
    """Return (evals, evecs) for the Haar-basis NULL without re-diagonalizing a full matrix."""
    rng = rng_from_seed(stable_hash_int(f"{tag}|{seed}"))
    d = evals.size
    U = haar_random_unitary(d, rng)
    return np.array(evals, dtype=float), U


def amplitude_entropy_bits(state: np.ndarray, eps: float = 1e-12) -> float:
    p = np.abs(state) ** 2
    p = p / (p.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def dominant_mask_by_mass(p: np.ndarray, keep_mass: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    keep_mass = float(keep_mass)
    if p.size == 0:
        return np.zeros_like(p, dtype=bool)
    keep_mass = max(0.0, min(1.0, keep_mass))
    order = np.argsort(-p, kind="mergesort")
    cum = 0.0
    mask = np.zeros(p.size, dtype=bool)
    for idx in order:
        mask[idx] = True
        cum += float(p[idx])
        if cum >= keep_mass and np.any(mask):
            break
    if not np.any(mask):
        mask[order[0]] = True
    return mask


def leakage_proxy_from_state(state: np.ndarray, keep_mask: np.ndarray, eps: float = 1e-12) -> float:
    p = np.abs(state) ** 2
    p = p / (p.sum() + eps)
    kept = float(np.sum(p[keep_mask])) if np.any(keep_mask) else 0.0
    return float(max(0.0, 1.0 - kept))


def leakage_proxy_fast(
    evals: np.ndarray,
    evecs: np.ndarray,
    i: int,
    j: int,
    times: List[float],
    keep_mass: float,
) -> Tuple[float, int]:
    vi = evecs[:, i]
    vj = evecs[:, j]
    psi0 = (vi + vj) / np.sqrt(2.0)

    p0 = np.abs(psi0) ** 2
    p0 = p0 / max(1e-12, float(np.sum(p0)))
    keep_mask = dominant_mask_by_mass(p0, keep_mass=keep_mass)
    dom = int(np.sum(keep_mask))
    dom = max(1, dom)

    Ei = float(evals[i])
    Ej = float(evals[j])

    leaks = []
    for t in times:
        ph_i = np.exp(-1j * Ei * float(t))
        ph_j = np.exp(-1j * Ej * float(t))
        psi_t = (ph_i * vi + ph_j * vj) / np.sqrt(2.0)
        leaks.append(leakage_proxy_from_state(psi_t, keep_mask))
    return float(np.mean(leaks)), dom


# ----------------------------
# Signatures / Candidates
# ----------------------------

SigAbs = Tuple[int, int, int, int]             # (dom, ent_bin_i, ent_bin_j, leak_bin)
SigQGlobal = Tuple[int, int, int, int]         # (dom, ent_q_lo, ent_q_hi, leak_bin)
SigQGCoarse = Tuple[int, int, int, int]        # (dom_bin, ent_q_lo, ent_q_hi, leak_bin_coarse)


def signature_key_abs(dom: int, ent_i: float, ent_j: float, leak: float, ent_step: float, leak_step: float) -> SigAbs:
    bi = bin_index(ent_i, ent_step)
    bj = bin_index(ent_j, ent_step)
    lo, hi = (bi, bj) if bi <= bj else (bj, bi)
    lb = bin_index(leak, leak_step)
    return (int(dom), int(lo), int(hi), int(lb))


def signature_key_q_global(dom: int, ent_i: float, ent_j: float, leak: float, edges: np.ndarray, q_bins: int, leak_step: float) -> SigQGlobal:
    qi = quantile_bin(ent_i, edges, q_bins)
    qj = quantile_bin(ent_j, edges, q_bins)
    lo, hi = (qi, qj) if qi <= qj else (qj, qi)
    lb = bin_index(leak, leak_step)
    return (int(dom), int(lo), int(hi), int(lb))


def dom_to_bin_ratio(dom: int, d: int) -> int:
    """
    Dimension-aware coarse binning for dom_count using dom/d ratio.
    Bins: [0..0.25], (0.25..0.5], (0.5..0.75], (0.75..1.0]
    Returns an integer in {0,1,2,3}.
    """
    dom = int(max(1, dom))
    d = int(max(1, d))
    r = float(dom) / float(d)
    if r <= 0.25:
        return 0
    if r <= 0.50:
        return 1
    if r <= 0.75:
        return 2
    return 3


def leak_bin_coarse_from_fine(leak_bin: int) -> int:
    # Collapse fine leakage bins: 0 -> 0, 1 -> 1, >=2 -> 2
    lb = int(leak_bin)
    if lb <= 0:
        return 0
    if lb == 1:
        return 1
    return 2


def signature_key_qg_coarse(
    dom: int,
    ent_i: float,
    ent_j: float,
    leak: float,
    d: int,
    edges: np.ndarray,
    q_bins_coarse: int,
    leak_step_fine: float
) -> SigQGCoarse:
    qi = quantile_bin(ent_i, edges, q_bins_coarse)
    qj = quantile_bin(ent_j, edges, q_bins_coarse)
    lo, hi = (qi, qj) if qi <= qj else (qj, qi)

    lb_fine = bin_index(leak, leak_step_fine)
    lb = leak_bin_coarse_from_fine(lb_fine)
    return (int(dom_to_bin_ratio(dom, d)), int(lo), int(hi), int(lb))


@dataclass(frozen=True)
class Candidate:
    seed: int
    model: str
    i: int
    j: int
    delta_e: float
    ent_i: float
    ent_j: float
    dom: int
    leak: float
    score: float
    sig_abs: SigAbs
    sig_qg: SigQGlobal
    sig_qg_coarse: SigQGCoarse


def interestingness_score(dom: int, leak: float, dom_weight: float = 0.6, leak_weight: float = 0.4) -> float:
    dom = max(1, int(dom))
    dom_term = 1.0 / (1.0 + float(dom))
    leak_term = 1.0 - float(leak)
    s = dom_weight * dom_term + leak_weight * leak_term
    return float(max(0.0, min(1.0, s)))


def find_neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    pairs = []
    for k in range(len(evals) - 1):
        de = float(abs(evals[k + 1] - evals[k]))
        if de < float(eps):
            pairs.append((k, k + 1, de))
    return pairs


def generate_candidates_for_seed(
    model: str,
    seed: int,
    cache: PauliCache,
    n_terms: int,
    eps_neighbor: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    keep_mass: float,
) -> List[Candidate]:
    """
    Generate candidates for one seed under REAL or NULL_HAAR_BASIS (spectrum-matched).
    Quantile signatures are assigned later at the batch level (pooled edges).
    """
    H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, seed)

    if model == "REAL":
        H = H_real
        evals, evecs = np.linalg.eigh(H)
    elif model == "NULL_HAAR_BASIS":
        evals_real, _ = np.linalg.eigh(H_real)
        H = null_haar_basis_hamiltonian(evals_real, seed=seed)
        evals, evecs = np.linalg.eigh(H)
    else:
        raise ValueError(f"Unknown model: {model}")

    pairs = find_neighbor_pairs(evals, eps_neighbor)
    if not pairs:
        return []

    cands: List[Candidate] = []
    for (i, j, de) in pairs:
        ent_i = amplitude_entropy_bits(evecs[:, i])
        ent_j = amplitude_entropy_bits(evecs[:, j])
        leak, dom = leakage_proxy_fast(evals, evecs, i, j, times, keep_mass=keep_mass)
        score = interestingness_score(dom, leak)
        sig_abs = signature_key_abs(dom, ent_i, ent_j, leak, ent_step, leak_step)
        cands.append(
            Candidate(
                seed=seed,
                model=model,
                i=i, j=j, delta_e=de,
                ent_i=ent_i, ent_j=ent_j,
                dom=dom, leak=leak, score=score,
                sig_abs=sig_abs,
                sig_qg=(0, 0, 0, 0),           # filled later
                sig_qg_coarse=(0, 0, 0, 0),    # filled later
            )
        )
    return cands


def assign_quantile_signatures(
    cands: List[Candidate],
    d: int,
    edges_fine: np.ndarray,
    q_bins_fine: int,
    edges_coarse: np.ndarray,
    q_bins_coarse: int,
    leak_step: float
) -> List[Candidate]:
    out: List[Candidate] = []
    for c in cands:
        sig_qg = signature_key_q_global(c.dom, c.ent_i, c.ent_j, c.leak, edges_fine, q_bins_fine, leak_step)
        sig_qg_coarse = signature_key_qg_coarse(c.dom, c.ent_i, c.ent_j, c.leak, d, edges_coarse, q_bins_coarse, leak_step)
        out.append(Candidate(**{**c.__dict__, "sig_qg": sig_qg, "sig_qg_coarse": sig_qg_coarse}))
    return out


# ----------------------------
# Enrichment + batch summaries
# ----------------------------

@dataclass(frozen=True)
class FamRow:
    sig: Tuple[int, int, int, int]
    overall: int
    stable: int
    expected: float
    p_tail: float
    neglog10_p: float
    enrichment: float

# ----------------------------
# Perturbation robustness (v17)
# ----------------------------

def _match_two_indices_unique(evals_base: np.ndarray, evals_pert: np.ndarray, i: int, j: int) -> Tuple[int, int]:
    """Match base eigenvalue indices (i,j) to unique indices in the perturbed spectrum by nearest energy."""
    Ei = float(evals_base[i]); Ej = float(evals_base[j])
    di = np.abs(evals_pert - Ei)
    dj = np.abs(evals_pert - Ej)
    k = int(np.argmin(di))
    l = int(np.argmin(dj))
    if k != l:
        return k, l
    # Resolve collision by giving the second-closest match to the worse of the two
    order_i = np.argsort(di)
    order_j = np.argsort(dj)
    # Prefer keeping the closer one fixed
    if di[k] <= dj[l]:
        # move j
        for cand in order_j:
            cand = int(cand)
            if cand != k:
                return k, cand
    else:
        # move i
        for cand in order_i:
            cand = int(cand)
            if cand != l:
                return cand, l
    # Fallback (should not happen for d>=2)
    return k, l


def perturbation_robustness_summary_for_batch(
    *,
    cache: PauliCache,
    seed_start: int,
    n_seeds: int,
    n_terms: int,
    perturb_terms: Optional[int],
    eps_neighbor: float,
    times: List[float],
    keep_mass: float,
    ent_step: float,
    leak_step: float,
    edges_coarse: np.ndarray,
    q_bins_coarse: int,
    stable_frac: float,
    eta_list: List[float],
    reps: int,
    pairs_per_seed: int,
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """
    Compute perturbation robustness summaries for both REAL and NULL models.

    The perturbation is applied in the REAL parameter space:
      H_real' = H_real + eta * ΔH, where ΔH is a small random Pauli-sum.
    For each perturbation, NULL is rebuilt as a fresh Haar basis using the perturbed eigenvalues,
    preserving the spectrum-matched construction under perturbation.

    Returns a dict:
      eta -> { "REAL": metrics, "NULL": metrics }
    """
    d = cache.d
    perturb_terms_eff = int(perturb_terms) if perturb_terms is not None else int(n_terms)

    # accumulators: eta -> model -> sums
    out: Dict[float, Dict[str, Dict[str, float]]] = {}
    for eta in eta_list:
        out[float(eta)] = {
            "REAL": {"pairs": 0.0, "retained": 0.0, "sig_retained": 0.0, "d_ent": 0.0, "d_leak": 0.0},
            "NULL": {"pairs": 0.0, "retained": 0.0, "sig_retained": 0.0, "d_ent": 0.0, "d_leak": 0.0},
        }

    if not eta_list or reps <= 0 or n_seeds <= 0:
        return out

    for s in range(int(n_seeds)):
        seed = int(seed_start + s)

        # Base REAL spectrum/eigenvectors
        H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, seed)
        evals_base, evecs_real_base = np.linalg.eigh(H_real)

        # Base NULL eigenvectors (Haar basis) with same eigenvalues
        _, evecs_null_base = null_haar_basis_eigs(evals_base, seed=seed, tag="NULL_HAAR_BASIS")

        pairs = find_neighbor_pairs(evals_base, eps_neighbor)
        if not pairs:
            continue

        # Precompute base candidate features for both models; optionally select top pairs_per_seed by score.
        base_lists: Dict[str, List[Tuple[int,int,float,float,int,float,SigQGCoarse]]] = {"REAL": [], "NULL": []}
        for model, evecs in [("REAL", evecs_real_base), ("NULL", evecs_null_base)]:
            feats: List[Tuple[int,int,float,float,int,float,SigQGCoarse,float]] = []
            for (i, j, _de) in pairs:
                ent_i = amplitude_entropy_bits(evecs[:, i])
                ent_j = amplitude_entropy_bits(evecs[:, j])
                leak, dom = leakage_proxy_fast(evals_base, evecs, i, j, times, keep_mass=keep_mass)
                score = interestingness_score(dom, leak)
                sig_c = signature_key_qg_coarse(dom, ent_i, ent_j, leak, d, edges_coarse, q_bins_coarse, leak_step)
                feats.append((i, j, ent_i, ent_j, dom, leak, sig_c, score))
            feats.sort(key=lambda t: t[-1])  # lower score = more "stable/interesting"
            take = int(max(1, min(int(pairs_per_seed), len(feats))))
            base_lists[model] = [(i, j, ent_i, ent_j, dom, leak, sig_c) for (i, j, ent_i, ent_j, dom, leak, sig_c, _score) in feats[:take]]

        for eta in eta_list:
            eta = float(eta)
            for rep in range(int(reps)):
                # Perturb in REAL space (deterministic)
                seed_pert = stable_hash_int(f"PERT|{seed}|eta{eta:.6g}|rep{rep}")
                dH = build_random_pauli_hamiltonian_cached(cache, perturb_terms_eff, seed_pert)
                H_real_p = H_real + (eta * dH)
                H_real_p = 0.5 * (H_real_p + H_real_p.conj().T)
                evals_p, evecs_real_p = np.linalg.eigh(H_real_p)

                # NULL under perturbation: Haar basis with perturbed eigenvalues (deterministic per rep)
                tag = f"NULL_HAAR_BASIS_PERT|eta{eta:.6g}|rep{rep}"
                _, evecs_null_p = null_haar_basis_eigs(evals_p, seed=seed, tag=tag)

                for model, evecs_base, evecs_p, base_items in [
                    ("REAL", evecs_real_base, evecs_real_p, base_lists["REAL"]),
                    ("NULL", evecs_null_base, evecs_null_p, base_lists["NULL"]),
                ]:
                    for (i, j, ent_i0, ent_j0, _dom0, leak0, sig0) in base_items:
                        k, l = _match_two_indices_unique(evals_base, evals_p, i, j)
                        if k == l:
                            continue
                        if k > l:
                            k, l = l, k
                        de_p = float(abs(evals_p[l] - evals_p[k]))
                        retained = (l == k + 1) and (de_p < float(eps_neighbor))

                        ent_i1 = amplitude_entropy_bits(evecs_p[:, k])
                        ent_j1 = amplitude_entropy_bits(evecs_p[:, l])
                        leak1, dom1 = leakage_proxy_fast(evals_p, evecs_p, k, l, times, keep_mass=keep_mass)
                        sig1 = signature_key_qg_coarse(dom1, ent_i1, ent_j1, leak1, d, edges_coarse, q_bins_coarse, leak_step)

                        acc = out[eta][model]
                        acc["pairs"] += 1.0
                        acc["retained"] += 1.0 if retained else 0.0
                        acc["sig_retained"] += 1.0 if (sig1 == sig0) else 0.0
                        acc["d_ent"] += 0.5 * (abs(ent_i1 - ent_i0) + abs(ent_j1 - ent_j0))
                        acc["d_leak"] += abs(leak1 - leak0)

    # finalize to rates/means
    for eta in eta_list:
        eta = float(eta)
        for model in ["REAL", "NULL"]:
            acc = out[eta][model]
            n = max(1.0, acc["pairs"])
            acc["pair_retention_rate"] = acc["retained"] / n
            acc["sig_coarse_retention_rate"] = acc["sig_retained"] / n
            acc["mean_abs_d_entropy_bits"] = acc["d_ent"] / n
            acc["mean_abs_d_leak"] = acc["d_leak"] / n
            acc["pairs_total"] = acc["pairs"]
            # remove raw sums to keep output clean
            for k in ["pairs","retained","sig_retained","d_ent","d_leak"]:
                acc.pop(k, None)

    return out




def family_rows(
    sigs: List[Tuple[int, int, int, int]],
    stable_mask: np.ndarray,
    alpha: float,
) -> Tuple[List[FamRow], Dict[Tuple[int, int, int, int], Tuple[int, int]], float]:
    overall = Counter(sigs)
    stable = Counter([sigs[i] for i in range(len(sigs)) if stable_mask[i]])

    n_all = len(sigs)
    n_stable = int(np.sum(stable_mask))
    stable_rate = n_stable / max(1, n_all)

    K = max(1, len(overall))
    rows: List[FamRow] = []
    counts: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}

    for sig, o in overall.items():
        st = stable.get(sig, 0)
        counts[sig] = (int(o), int(st))

        p_all = (o + alpha) / (n_all + alpha * K)
        p_st = (st + alpha) / (n_stable + alpha * K)
        enr = (p_st / p_all) / max(1e-12, stable_rate)

        p_tail = binom_tail_ge(int(st), int(o), stable_rate) if o > 0 else 1.0
        p_tail = max(1e-300, min(1.0, float(p_tail)))
        neglog10 = -math.log10(p_tail)

        rows.append(
            FamRow(
                sig=sig,
                overall=int(o),
                stable=int(st),
                expected=float(o) * stable_rate,
                p_tail=float(p_tail),
                neglog10_p=float(neglog10),
                enrichment=float(enr),
            )
        )

    rows.sort(key=lambda r: (-r.neglog10_p, -r.enrichment, -r.stable, -r.overall))
    return rows, counts, stable_rate


def top_k_families(rows: List[FamRow], k: int, min_overall: int, min_stable: int, p_tail_max: Optional[float]) -> List[FamRow]:
    out: List[FamRow] = []
    for r in rows:
        if r.overall >= min_overall and r.stable >= min_stable:
            if p_tail_max is None or r.p_tail <= float(p_tail_max):
                out.append(r)
        if len(out) >= k:
            break
    return out


def summarize_entropy_effect(ent_vals: np.ndarray, stable_mask: np.ndarray, rng: np.random.Generator, B: int = 200) -> Tuple[float, Tuple[float, float]]:
    all_med = float(np.median(ent_vals))
    st_med = float(np.median(ent_vals[stable_mask])) if np.any(stable_mask) else all_med
    eff = float(st_med - all_med)

    n = ent_vals.size
    idx_all = np.arange(n)
    effects = []
    for _ in range(int(B)):
        samp = rng.choice(idx_all, size=n, replace=True)
        samp_vals = ent_vals[samp]
        samp_mask = stable_mask[samp]
        all_m = float(np.median(samp_vals))
        st_m = float(np.median(samp_vals[samp_mask])) if np.any(samp_mask) else all_m
        effects.append(st_m - all_m)
    lo, hi = np.quantile(np.array(effects, dtype=float), [0.025, 0.975])
    return eff, (float(lo), float(hi))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled = math.sqrt(max(1e-12, ((x.size - 1) * vx + (y.size - 1) * vy) / max(1, (x.size + y.size - 2))))
    return float((mx - my) / pooled)


def stable_mask_from_scores_and_leak(
    scores: np.ndarray,
    leaks: np.ndarray,
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
) -> np.ndarray:
    """
    Fixed-fraction stable selection per model (v15):

    Select exactly k = round(stable_frac * n) (minimum 1 if n>0) items per model,
    based on score (lower is better). Optionally impose a leakage constraint:

      - absolute: leak <= stable_leak_max
      - or quantile: leak <= quantile(leak, stable_leak_quantile)

    If a leakage constraint is provided:
      - If eligible (leak <= threshold) count >= k: pick the best k among eligible by score.
      - If eligible count < k: pick all eligible, then fill remainder from ineligible by (score, leak).
    """
    n = int(scores.size)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    stable_frac = float(stable_frac)
    stable_frac = max(0.0, min(1.0, stable_frac))
    k = int(round(stable_frac * n))
    k = max(1, min(n, k))

    order = np.lexsort((leaks, scores))  # score primary, leak secondary

    if stable_leak_max is None and stable_leak_quantile is None:
        chosen = order[:k]
        mask = np.zeros((n,), dtype=bool)
        mask[chosen] = True
        return mask

    if stable_leak_max is not None:
        leak_thr = float(stable_leak_max)
    else:
        q = float(stable_leak_quantile) if stable_leak_quantile is not None else 1.0
        q = max(0.0, min(1.0, q))
        leak_thr = float(np.quantile(leaks, q))

    eligible = leaks <= leak_thr
    eligible_order = [i for i in order if eligible[i]]

    chosen: List[int] = []
    if len(eligible_order) >= k:
        chosen = eligible_order[:k]
    else:
        chosen = eligible_order[:]
        need = k - len(chosen)
        if need > 0:
            for i in order:
                if eligible[i]:
                    continue
                chosen.append(i)
                need -= 1
                if need == 0:
                    break

    mask = np.zeros((n,), dtype=bool)
    mask[chosen] = True
    return mask


# ----------------------------
# Batch runner (paired REAL + NULL)
# ----------------------------

@dataclass
class BatchResult:
    batch_id: int
    seed_offset: int
    model: str
    n_candidates: int
    stable_rate: float
    stable_rate_scoreonly: float
    score_stats: Tuple[float, float, float]
    leak_stats: Tuple[float, float, float]
    ent_stats: Tuple[float, float, float]
    dom_stats: Tuple[float, float, float]  # mean, median, max
    ent_pool: np.ndarray
    leak_vals: np.ndarray
    dom_vals: np.ndarray
    top_qg: List[FamRow]
    top_qg_coarse: List[FamRow]
    effect_entropy_bits: float
    effect_ci: Tuple[float, float]
    top_keys_qg: List[SigQGlobal]
    top_keys_qg_coarse: List[SigQGCoarse]
    counts_qg: Dict[SigQGlobal, Tuple[int, int]]
    counts_qg_coarse: Dict[SigQGCoarse, Tuple[int, int]]


def compute_batch_result(
    *,
    batch_id: int,
    seed_offset: int,
    model: str,
    cands: List[Candidate],
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    p_tail_max: Optional[float],
    bootstrap: int,
) -> BatchResult:
    if not cands:
        return BatchResult(
            batch_id=batch_id, seed_offset=seed_offset, model=model,
            n_candidates=0,
            stable_rate=0.0, stable_rate_scoreonly=0.0,
            score_stats=(0.0, 0.0, 0.0),
            leak_stats=(0.0, 0.0, 0.0),
            ent_stats=(0.0, 0.0, 0.0),
            dom_stats=(0.0, 0.0, 0.0),
            ent_pool=np.array([], dtype=float),
            leak_vals=np.array([], dtype=float),
            dom_vals=np.array([], dtype=float),
            top_qg=[], top_qg_coarse=[],
            effect_entropy_bits=0.0, effect_ci=(0.0, 0.0),
            top_keys_qg=[], top_keys_qg_coarse=[],
            counts_qg={}, counts_qg_coarse={},
        )

    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak for c in cands], dtype=float)
    doms = np.array([c.dom for c in cands], dtype=float)
    ent_pool = np.array([c.ent_i for c in cands] + [c.ent_j for c in cands], dtype=float)

    mask_scoreonly = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, None, None)
    stable_rate_scoreonly = float(np.sum(mask_scoreonly) / max(1, len(cands)))

    mask = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, stable_leak_max, stable_leak_quantile)

    sigs_qg = [c.sig_qg for c in cands]
    rows_qg, counts_qg, stable_rate_qg = family_rows(sigs_qg, mask, alpha=alpha)
    top_qg = top_k_families(rows_qg, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    sigs_qg_c = [c.sig_qg_coarse for c in cands]
    rows_qg_c, counts_qg_c, _ = family_rows(sigs_qg_c, mask, alpha=alpha)
    top_qg_c = top_k_families(rows_qg_c, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    rng_eff = rng_from_seed(stable_hash_int(f"{model}|batch{batch_id}|eff"))
    eff, ci = summarize_entropy_effect(ent_pool, np.repeat(mask, 2), rng_eff, B=bootstrap)

    return BatchResult(
        batch_id=batch_id, seed_offset=seed_offset, model=model,
        n_candidates=len(cands),
        stable_rate=float(stable_rate_qg),
        stable_rate_scoreonly=float(stable_rate_scoreonly),
        score_stats=(float(scores.mean()), float(np.median(scores)), float(scores.max())),
        leak_stats=(float(leaks.mean()), float(np.median(leaks)), float(leaks.min())),
        ent_stats=(float(ent_pool.mean()), float(np.median(ent_pool)), float(ent_pool.max())),
        dom_stats=(float(doms.mean()), float(np.median(doms)), float(doms.max())),
        ent_pool=ent_pool,
        leak_vals=leaks,
        dom_vals=doms,
        top_qg=top_qg,
        top_qg_coarse=top_qg_c,
        effect_entropy_bits=eff,
        effect_ci=ci,
        top_keys_qg=[r.sig for r in top_qg],
        top_keys_qg_coarse=[r.sig for r in top_qg_c],
        counts_qg=counts_qg,
        counts_qg_coarse=counts_qg_c,
    )


def run_paired_batches(
    *,
    cache: PauliCache,
    n_terms: int,
    seeds_per_batch: int,
    n_batches: int,
    base_seed: int,
    batch_stride: int,
    eps_neighbor: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    keep_mass: float,
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    q_bins: int,
    q_bins_coarse: int,
    p_tail_max: Optional[float],
    bootstrap: int,
    perturb_eta: List[float],
    perturb_reps: int,
    perturb_seeds: int,
    perturb_terms: Optional[int],
    perturb_pairs_per_seed: int,
) -> Tuple[List[BatchResult], List[BatchResult], List[Dict[str, float]], List[Dict[float, Dict[str, Dict[str, float]]]]]:
    real_results: List[BatchResult] = []
    null_results: List[BatchResult] = []
    scoreboards: List[Dict[str, float]] = []
    perturb_summaries: List[Dict[float, Dict[str, Dict[str, float]]]] = []

    d = cache.d

    for b in range(n_batches):
        offset = base_seed + b * batch_stride
        t0 = time.time()

        real_cands: List[Candidate] = []
        null_cands: List[Candidate] = []

        for s in range(seeds_per_batch):
            seed = offset + s
            real_cands.extend(
                generate_candidates_for_seed(
                    model="REAL",
                    seed=seed,
                    cache=cache,
                    n_terms=n_terms,
                    eps_neighbor=eps_neighbor,
                    ent_step=ent_step,
                    leak_step=leak_step,
                    times=times,
                    keep_mass=keep_mass,
                )
            )
            null_cands.extend(
                generate_candidates_for_seed(
                    model="NULL_HAAR_BASIS",
                    seed=seed,
                    cache=cache,
                    n_terms=n_terms,
                    eps_neighbor=eps_neighbor,
                    ent_step=ent_step,
                    leak_step=leak_step,
                    times=times,
                    keep_mass=keep_mass,
                )
            )

        pooled_ent = np.array(
            [c.ent_i for c in real_cands] + [c.ent_j for c in real_cands] +
            [c.ent_i for c in null_cands] + [c.ent_j for c in null_cands],
            dtype=float
        )

        edges_fine = make_quantile_edges(pooled_ent, q_bins=q_bins) if pooled_ent.size else make_quantile_edges(np.array([0.0]), q_bins=q_bins)
        edges_coarse = make_quantile_edges(pooled_ent, q_bins=q_bins_coarse) if pooled_ent.size else make_quantile_edges(np.array([0.0]), q_bins=q_bins_coarse)

        real_cands = assign_quantile_signatures(real_cands, d, edges_fine, q_bins, edges_coarse, q_bins_coarse, leak_step)
        null_cands = assign_quantile_signatures(null_cands, d, edges_fine, q_bins, edges_coarse, q_bins_coarse, leak_step)

        # v17: perturbation robustness (optional; uses only a small prefix of seeds to control runtime)
        pert_summary: Dict[float, Dict[str, Dict[str, float]]] = {}
        if perturb_eta and int(perturb_seeds) > 0:
            seed_start = int(offset)
            n_use = int(min(int(perturb_seeds), int(seeds_per_batch)))
            pert_summary = perturbation_robustness_summary_for_batch(
                cache=cache,
                seed_start=seed_start,
                n_seeds=n_use,
                n_terms=n_terms,
                perturb_terms=perturb_terms,
                eps_neighbor=eps_neighbor,
                times=times,
                keep_mass=keep_mass,
                ent_step=ent_step,
                leak_step=leak_step,
                edges_coarse=edges_coarse,
                q_bins_coarse=q_bins_coarse,
                stable_frac=stable_frac,
                eta_list=list(perturb_eta),
                reps=int(perturb_reps),
                pairs_per_seed=int(perturb_pairs_per_seed),
            )
        perturb_summaries.append(pert_summary)


        elapsed = time.time() - t0
        print(f"Batch {b+1}/{n_batches} generated: REAL={len(real_cands)} NULL={len(null_cands)} (elapsed {elapsed:.1f}s)")

        R = compute_batch_result(
            batch_id=b,
            seed_offset=offset,
            model="REAL",
            cands=real_cands,
            stable_frac=stable_frac,
            stable_leak_max=stable_leak_max,
            stable_leak_quantile=stable_leak_quantile,
            topK=topK,
            min_overall=min_overall,
            min_stable=min_stable,
            alpha=alpha,
            p_tail_max=p_tail_max,
            bootstrap=bootstrap,
        )
        N = compute_batch_result(
            batch_id=b,
            seed_offset=offset,
            model="NULL_HAAR_BASIS",
            cands=null_cands,
            stable_frac=stable_frac,
            stable_leak_max=stable_leak_max,
            stable_leak_quantile=stable_leak_quantile,
            topK=topK,
            min_overall=min_overall,
            min_stable=min_stable,
            alpha=alpha,
            p_tail_max=p_tail_max,
            bootstrap=bootstrap,
        )

        real_results.append(R)
        null_results.append(N)

        sb = {}
        sb["delta_median_entropy_bits"] = safe_median(R.ent_pool) - safe_median(N.ent_pool)
        sb["delta_median_leak"] = safe_median(R.leak_vals) - safe_median(N.leak_vals)
        sb["delta_median_dom"] = safe_median(R.dom_vals) - safe_median(N.dom_vals)
        sb["entropy_cohens_d"] = cohens_d(R.ent_pool, N.ent_pool)
        scoreboards.append(sb)

    return real_results, null_results, scoreboards, perturb_summaries


# ----------------------------
# Reporting
# ----------------------------

def format_top(rows: List[FamRow], label: str, show: int = 10) -> str:
    lines = []
    lines.append(f"Top signature families ({label}):")
    lines.append("Format: sig=(a,b,c,d) | overall | stable | expected | p_tail | -log10(p) | enrichment")
    for r in rows[:show]:
        lines.append(
            f"  {r.sig} | {r.overall:7d} | {r.stable:7d} | {r.expected:9.2f} | {r.p_tail:8.2e} | {r.neglog10_p:9.2f} | {r.enrichment:9.2f}x"
        )
    if len(rows) > show:
        lines.append(f"  ... ({len(rows)} total, showing {show})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0_7 convergence harness v17 (v16 baseline + perturbation robustness; n_qubits=4 default).")
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--n_terms", type=int, default=5)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--batch_stride", type=int, default=1000000)
    ap.add_argument("--eps_neighbor", type=float, default=0.05)

    ap.add_argument("--keep_mass", type=float, default=0.90)
    ap.add_argument("--ent_step", type=float, default=0.1)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])

    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--stable_leak_max", type=float, default=None)
    ap.add_argument("--stable_leak_quantile", type=float, default=None)

    # v17: perturbation robustness controls (disabled unless --perturb_eta is provided and --perturb_seeds > 0)
    ap.add_argument("--perturb_eta", type=float, nargs="+", default=[])
    ap.add_argument("--perturb_reps", type=int, default=3)
    ap.add_argument("--perturb_seeds", type=int, default=0, help="Number of seeds per batch used for perturbation robustness (0 disables).")
    ap.add_argument("--perturb_terms", type=int, default=None, help="Number of Pauli terms in ΔH (default: same as --n_terms).")
    ap.add_argument("--perturb_pairs_per_seed", type=int, default=5, help="Number of near-degenerate pairs per seed used in robustness summary (lowest-score subset).")

    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=3)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--q_bins", type=int, default=10)
    ap.add_argument("--q_bins_coarse", type=int, default=6)
    ap.add_argument("--p_tail_max", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=200)

    ap.add_argument("--output", type=str, default="v0_7_convergence_families_v17_output.txt")
    args = ap.parse_args()

    cache = PauliCache.build(args.n_qubits)
    d = cache.d

    header = []
    header.append("=== v0_7: Convergence + Baseline Calibration + Perturbation Robustness (v17) ===")
    header.append(f"Qubits: {args.n_qubits} (d={d}) | terms={args.n_terms}")
    header.append(f"Batches: {args.batches} × {args.seeds_per_batch} seeds (base_seed={args.base_seed}, stride={args.batch_stride})")
    header.append(f"Neighbor eps={args.eps_neighbor:.3f}")
    header.append(f"Dominant set: keep_mass={args.keep_mass:.2f} (mass-based; guarantees non-empty mask)")
    header.append(f"Bins (SigAbs): ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
    header.append(f"Bins (SigQ_GLOBAL fine): q_bins={args.q_bins} (pooled REAL+NULL per batch)")
    header.append(f"Bins (SigQG_COARSE): q_bins_coarse={args.q_bins_coarse} + dom_bin(dom/d) + leak_bin_coarse (pooled REAL+NULL per batch)")
    header.append(f"Leakage proxy times={args.times} (FAST analytic evolution in eigenpair)")
    header.append(f"Stable selection: stable_frac={args.stable_frac:.3f} per model (optional leak constraint max={args.stable_leak_max}, q={args.stable_leak_quantile})")
    header.append(f"Perturbation robustness: eta={args.perturb_eta} | reps={args.perturb_reps} | seeds={args.perturb_seeds} | pairs/seed={args.perturb_pairs_per_seed} | dH_terms={args.perturb_terms}")
    header.append(f"TopK={args.topK} | min_overall={args.min_overall} | min_stable={args.min_stable} | alpha={args.alpha}")
    header.append(f"Optional family filter: p_tail_max={args.p_tail_max}")
    header.append("Pauli ops: precomputed cache (excluding all-I)")
    header.append("")
    header_text = "\n".join(header)

    with open(args.output, "w", encoding="utf-8") as f:
        def out(s: str = "") -> None:
            print(s)
            f.write(s + "\n")

        out(header_text)

        real, null, scoreboards, perturb_summaries = run_paired_batches(
            cache=cache,
            n_terms=args.n_terms,
            seeds_per_batch=args.seeds_per_batch,
            n_batches=args.batches,
            base_seed=args.base_seed,
            batch_stride=args.batch_stride,
            eps_neighbor=args.eps_neighbor,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=list(args.times),
            keep_mass=args.keep_mass,
            stable_frac=args.stable_frac,
            stable_leak_max=args.stable_leak_max,
            stable_leak_quantile=args.stable_leak_quantile,
            topK=args.topK,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            q_bins=args.q_bins,
            q_bins_coarse=args.q_bins_coarse,
            p_tail_max=args.p_tail_max,
            bootstrap=args.bootstrap,
            perturb_eta=list(args.perturb_eta),
            perturb_reps=args.perturb_reps,
            perturb_seeds=args.perturb_seeds,
            perturb_terms=args.perturb_terms,
            perturb_pairs_per_seed=args.perturb_pairs_per_seed,
        )

        out("")
        for b in range(args.batches):
            out("----------------------------------------------")
            out(f"Batch {b+1}/{args.batches} (seed_offset={real[b].seed_offset})")

            R = real[b]
            out(f"Model: REAL | candidates={R.n_candidates} | stable_rate={R.stable_rate:.4f} (score-only ref={R.stable_rate_scoreonly:.4f})")
            out(f"  score(mean/median/max)={R.score_stats[0]:.3f}/{R.score_stats[1]:.3f}/{R.score_stats[2]:.3f}")
            out(f"  leakage(mean/median/min)={R.leak_stats[0]:.3f}/{R.leak_stats[1]:.3f}/{R.leak_stats[2]:.3f}")
            out(f"  entropy(mean/median/max)={R.ent_stats[0]:.3f}/{R.ent_stats[1]:.3f}/{R.ent_stats[2]:.3f}")
            out(f"  dom_count(mean/median/max)={R.dom_stats[0]:.2f}/{R.dom_stats[1]:.1f}/{R.dom_stats[2]:.0f}")
            out(f"  entropy effect (stable-overall)={R.effect_entropy_bits:+.3f} bits  CI95={R.effect_ci}")
            out("")
            out(format_top(R.top_qg, "REAL, SigQ_GLOBAL (fine)"))
            out("")
            out(format_top(R.top_qg_coarse, "REAL, SigQG_COARSE"))
            out("")

            N = null[b]
            out(f"Model: NULL_HAAR_BASIS | candidates={N.n_candidates} | stable_rate={N.stable_rate:.4f} (score-only ref={N.stable_rate_scoreonly:.4f})")
            out(f"  score(mean/median/max)={N.score_stats[0]:.3f}/{N.score_stats[1]:.3f}/{N.score_stats[2]:.3f}")
            out(f"  leakage(mean/median/min)={N.leak_stats[0]:.3f}/{N.leak_stats[1]:.3f}/{N.leak_stats[2]:.3f}")
            out(f"  entropy(mean/median/max)={N.ent_stats[0]:.3f}/{N.ent_stats[1]:.3f}/{N.ent_stats[2]:.3f}")
            out(f"  dom_count(mean/median/max)={N.dom_stats[0]:.2f}/{N.dom_stats[1]:.1f}/{N.dom_stats[2]:.0f}")
            out(f"  entropy effect (stable-overall)={N.effect_entropy_bits:+.3f} bits  CI95={N.effect_ci}")
            out("")
            out(format_top(N.top_qg, "NULL, SigQ_GLOBAL (fine)"))
            out("")
            out(format_top(N.top_qg_coarse, "NULL, SigQG_COARSE"))
            out("")

            # v17: Perturbation robustness summary (optional)
            ps = perturb_summaries[b] if b < len(perturb_summaries) else {}
            if ps:
                out("Perturbation robustness (spectrum-matched under perturbation):")
                out(f"  settings: eta={args.perturb_eta} | reps={args.perturb_reps} | seeds_used={min(args.perturb_seeds, args.seeds_per_batch)} | pairs/seed={args.perturb_pairs_per_seed} | dH_terms={args.perturb_terms}")
                for eta in sorted(ps.keys()):
                    eta_f = float(eta)
                    for model in ["REAL", "NULL"]:
                        mtr = ps[eta_f][model]
                        out(
                            f"  eta={eta_f:.6g} | {model}: "
                            f"pair_retention={mtr['pair_retention_rate']:.3f} | "
                            f"sig_coarse_retention={mtr['sig_coarse_retention_rate']:.3f} | "
                            f"mean|Δentropy|={mtr['mean_abs_d_entropy_bits']:.3f} bits | "
                            f"mean|Δleak|={mtr['mean_abs_d_leak']:.3f} | "
                            f"pairs={int(mtr['pairs_total'])}"
                        )
                out("")

            ov_fine = jaccard(R.top_keys_qg, N.top_keys_qg)
            ov_coarse = jaccard(R.top_keys_qg_coarse, N.top_keys_qg_coarse)
            out(f"Batch {b+1}: Jaccard(Top-{args.topK}) REAL vs NULL: fine={ov_fine:.3f} | coarse={ov_coarse:.3f}")
            out("")

            sb = scoreboards[b]
            out("Baseline scoreboard (REAL - NULL):")
            out(f"  delta median entropy (bits): {sb['delta_median_entropy_bits']:+.3f}")
            out(f"  delta median leakage       : {sb['delta_median_leak']:+.3f}")
            out(f"  delta median dom_count     : {sb['delta_median_dom']:+.3f}")
            out(f"  entropy Cohen's d          : {sb['entropy_cohens_d']:+.3f}")
            out("")

        out("----------------------------------------------")
        out("=== Convergence diagnostics (REAL) ===")
        real_sets_fine = [set(r.top_keys_qg) for r in real]
        real_sets_coarse = [set(r.top_keys_qg_coarse) for r in real]
        for i in range(len(real_sets_fine)):
            for j in range(i + 1, len(real_sets_fine)):
                out(f"REAL overlap fine:   Jaccard(Top-{args.topK}) batch{i+1} vs batch{j+1} = {jaccard(real_sets_fine[i], real_sets_fine[j]):.3f}")
                out(f"REAL overlap coarse: Jaccard(Top-{args.topK}) batch{i+1} vs batch{j+1} = {jaccard(real_sets_coarse[i], real_sets_coarse[j]):.3f}")

        out("")
        out("=== Notes (scientific reading) ===")
        out("1) Compare REAL vs NULL primarily via deltas/effect sizes and batch stability, not raw entropy levels (d differs with n_qubits).")
        out("2) Use fine families for within-model discovery; use coarse families for cross-model interpretability.")
        out("3) If results at n=4 resemble n=3 (stable deltas + stable overlaps), that is strong qualitative evidence the effect is not a 3-qubit artifact.")
        out("")
        out("=== End of v16 ===")


if __name__ == "__main__":
    main()
