#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v13.py

Purpose
-------
Step-wise refinement of v12 "convergence + baseline calibration" harness.

What v13 refines vs v12
-----------------------
A) Cross-model comparability when signature supports are disjoint
   1) Adds a COARSE signature key in addition to SigQ_GLOBAL (fine):
        SigQG_COARSE = (dom_bin, ent_q_lo, ent_q_hi, leak_bin_coarse)
      where dom and leakage bins are coarsened to reduce support mismatch.
   2) Adds a nearest-neighbor (NN) paired view:
      If a src Top-family is absent in dst (overall=0), we report the nearest
      dst signature in feature space (L1 distance on components with weights).

B) "Stable" definition robustness
   1) Keeps score-top stable_frac selection, but allows an additional leakage
      constraint:
        - absolute cutoff: leak <= stable_leak_max
        - or quantile cutoff: leak <= quantile(leak, stable_leak_quantile)
      This reduces cases where "stable" is dominated by one metric and improves
      interpretability.

C) Baseline scoreboard
   1) Per batch, reports distribution-level REAL-NULL separation:
        - delta median entropy (bits)
        - delta median leakage
        - delta median dom_count
        - Cohen's d for entropy pool (REAL vs NULL)
   2) Keeps binomial tail p-values and enrichment for continuity.

Dependencies
------------
Only numpy (no Qiskit, no SciPy).

Notes
-----
- v13 remains a small-n (dense) harness (e.g., n_qubits=3).
- Fine families are for within-model discovery.
- Coarse + NN paired views are for cross-model interpretability.

Template provenance
-------------------
Derived from v12 baseline script.
"""

from __future__ import annotations

import argparse
import math
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional
from collections import Counter

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
    """
    Exact tail probability P(X >= k), X ~ Binomial(n, p).
    Uses log-sum-exp for stability; intended for moderate n (counts per signature are small).
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_terms = [_log_binom_pmf(x, n, p) for x in range(k, n + 1)]
    return float(math.exp(_logsumexp(log_terms)))


# ----------------------------
# Physics-ish primitives (local)
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


def build_random_pauli_hamiltonian(n_qubits: int, n_terms: int, seed: int) -> np.ndarray:
    rng = rng_from_seed(seed)
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)
    for _ in range(n_terms):
        while True:
            s = [PAULIS[int(rng.integers(0, 4))] for _ in range(n_qubits)]
            if any(p != "I" for p in s):
                break
        coeff = float(rng.uniform(-1.0, 1.0))
        P = kron_n([PAULI_MATS[p] for p in s])
        H = H + coeff * P
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


def dom_to_bin(dom: int) -> int:
    # Default coarse bins for d=8 (n=3): {1-2}, {3-4}, {5-8}
    dom = int(max(1, dom))
    if dom <= 2:
        return 0
    if dom <= 4:
        return 1
    return 2


def leak_bin_coarse_from_fine(leak_bin: int) -> int:
    # Collapse fine leakage bins: 0 -> 0, 1 -> 1, >=2 -> 2
    lb = int(leak_bin)
    if lb <= 0:
        return 0
    if lb == 1:
        return 1
    return 2


def signature_key_qg_coarse(dom: int, ent_i: float, ent_j: float, leak: float,
                            edges: np.ndarray, q_bins_coarse: int,
                            leak_step_fine: float) -> SigQGCoarse:
    # Use its own (coarser) quantile bins, but still pooled edges constructed at that resolution.
    qi = quantile_bin(ent_i, edges, q_bins_coarse)
    qj = quantile_bin(ent_j, edges, q_bins_coarse)
    lo, hi = (qi, qj) if qi <= qj else (qj, qi)

    lb_fine = bin_index(leak, leak_step_fine)
    lb = leak_bin_coarse_from_fine(lb_fine)
    return (int(dom_to_bin(dom)), int(lo), int(hi), int(lb))


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
    n_qubits: int,
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
    H_real = build_random_pauli_hamiltonian(n_qubits, n_terms, seed)

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
    edges_fine: np.ndarray,
    q_bins_fine: int,
    edges_coarse: np.ndarray,
    q_bins_coarse: int,
    leak_step: float
) -> List[Candidate]:
    out: List[Candidate] = []
    for c in cands:
        sig_qg = signature_key_q_global(c.dom, c.ent_i, c.ent_j, c.leak, edges_fine, q_bins_fine, leak_step)
        sig_qg_coarse = signature_key_qg_coarse(c.dom, c.ent_i, c.ent_j, c.leak, edges_coarse, q_bins_coarse, leak_step)
        out.append(Candidate(**{**c.__dict__, "sig_qg": sig_qg, "sig_qg_coarse": sig_qg_coarse}))
    return out


# ----------------------------
# Enrichment + batch summaries
# ----------------------------

def stable_mask_from_scores_and_leak(
    scores: np.ndarray,
    leaks: np.ndarray,
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
) -> np.ndarray:
    """
    Base stable set: top stable_frac by score.
    Optionally intersect with a leakage constraint:
      - absolute: leak <= stable_leak_max
      - or quantile: leak <= quantile(leak, stable_leak_quantile)
    If the intersection becomes empty, fall back to score-only mask.
    """
    n = scores.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    k = max(1, int(math.ceil(float(stable_frac) * n)))
    idx = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[idx[:k]] = True

    if stable_leak_max is not None:
        mask2 = mask & (leaks <= float(stable_leak_max))
        if np.any(mask2):
            return mask2
        return mask  # fallback

    if stable_leak_quantile is not None:
        q = float(stable_leak_quantile)
        q = max(0.0, min(1.0, q))
        thr = float(np.quantile(leaks, q))
        mask2 = mask & (leaks <= thr)
        if np.any(mask2):
            return mask2
        return mask  # fallback

    return mask


@dataclass(frozen=True)
class FamRow:
    sig: Tuple[int, int, int, int]
    overall: int
    stable: int
    expected: float
    p_tail: float
    neglog10_p: float
    enrichment: float


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
    top_abs: List[FamRow]
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
            n_candidates=0, stable_rate=0.0, stable_rate_scoreonly=0.0,
            score_stats=(0.0, 0.0, 0.0),
            leak_stats=(0.0, 0.0, 0.0),
            ent_stats=(0.0, 0.0, 0.0),
            dom_stats=(0.0, 0.0, 0.0),
            ent_pool=np.array([], dtype=float),
            leak_vals=np.array([], dtype=float),
            dom_vals=np.array([], dtype=float),
            top_abs=[], top_qg=[], top_qg_coarse=[],
            effect_entropy_bits=0.0, effect_ci=(0.0, 0.0),
            top_keys_qg=[], top_keys_qg_coarse=[],
            counts_qg={}, counts_qg_coarse={},
        )

    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak for c in cands], dtype=float)
    doms = np.array([c.dom for c in cands], dtype=float)
    ent_pool = np.array([c.ent_i for c in cands] + [c.ent_j for c in cands], dtype=float)

    # Score-only stable mask for reference rate
    mask_scoreonly = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, None, None)
    stable_rate_scoreonly = float(np.sum(mask_scoreonly) / max(1, len(cands)))

    # Robust mask with optional leakage constraint
    mask = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, stable_leak_max, stable_leak_quantile)

    sigs_abs = [c.sig_abs for c in cands]
    rows_abs, _, stable_rate = family_rows(sigs_abs, mask, alpha=alpha)
    top_abs = top_k_families(rows_abs, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    sigs_qg = [c.sig_qg for c in cands]
    rows_qg, counts_qg, stable_rate_qg = family_rows(sigs_qg, mask, alpha=alpha)
    stable_rate = stable_rate_qg
    top_qg = top_k_families(rows_qg, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    sigs_qg_c = [c.sig_qg_coarse for c in cands]
    rows_qg_c, counts_qg_c, _stable_rate_c = family_rows(sigs_qg_c, mask, alpha=alpha)
    top_qg_c = top_k_families(rows_qg_c, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    rng_eff = rng_from_seed(stable_hash_int(f"{model}|batch{batch_id}|eff"))
    eff, ci = summarize_entropy_effect(ent_pool, np.repeat(mask, 2), rng_eff, B=bootstrap)

    return BatchResult(
        batch_id=batch_id, seed_offset=seed_offset, model=model,
        n_candidates=len(cands),
        stable_rate=float(stable_rate),
        stable_rate_scoreonly=float(stable_rate_scoreonly),
        score_stats=(float(scores.mean()), float(np.median(scores)), float(scores.max())),
        leak_stats=(float(leaks.mean()), float(np.median(leaks)), float(leaks.min())),
        ent_stats=(float(ent_pool.mean()), float(np.median(ent_pool)), float(ent_pool.max())),
        dom_stats=(float(doms.mean()), float(np.median(doms)), float(doms.max())),
        ent_pool=ent_pool,
        leak_vals=leaks,
        dom_vals=doms,
        top_abs=top_abs,
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
    n_qubits: int,
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
) -> Tuple[List[BatchResult], List[BatchResult], List[Dict[str, float]]]:
    real_results: List[BatchResult] = []
    null_results: List[BatchResult] = []
    scoreboards: List[Dict[str, float]] = []

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
                    n_qubits=n_qubits,
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
                    n_qubits=n_qubits,
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

        real_cands = assign_quantile_signatures(real_cands, edges_fine, q_bins, edges_coarse, q_bins_coarse, leak_step)
        null_cands = assign_quantile_signatures(null_cands, edges_fine, q_bins, edges_coarse, q_bins_coarse, leak_step)

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

        # Scoreboard metrics (REAL - NULL)
        sb = {}
        sb["delta_median_entropy_bits"] = safe_median(R.ent_pool) - safe_median(N.ent_pool)
        sb["delta_median_leak"] = safe_median(R.leak_vals) - safe_median(N.leak_vals)
        sb["delta_median_dom"] = safe_median(R.dom_vals) - safe_median(N.dom_vals)
        sb["entropy_cohens_d"] = cohens_d(R.ent_pool, N.ent_pool)  # mean difference / pooled std
        scoreboards.append(sb)

    return real_results, null_results, scoreboards


# ----------------------------
# Paired reporting helpers (exact + nearest neighbor)
# ----------------------------

def _sig_array_and_list(keys: Iterable[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    lst = list(keys)
    if not lst:
        return np.zeros((0, 4), dtype=int), []
    arr = np.array(lst, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 4:
        arr = arr.reshape((-1, 4))
    return arr, lst


def nn_match_signature(
    sig: Tuple[int, int, int, int],
    dst_keys_arr: np.ndarray,
    dst_keys_list: List[Tuple[int, int, int, int]],
    w: Tuple[float, float, float, float] = (2.0, 1.0, 1.0, 1.0),
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Returns nearest neighbor signature in dst and its weighted L1 distance.
    """
    if dst_keys_arr.size == 0:
        return None, float("inf")
    s = np.array(sig, dtype=float).reshape((1, 4))
    dif = np.abs(dst_keys_arr.astype(float) - s)
    wv = np.array(w, dtype=float).reshape((1, 4))
    dist = np.sum(dif * wv, axis=1)
    k = int(np.argmin(dist))
    return dst_keys_list[k], float(dist[k])


# ----------------------------
# Reporting
# ----------------------------

def format_top(rows: List[FamRow], label: str, show: int = 10) -> str:
    lines = []
    lines.append(f"Top signature families ({label}):")
    lines.append("Format: sig=(a,b,c,d) | overall | stable | expected | p_tail | -log10(p) | enrichment")
    for r in rows[:show]:
        lines.append(
            f"  {r.sig} | {r.overall:5d} | {r.stable:5d} | {r.expected:7.2f} | {r.p_tail:8.2e} | {r.neglog10_p:9.2f} | {r.enrichment:9.2f}x"
        )
    if len(rows) > show:
        lines.append(f"  ... ({len(rows)} total, showing {show})")
    return "\n".join(lines)


def format_paired_exact_nn(
    top_keys: List[Tuple[int, int, int, int]],
    src_label: str,
    src_counts: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    dst_label: str,
    dst_counts: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    dst_stable_rate: float,
    show: int = 10,
    nn: bool = True,
    nn_weights: Tuple[float, float, float, float] = (2.0, 1.0, 1.0, 1.0),
) -> str:
    """
    Paired view with optional nearest-neighbor fallback when exact match is absent in dst.
    """
    dst_arr, dst_list = _sig_array_and_list(dst_counts.keys())

    lines = []
    lines.append(f"Paired view: {src_label} Top families scored under {dst_label}")
    lines.append("sig | src(o,s) | dst(o,s) | dst expected | dst p_tail | dst -log10(p) | nn_sig | nn_dist")

    for sig in top_keys[:show]:
        so, ss = src_counts.get(sig, (0, 0))
        do, ds = dst_counts.get(sig, (0, 0))
        exp = do * dst_stable_rate
        p_tail = binom_tail_ge(ds, do, dst_stable_rate) if do > 0 else 1.0
        p_tail = max(1e-300, min(1.0, float(p_tail)))
        neglog10 = -math.log10(p_tail)

        nn_sig, nn_dist = (None, float("inf"))
        if nn and do == 0:
            nn_sig, nn_dist = nn_match_signature(sig, dst_arr, dst_list, w=nn_weights)

        lines.append(
            f"{sig} | ({so},{ss}) | ({do},{ds}) | {exp:7.2f} | {p_tail:8.2e} | {neglog10:9.2f} | {nn_sig} | {nn_dist:6.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0_7 convergence harness v13 (coarse signatures + NN paired view + robust stable definition + scoreboard).")
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--n_terms", type=int, default=5)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--batch_stride", type=int, default=1000000)
    ap.add_argument("--eps_neighbor", type=float, default=0.05)

    ap.add_argument("--keep_mass", type=float, default=0.90, help="Cumulative probability mass to keep for the dominant mask.")
    ap.add_argument("--ent_step", type=float, default=0.1)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])

    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--stable_leak_max", type=float, default=None, help="Optional: additionally require leak <= this value for stable candidates.")
    ap.add_argument("--stable_leak_quantile", type=float, default=None, help="Optional: additionally require leak <= quantile(leak, q) for stable candidates (q in [0,1]).")

    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=3)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--q_bins", type=int, default=10)
    ap.add_argument("--q_bins_coarse", type=int, default=6, help="Coarse quantile bins for cross-model support.")
    ap.add_argument("--p_tail_max", type=float, default=None, help="Optional filter: keep only families with p_tail <= p_tail_max.")
    ap.add_argument("--bootstrap", type=int, default=200)

    ap.add_argument("--paired_nn", action="store_true", help="Enable nearest-neighbor fallback in paired view when exact dst match is absent.")
    ap.add_argument("--nn_w_dom", type=float, default=2.0)
    ap.add_argument("--nn_w_e1", type=float, default=1.0)
    ap.add_argument("--nn_w_e2", type=float, default=1.0)
    ap.add_argument("--nn_w_leak", type=float, default=1.0)

    ap.add_argument("--output", type=str, default="v0_7_convergence_families_v13_output.txt")
    args = ap.parse_args()

    header = []
    header.append("=== v0_7: Convergence + Baseline Calibration (v13) ===")
    header.append(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.n_terms}")
    header.append(f"Batches: {args.batches} × {args.seeds_per_batch} seeds (base_seed={args.base_seed}, stride={args.batch_stride})")
    header.append(f"Neighbor eps={args.eps_neighbor:.3f}")
    header.append(f"Dominant set: keep_mass={args.keep_mass:.2f} (mass-based; guarantees non-empty mask)")
    header.append(f"Bins (SigAbs): ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
    header.append(f"Bins (SigQ_GLOBAL fine): q_bins={args.q_bins} (pooled REAL+NULL per batch)")
    header.append(f"Bins (SigQG_COARSE): q_bins_coarse={args.q_bins_coarse} + dom_bin + leak_bin_coarse (pooled REAL+NULL per batch)")
    header.append(f"Leakage proxy times={args.times} (FAST analytic evolution in eigenpair)")
    header.append(f"Stable selection: stable_frac={args.stable_frac:.3f} with optional leak constraint (max={args.stable_leak_max}, q={args.stable_leak_quantile})")
    header.append(f"TopK={args.topK} | min_overall={args.min_overall} | min_stable={args.min_stable} | alpha={args.alpha}")
    header.append(f"Optional family filter: p_tail_max={args.p_tail_max}")
    header.append(f"Paired NN fallback: {bool(args.paired_nn)} (weights dom/e1/e2/leak = {args.nn_w_dom}/{args.nn_w_e1}/{args.nn_w_e2}/{args.nn_w_leak})")
    header.append("")
    header_text = "\n".join(header)

    with open(args.output, "w", encoding="utf-8") as f:
        def out(s: str = "") -> None:
            print(s)
            f.write(s + "\n")

        out(header_text)

        real, null, scoreboards = run_paired_batches(
            n_qubits=args.n_qubits,
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
        )

        nn_weights = (args.nn_w_dom, args.nn_w_e1, args.nn_w_e2, args.nn_w_leak)

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

            ov_fine = jaccard(R.top_keys_qg, N.top_keys_qg)
            ov_coarse = jaccard(R.top_keys_qg_coarse, N.top_keys_qg_coarse)
            out(f"Batch {b+1}: Jaccard(Top-{args.topK}) REAL vs NULL: fine(SigQ_GLOBAL)={ov_fine:.3f} | coarse(SigQG_COARSE)={ov_coarse:.3f}")
            out("")

            out(format_paired_exact_nn(
                top_keys=R.top_keys_qg,
                src_label="REAL fine",
                src_counts=R.counts_qg,
                dst_label="NULL fine",
                dst_counts=N.counts_qg,
                dst_stable_rate=N.stable_rate,
                show=min(10, args.topK),
                nn=bool(args.paired_nn),
                nn_weights=nn_weights,
            ))
            out("")
            out(format_paired_exact_nn(
                top_keys=N.top_keys_qg,
                src_label="NULL fine",
                src_counts=N.counts_qg,
                dst_label="REAL fine",
                dst_counts=R.counts_qg,
                dst_stable_rate=R.stable_rate,
                show=min(10, args.topK),
                nn=bool(args.paired_nn),
                nn_weights=nn_weights,
            ))
            out("")
            out(format_paired_exact_nn(
                top_keys=R.top_keys_qg_coarse,
                src_label="REAL coarse",
                src_counts=R.counts_qg_coarse,
                dst_label="NULL coarse",
                dst_counts=N.counts_qg_coarse,
                dst_stable_rate=N.stable_rate,
                show=min(10, args.topK),
                nn=False,
            ))
            out("")
            out(format_paired_exact_nn(
                top_keys=N.top_keys_qg_coarse,
                src_label="NULL coarse",
                src_counts=N.counts_qg_coarse,
                dst_label="REAL coarse",
                dst_counts=R.counts_qg_coarse,
                dst_stable_rate=R.stable_rate,
                show=min(10, args.topK),
                nn=False,
            ))
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
        out("1) Use fine families for within-model discovery; use coarse + NN paired views for cross-model interpretability.")
        out("2) If fine cross-model overlap is ~0 but coarse overlap is >0, the signal is likely present but too sparse under fine hashing.")
        out("3) If both fine and coarse overlaps are ~0, REAL and NULL are genuinely separated under current features / generator.")
        out("4) Consider tightening stable via --stable_leak_max (e.g., 0.05) or --stable_leak_quantile (e.g., 0.25) if stable sets look noisy.")
        out("")
        out("=== End of v13 ===")


if __name__ == "__main__":
    main()
