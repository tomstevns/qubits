#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v10.py

Purpose
-------
Convergence + calibration harness for the v0_6.1-style "signature family" pipeline,
with one key fix motivated by v9 outputs:

  *Absolute entropy bins can be disjoint between REAL and NULL (e.g., REAL low-entropy,
   NULL high-entropy), which makes cross-model family keys non-overlapping.*

v10 therefore adds a *quantile-binned* signature key (SigQ) for cross-model comparison,
while keeping the original *absolute-binned* signature key (SigAbs) for continuity.

What this script reports
------------------------
For each of 3 batches (default), it runs:

  - REAL model: Pauli-sum Hamiltonians (3 qubits, fixed number of terms)
  - NULL_HAAR_BASIS: spectrum-matched "chance" baseline with Haar-random eigenvectors

Per model and per batch, it computes candidates (near-degenerate neighbor pairs),
then assigns signature keys based on:
  (dom_count, entropy_bin_i, entropy_bin_j, leakage_bin)

Two key modes:
  - SigAbs: absolute bins in entropy (bits) and leakage (0..1)
  - SigQ  : entropy bins are quantiles (0..Q-1) computed *within the batch+model*

Cross-model calibration then uses SigQ by default (because it guarantees overlap in labels).

No external dependencies beyond numpy are used (Qiskit not required).
"""

from __future__ import annotations

import argparse
import math
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
from collections import Counter, defaultdict

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def stable_hash_int(s: str) -> int:
    """Deterministic int hash for reproducible shuffles independent of Python's hash randomization."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def bin_index(x: float, step: float) -> int:
    """Non-negative bin index; step must be > 0."""
    if step <= 0:
        raise ValueError("step must be > 0")
    return int(math.floor(float(x) / float(step) + 1e-12))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def make_quantile_edges(values: np.ndarray, q_bins: int) -> np.ndarray:
    """
    Quantile edges for q_bins categories.
    Returns an array of length q_bins+1 with edges[0]=-inf and edges[-1]=+inf.
    """
    if q_bins < 2:
        raise ValueError("q_bins must be >= 2")
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        # Degenerate case: no values
        edges = np.linspace(0.0, 1.0, q_bins + 1)
    else:
        qs = np.linspace(0.0, 1.0, q_bins + 1)
        # numpy supports method=... in newer versions; fallback safely
        try:
            edges = np.quantile(v, qs, method="linear")
        except TypeError:
            edges = np.quantile(v, qs)
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Ensure monotone non-decreasing (guard numerical noise)
    edges = np.maximum.accumulate(edges)
    return edges


def quantile_bin(x: float, edges: np.ndarray, q_bins: int) -> int:
    """Map x to [0, q_bins-1] using quantile edges."""
    idx = int(np.searchsorted(edges, float(x), side="right") - 1)
    return clamp_int(idx, 0, q_bins - 1)


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
    """
    Random Pauli-sum Hamiltonian (dense), small n only.
    Coefficients ~ Uniform(-1,1), Pauli strings uniformly sampled excluding all-identity.
    """
    rng = rng_from_seed(seed)
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)
    for _ in range(n_terms):
        s = []
        # avoid identity-only terms by forcing at least one non-I
        while True:
            s = [PAULIS[int(rng.integers(0, 4))] for _ in range(n_qubits)]
            if any(p != "I" for p in s):
                break
        coeff = float(rng.uniform(-1.0, 1.0))
        P = kron_n([PAULI_MATS[p] for p in s])
        H = H + coeff * P
    # Hermitian by construction; enforce symmetry numerically
    H = 0.5 * (H + H.conj().T)
    return H


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR of complex Gaussian."""
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    # Normalize phases
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph
    return Q


def null_haar_basis_hamiltonian(evals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Spectrum-matched null: keep eigenvalues, randomize eigenvectors Haar-uniformly.
    H_null = U diag(evals) U^\dagger
    """
    d = evals.size
    U = haar_random_unitary(d, rng)
    H = U @ np.diag(evals) @ U.conj().T
    H = 0.5 * (H + H.conj().T)
    return H


def amplitude_entropy_bits(state: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of measurement distribution in computational basis (bits)."""
    p = np.abs(state) ** 2
    p = p / (p.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def dominant_basis_count(state: np.ndarray, threshold: float) -> int:
    """
    Number of basis states with probability >= threshold.
    Uses p=|amp|^2.
    """
    p = np.abs(state) ** 2
    return int(np.sum(p >= float(threshold)))


def leakage_proxy_from_state(
    state: np.ndarray,
    dominant_mask: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Simple leakage proxy: 1 - sum_{x in dominant} |amp_x|^2
    """
    p = np.abs(state) ** 2
    p = p / (p.sum() + eps)
    kept = float(np.sum(p[dominant_mask]))
    return float(max(0.0, 1.0 - kept))


def time_evolve_state(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """
    Exact time evolution via diagonalization: psi(t)=V exp(-i E t) V^† psi0
    (OK for d=8).
    """
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * float(t))
    psi_t = evecs @ (phases * (evecs.conj().T @ psi0))
    return psi_t


def leakage_proxy(
    H: np.ndarray,
    i: int,
    j: int,
    evecs: np.ndarray,
    times: List[float],
    dom_threshold: float,
) -> float:
    """
    Lightweight leakage proxy:
      - build an initial state as equal superposition of the two eigenvectors i and j
      - identify dominant computational basis states in psi(0) using dom_threshold
      - evolve under H for each t and report average leakage outside dominant mask
    """
    psi0 = (evecs[:, i] + evecs[:, j]) / np.sqrt(2.0)
    p0 = np.abs(psi0) ** 2
    dom_mask = p0 >= float(dom_threshold)
    leaks = []
    for t in times:
        psi_t = time_evolve_state(H, psi0, t)
        leaks.append(leakage_proxy_from_state(psi_t, dom_mask))
    return float(np.mean(leaks))


# ----------------------------
# Signatures / Candidates
# ----------------------------

SigAbs = Tuple[int, int, int, int]  # (dom, ent_bin_i, ent_bin_j, leak_bin)
SigQ = Tuple[int, int, int, int]    # (dom, ent_q_i, ent_q_j, leak_bin)


def signature_key_abs(dom: int, ent_i: float, ent_j: float, leak: float, ent_step: float, leak_step: float) -> SigAbs:
    bi = bin_index(ent_i, ent_step)
    bj = bin_index(ent_j, ent_step)
    lo, hi = (bi, bj) if bi <= bj else (bj, bi)
    lb = bin_index(leak, leak_step)
    return (int(dom), int(lo), int(hi), int(lb))


def signature_key_q(dom: int, ent_i: float, ent_j: float, leak: float, q_edges: np.ndarray, q_bins: int, leak_step: float) -> SigQ:
    qi = quantile_bin(ent_i, q_edges, q_bins)
    qj = quantile_bin(ent_j, q_edges, q_bins)
    lo, hi = (qi, qj) if qi <= qj else (qj, qi)
    lb = bin_index(leak, leak_step)
    return (int(dom), int(lo), int(hi), int(lb))


@dataclass(frozen=True)
class Candidate:
    seed: int
    i: int
    j: int
    delta_e: float
    ent_i: float
    ent_j: float
    dom: int
    leak: float
    score: float
    sig_abs: SigAbs
    sig_q: SigQ


def interestingness_score(dom: int, leak: float, dom_weight: float = 0.6, leak_weight: float = 0.4) -> float:
    """
    Score in [0,1] roughly; lower dom and lower leak => higher score.
    This is intentionally simple and monotone.
    """
    dom_term = 1.0 / (1.0 + float(dom))          # dom=1 -> 0.5, dom=2 -> 0.333, ...
    leak_term = 1.0 - float(leak)                # leak=0 -> 1, leak=1 -> 0
    s = dom_weight * dom_term + leak_weight * leak_term
    return float(max(0.0, min(1.0, s)))


# ----------------------------
# Candidate generation
# ----------------------------

def find_neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    """Only compare adjacent eigenvalues (fast neighbor scan)."""
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
    dom_threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    rng_null: np.random.Generator,
    q_bins: int,
) -> List[Candidate]:
    """
    Generate candidates for one seed.
    For REAL: build a random Pauli Hamiltonian
    For NULL_HAAR_BASIS: build REAL first to get evals, then randomize eigenvectors.
    """
    d = 2 ** n_qubits
    H_real = build_random_pauli_hamiltonian(n_qubits, n_terms, seed)

    if model == "REAL":
        H = H_real
        evals, evecs = np.linalg.eigh(H)
    elif model == "NULL_HAAR_BASIS":
        evals_real, _ = np.linalg.eigh(H_real)
        H = null_haar_basis_hamiltonian(evals_real, rng_null)
        evals, evecs = np.linalg.eigh(H)
    else:
        raise ValueError(f"Unknown model: {model}")

    pairs = find_neighbor_pairs(evals, eps_neighbor)
    if not pairs:
        return []

    # Precompute entropy distribution for this seed? We want batch-level quantiles,
    # not seed-level. So we collect raw candidates first and add sig_q later in the batch.
    raw = []
    for (i, j, de) in pairs:
        ent_i = amplitude_entropy_bits(evecs[:, i])
        ent_j = amplitude_entropy_bits(evecs[:, j])
        # dom count from initial superposition in computational basis
        psi0 = (evecs[:, i] + evecs[:, j]) / np.sqrt(2.0)
        dom = dominant_basis_count(psi0, dom_threshold)
        leak = leakage_proxy(H, i, j, evecs, times, dom_threshold)
        score = interestingness_score(dom, leak)
        sig_abs = signature_key_abs(dom, ent_i, ent_j, leak, ent_step, leak_step)
        raw.append((i, j, de, ent_i, ent_j, dom, leak, score, sig_abs))

    # placeholder SigQ, set later
    cands = [
        Candidate(
            seed=seed,
            i=i, j=j, delta_e=de,
            ent_i=ent_i, ent_j=ent_j,
            dom=dom, leak=leak, score=score,
            sig_abs=sig_abs,
            sig_q=(0, 0, 0, 0),
        )
        for (i, j, de, ent_i, ent_j, dom, leak, score, sig_abs) in raw
    ]
    return cands


def assign_quantile_signatures(cands: List[Candidate], q_bins: int, leak_step: float) -> List[Candidate]:
    """Assign SigQ to candidates based on within-batch entropy quantiles."""
    if not cands:
        return []
    ent_vals = np.array([c.ent_i for c in cands] + [c.ent_j for c in cands], dtype=float)
    edges = make_quantile_edges(ent_vals, q_bins=q_bins)
    updated = []
    for c in cands:
        sig_q = signature_key_q(c.dom, c.ent_i, c.ent_j, c.leak, edges, q_bins, leak_step)
        updated.append(Candidate(**{**c.__dict__, "sig_q": sig_q}))
    return updated


# ----------------------------
# Enrichment + batch summaries
# ----------------------------

def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    """Top stable_frac by score; ties break stably by index order."""
    n = scores.size
    k = max(1, int(math.ceil(stable_frac * n)))
    idx = np.argsort(-scores, kind="mergesort")  # stable sort
    mask = np.zeros(n, dtype=bool)
    mask[idx[:k]] = True
    return mask


def enrichment_rows(
    sigs: List[Tuple[int, int, int, int]],
    stable_mask: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[Tuple[Tuple[int, int, int, int], int, int, float]], Dict[Tuple[int, int, int, int], int], Dict[Tuple[int, int, int, int], int]]:
    """
    Compute enrichment per signature:
      enrichment = (stable_freq / overall_freq) / stable_frac
    with Laplace smoothing alpha on counts to avoid zeros.
    """
    overall = Counter(sigs)
    stable = Counter([sigs[i] for i in range(len(sigs)) if stable_mask[i]])
    n_all = len(sigs)
    n_stable = int(np.sum(stable_mask))
    stable_frac = n_stable / max(1, n_all)

    rows = []
    for sig in overall.keys():
        o = overall[sig]
        st = stable.get(sig, 0)
        # smoothed probabilities
        p_all = (o + alpha) / (n_all + alpha * len(overall))
        p_st = (st + alpha) / (n_stable + alpha * len(overall))
        enr = (p_st / p_all) / max(1e-12, stable_frac)
        rows.append((sig, o, st, float(enr)))

    rows.sort(key=lambda x: x[3], reverse=True)
    return rows, dict(overall), dict(stable)


def top_k_families(
    rows: List[Tuple[Tuple[int, int, int, int], int, int, float]],
    k: int,
    min_overall: int,
    min_stable: int,
) -> List[Tuple[Tuple[int, int, int, int], int, int, float]]:
    out = []
    for (sig, o, st, enr) in rows:
        if o >= min_overall and st >= min_stable:
            out.append((sig, o, st, enr))
        if len(out) >= k:
            break
    return out


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def summarize_entropy_effect(ent_vals: np.ndarray, stable_mask: np.ndarray, rng: np.random.Generator, B: int = 200) -> Tuple[float, Tuple[float, float]]:
    """
    Effect size: median(entropy) in stable set minus median(entropy) overall.
    Returns (effect, 95% bootstrap CI).
    """
    all_med = float(np.median(ent_vals))
    st_med = float(np.median(ent_vals[stable_mask]))
    eff = float(st_med - all_med)

    # bootstrap
    n = ent_vals.size
    effects = []
    idx_all = np.arange(n)
    for _ in range(int(B)):
        samp = rng.choice(idx_all, size=n, replace=True)
        samp_vals = ent_vals[samp]
        samp_mask = stable_mask[samp]
        all_m = float(np.median(samp_vals))
        st_m = float(np.median(samp_vals[samp_mask])) if np.any(samp_mask) else all_m
        effects.append(st_m - all_m)
    lo, hi = np.quantile(np.array(effects), [0.025, 0.975])
    return eff, (float(lo), float(hi))


# ----------------------------
# Batch runner
# ----------------------------

@dataclass
class BatchResult:
    model: str
    batch_id: int
    seed_offset: int
    n_candidates: int
    score_stats: Tuple[float, float, float]      # mean, median, max
    leak_stats: Tuple[float, float, float]       # mean, median, min
    ent_stats: Tuple[float, float, float]        # mean, median, max (pooled i/j)
    dom_median: float
    top_abs: List[Tuple[SigAbs, int, int, float]]
    top_q: List[Tuple[SigQ, int, int, float]]
    effect_entropy_bits: float
    effect_ci: Tuple[float, float]


def run_batches(
    model: str,
    *,
    n_qubits: int,
    n_terms: int,
    seeds_per_batch: int,
    n_batches: int,
    base_seed: int,
    batch_stride: int,
    eps_neighbor: float,
    dom_threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    stable_frac: float,
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    q_bins: int,
    bootstrap: int,
) -> List[BatchResult]:
    results: List[BatchResult] = []
    rng_null = rng_from_seed(base_seed + 99991)

    for b in range(n_batches):
        offset = base_seed + b * batch_stride
        t0 = time.time()
        cands: List[Candidate] = []
        for s in range(seeds_per_batch):
            seed = offset + s
            cands.extend(
                generate_candidates_for_seed(
                    model=model,
                    seed=seed,
                    n_qubits=n_qubits,
                    n_terms=n_terms,
                    eps_neighbor=eps_neighbor,
                    dom_threshold=dom_threshold,
                    ent_step=ent_step,
                    leak_step=leak_step,
                    times=times,
                    rng_null=rng_null,
                    q_bins=q_bins,
                )
            )

        # Assign SigQ on the batch as a whole
        cands = assign_quantile_signatures(cands, q_bins=q_bins, leak_step=leak_step)

        elapsed = time.time() - t0

        if not cands:
            results.append(
                BatchResult(
                    model=model, batch_id=b, seed_offset=offset,
                    n_candidates=0,
                    score_stats=(0.0, 0.0, 0.0),
                    leak_stats=(0.0, 0.0, 0.0),
                    ent_stats=(0.0, 0.0, 0.0),
                    dom_median=0.0,
                    top_abs=[], top_q=[],
                    effect_entropy_bits=0.0,
                    effect_ci=(0.0, 0.0),
                )
            )
            continue

        scores = np.array([c.score for c in cands], dtype=float)
        leaks = np.array([c.leak for c in cands], dtype=float)
        doms = np.array([c.dom for c in cands], dtype=float)
        ent_pool = np.array([c.ent_i for c in cands] + [c.ent_j for c in cands], dtype=float)

        mask = stable_mask_from_scores(scores, stable_frac)

        # Two key modes
        sigs_abs = [c.sig_abs for c in cands]
        rows_abs, _, _ = enrichment_rows(sigs_abs, mask, alpha=alpha)
        top_abs = top_k_families(rows_abs, k=topK, min_overall=min_overall, min_stable=min_stable)

        sigs_q = [c.sig_q for c in cands]
        rows_q, _, _ = enrichment_rows(sigs_q, mask, alpha=alpha)
        top_q = top_k_families(rows_q, k=topK, min_overall=min_overall, min_stable=min_stable)

        # Distribution-level effect on entropy (pooled i/j)
        rng_eff = rng_from_seed(stable_hash_int(f"{model}|batch{b}|eff"))
        eff, ci = summarize_entropy_effect(ent_pool, np.repeat(mask, 2), rng_eff, B=bootstrap)

        br = BatchResult(
            model=model, batch_id=b, seed_offset=offset,
            n_candidates=len(cands),
            score_stats=(float(scores.mean()), float(np.median(scores)), float(scores.max())),
            leak_stats=(float(leaks.mean()), float(np.median(leaks)), float(leaks.min())),
            ent_stats=(float(ent_pool.mean()), float(np.median(ent_pool)), float(ent_pool.max())),
            dom_median=float(np.median(doms)),
            top_abs=top_abs,
            top_q=top_q,
            effect_entropy_bits=eff,
            effect_ci=ci,
        )

        results.append(br)

        print(f"Model: {model:14s} | batch={b+1}/{n_batches} | candidates={len(cands):6d} | elapsed={elapsed:.1f}s")

    return results


# ----------------------------
# Cross-model calibration (SigQ)
# ----------------------------

def cross_model_calibration(
    real: List[BatchResult],
    null: List[BatchResult],
    key_mode: str,
    topK: int,
) -> str:
    """
    Summarize cross-model results. Uses key_mode in {"SigQ","SigAbs"}.
    Returns a human-readable multi-line string.
    """
    if len(real) != len(null):
        return "Cross-model calibration skipped (batch counts differ)."

    lines = []
    lines.append("=== Cross-model calibration (REAL vs NULL) ===")
    lines.append(f"Key mode: {key_mode} (SigQ is recommended for cross-model family overlap)")
    lines.append("Interpretation: treat ratios > 1 as REAL being more enriched than NULL under matched settings.")
    lines.append("")

    # Collect per-batch top lists
    for b in range(len(real)):
        R = real[b]
        N = null[b]
        topR = R.top_q if key_mode == "SigQ" else R.top_abs
        topN = N.top_q if key_mode == "SigQ" else N.top_abs

        keysR = [x[0] for x in topR]
        keysN = [x[0] for x in topN]
        ov = jaccard(keysR, keysN)
        lines.append(f"Batch {b+1}: Jaccard(Top-{topK}) REAL vs NULL = {ov:.3f}")

    lines.append("")
    # Convergence across REAL batches
    if len(real) >= 2:
        keys_sets = [set([x[0] for x in (r.top_q if key_mode == 'SigQ' else r.top_abs)]) for r in real]
        ov01 = jaccard(keys_sets[0], keys_sets[1])
        lines.append(f"REAL batch overlap: Jaccard(topK) batch1 vs batch2 = {ov01:.3f}")
        if len(real) >= 3:
            ov02 = jaccard(keys_sets[0], keys_sets[2])
            ov12 = jaccard(keys_sets[1], keys_sets[2])
            lines.append(f"REAL batch overlap: Jaccard(topK) batch1 vs batch3 = {ov02:.3f}")
            lines.append(f"REAL batch overlap: Jaccard(topK) batch2 vs batch3 = {ov12:.3f}")

    lines.append("")
    # Distribution-level effect aggregation (entropy)
    effR = np.array([r.effect_entropy_bits for r in real], dtype=float)
    effN = np.array([n.effect_entropy_bits for n in null], dtype=float)
    lines.append("Distribution-level effect (entropy, bits): median(stable) - median(overall)")
    lines.append(f"  REAL: mean={effR.mean():+.3f}  across {len(effR)} batches")
    lines.append(f"  NULL: mean={effN.mean():+.3f}  across {len(effN)} batches")
    lines.append(f"  REAL-NULL (batch means): {(effR.mean()-effN.mean()):+.3f}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------
# Reporting
# ----------------------------

def format_top(top: List[Tuple[Tuple[int, int, int, int], int, int, float]], label: str, show: int = 10) -> str:
    lines = []
    lines.append(f"Top signature families ({label}):")
    lines.append("Format: sig=(dom, e1, e2, leak_bin) | overall | stable | enrichment")
    for (sig, o, st, enr) in top[:show]:
        lines.append(f"  {sig} | {o:5d} | {st:5d} | {enr:8.2f}x")
    if len(top) > show:
        lines.append(f"  ... ({len(top)} total, showing {show})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0_7 convergence harness v10 (adds quantile-binned signature keys).")
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--n_terms", type=int, default=5)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--batch_stride", type=int, default=1000000)
    ap.add_argument("--eps_neighbor", type=float, default=0.05)
    ap.add_argument("--dom_threshold", type=float, default=0.25)
    ap.add_argument("--ent_step", type=float, default=0.1)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=3)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing alpha for enrichment.")
    ap.add_argument("--q_bins", type=int, default=10, help="Quantile bins for SigQ entropy keys.")
    ap.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for entropy effect CI.")
    ap.add_argument("--cross_key", type=str, default="SigQ", choices=["SigQ", "SigAbs"])
    ap.add_argument("--output", type=str, default="v0_7_convergence_families_v10_output.txt")
    args = ap.parse_args()

    header = []
    header.append("=== v0_7: Convergence + Baseline Calibration (v10) ===")
    header.append(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.n_terms}")
    header.append(f"Batches: {args.batches} × {args.seeds_per_batch} seeds (base_seed={args.base_seed}, stride={args.batch_stride})")
    header.append(f"Neighbor eps={args.eps_neighbor:.3f} | dom_threshold={args.dom_threshold:.3f}")
    header.append(f"Bins (SigAbs): ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
    header.append(f"Bins (SigQ): q_bins={args.q_bins} (entropy quantiles within batch+model)")
    header.append(f"Leakage proxy times={args.times} | stable_frac={args.stable_frac:.3f} | topK={args.topK}")
    header.append(f"Cross-model key for overlap/calibration: {args.cross_key}")
    header.append("")
    header_text = "\n".join(header)

    # Tee output to file (UTF-8)
    with open(args.output, "w", encoding="utf-8") as f:
        def out(s: str = "") -> None:
            print(s)
            f.write(s + "\n")

        out(header_text)

        out("Running REAL batches...")
        real = run_batches(
            "REAL",
            n_qubits=args.n_qubits,
            n_terms=args.n_terms,
            seeds_per_batch=args.seeds_per_batch,
            n_batches=args.batches,
            base_seed=args.base_seed,
            batch_stride=args.batch_stride,
            eps_neighbor=args.eps_neighbor,
            dom_threshold=args.dom_threshold,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=list(args.times),
            stable_frac=args.stable_frac,
            topK=args.topK,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            q_bins=args.q_bins,
            bootstrap=args.bootstrap,
        )

        out("")
        out("Running NULL_HAAR_BASIS batches (spectrum-matched)...")
        null = run_batches(
            "NULL_HAAR_BASIS",
            n_qubits=args.n_qubits,
            n_terms=args.n_terms,
            seeds_per_batch=args.seeds_per_batch,
            n_batches=args.batches,
            base_seed=args.base_seed + 12345,   # offset to avoid any accidental correlation
            batch_stride=args.batch_stride,
            eps_neighbor=args.eps_neighbor,
            dom_threshold=args.dom_threshold,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=list(args.times),
            stable_frac=args.stable_frac,
            topK=args.topK,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            q_bins=args.q_bins,
            bootstrap=args.bootstrap,
        )

        out("")
        # Per-batch summaries
        for b in range(args.batches):
            out("----------------------------------------------")
            out(f"Batch {b+1}/{args.batches}")
            if b < len(real):
                R = real[b]
                out(f"Model: REAL | candidates={R.n_candidates}")
                out(f"  score(mean/median/max)={R.score_stats[0]:.3f}/{R.score_stats[1]:.3f}/{R.score_stats[2]:.3f}")
                out(f"  leakage(mean/median/min)={R.leak_stats[0]:.3f}/{R.leak_stats[1]:.3f}/{R.leak_stats[2]:.3f}")
                out(f"  entropy(mean/median/max)={R.ent_stats[0]:.3f}/{R.ent_stats[1]:.3f}/{R.ent_stats[2]:.3f}")
                out(f"  dom_count(median)={R.dom_median:.1f}")
                out(f"  entropy effect (stable-overall)={R.effect_entropy_bits:+.3f} bits  CI95={R.effect_ci}")
                out("")
                out(format_top(R.top_abs, "REAL, SigAbs"))
                out("")
                out(format_top(R.top_q, "REAL, SigQ"))
                out("")

            if b < len(null):
                N = null[b]
                out(f"Model: NULL_HAAR_BASIS | candidates={N.n_candidates}")
                out(f"  score(mean/median/max)={N.score_stats[0]:.3f}/{N.score_stats[1]:.3f}/{N.score_stats[2]:.3f}")
                out(f"  leakage(mean/median/min)={N.leak_stats[0]:.3f}/{N.leak_stats[1]:.3f}/{N.leak_stats[2]:.3f}")
                out(f"  entropy(mean/median/max)={N.ent_stats[0]:.3f}/{N.ent_stats[1]:.3f}/{N.ent_stats[2]:.3f}")
                out(f"  dom_count(median)={N.dom_median:.1f}")
                out(f"  entropy effect (stable-overall)={N.effect_entropy_bits:+.3f} bits  CI95={N.effect_ci}")
                out("")
                out(format_top(N.top_abs, "NULL, SigAbs"))
                out("")
                out(format_top(N.top_q, "NULL, SigQ"))
                out("")

        out("----------------------------------------------")
        out(cross_model_calibration(real, null, key_mode=args.cross_key, topK=args.topK))

        out("=== End of v10 ===")


if __name__ == "__main__":
    main()
