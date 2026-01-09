#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v11.py

Purpose
-------
Convergence + calibration harness for the v0_6.1-style "signature family" pipeline,
building directly on v10.

What v11 fixes (motivated by v10 output)
---------------------------------------
v10 sometimes produced dom_count(median)=0 and leakage(median)=1.000 for REAL.
That is a red flag: it typically means the "dominant-mask" used for leakage was empty
(because the fixed probability threshold was too strict for the chosen psi0).

If the mask is empty, the leakage proxy degenerates to L(t)=1 for all t, and the
interestingness score is distorted (dom=0 gives an artificially high dom_term).

v11 therefore:
  1) Replaces "dominant_mask by fixed probability threshold" with a *mass-based*
     dominant set: select the smallest set of basis states whose cumulative probability
     mass in |psi0|^2 reaches keep_mass (default 0.90). This guarantees a non-empty mask.
  2) Enforces dom_count >= 1 and reports dom_count statistics that are meaningful.
  3) Runs REAL and NULL in *paired batches using the same seeds* (lower variance, fairer).
  4) Introduces SigQ_GLOBAL: entropy quantile bins computed from the *pooled* entropy
     values of REAL+NULL within the same batch, so cross-model entropy bins are aligned.

Key idea
--------
Treat a "signature family" as:
  sig = (dom_count, entropy_bin_i, entropy_bin_j, leak_bin)

and test whether stable candidates (top stable_frac by score) show repeatable families
across batches, and whether REAL differs from a spectrum-matched NULL.

Dependencies
------------
Only numpy. No Qiskit.

Outputs
-------
- Per batch: summary stats for REAL and NULL.
- Top families under SigAbs and SigQ_GLOBAL (recommended for cross-model comparison).
- Overlap diagnostics (Jaccard) across batches and across models.
- Distribution-level entropy effect: median(entropy) stable - median(entropy) overall (with bootstrap CI).
"""

from __future__ import annotations

import argparse
import math
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Iterable
from collections import Counter

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def stable_hash_int(s: str) -> int:
    """Deterministic int hash for reproducibility independent of Python's hash randomization."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def bin_index(x: float, step: float) -> int:
    if step <= 0:
        raise ValueError("step must be > 0")
    return int(math.floor(float(x) / float(step) + 1e-12))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


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


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


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
    """Haar-random unitary via QR of complex Gaussian."""
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph
    return Q


def null_haar_basis_hamiltonian(evals: np.ndarray, seed: int) -> np.ndarray:
    """Spectrum-matched null: randomize eigenvectors Haar-uniformly with per-seed RNG."""
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
    """
    Choose the smallest set of basis states whose cumulative probability reaches keep_mass.
    Guarantees at least one basis state is selected.
    """
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


def time_evolve_state(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * float(t))
    psi_t = evecs @ (phases * (evecs.conj().T @ psi0))
    return psi_t


def leakage_proxy(H: np.ndarray, i: int, j: int, evecs: np.ndarray, times: List[float], keep_mass: float) -> Tuple[float, int]:
    """
    Leakage proxy:
      - psi0 = (|psi_i> + |psi_j>)/sqrt(2)
      - keep_mask = mass-based dominant set on |psi0|^2
      - leakage(t) = 1 - probability mass inside keep_mask
      - return mean leakage over times and dom_count = |keep_mask|
    """
    psi0 = (evecs[:, i] + evecs[:, j]) / np.sqrt(2.0)
    p0 = np.abs(psi0) ** 2
    p0 = p0 / max(1e-12, float(np.sum(p0)))
    keep_mask = dominant_mask_by_mass(p0, keep_mass=keep_mass)
    dom = int(np.sum(keep_mask))
    dom = max(1, dom)
    leaks = []
    for t in times:
        psi_t = time_evolve_state(H, psi0, t)
        leaks.append(leakage_proxy_from_state(psi_t, keep_mask))
    return float(np.mean(leaks)), dom


# ----------------------------
# Signatures / Candidates
# ----------------------------

SigAbs = Tuple[int, int, int, int]       # (dom, ent_bin_i, ent_bin_j, leak_bin)
SigQGlobal = Tuple[int, int, int, int]   # (dom, ent_q_i, ent_q_j, leak_bin)


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


def interestingness_score(dom: int, leak: float, dom_weight: float = 0.6, leak_weight: float = 0.4) -> float:
    """
    Simple monotone score in [0,1] (clamped):
      - prefers small dom (compressible support)
      - prefers small leak (dynamic stability)
    """
    dom = max(1, int(dom))
    dom_term = 1.0 / (1.0 + float(dom))   # dom=1 -> 0.5, dom=2 -> 0.333, ...
    leak_term = 1.0 - float(leak)         # leak=0 -> 1, leak=1 -> 0
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
    SigQGlobal is assigned later at the batch level.
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
        leak, dom = leakage_proxy(H, i, j, evecs, times, keep_mass=keep_mass)
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
                sig_qg=(0, 0, 0, 0),  # filled later
            )
        )
    return cands


def assign_sig_q_global(cands: List[Candidate], edges: np.ndarray, q_bins: int, leak_step: float) -> List[Candidate]:
    out: List[Candidate] = []
    for c in cands:
        sig_qg = signature_key_q_global(c.dom, c.ent_i, c.ent_j, c.leak, edges, q_bins, leak_step)
        out.append(Candidate(**{**c.__dict__, "sig_qg": sig_qg}))
    return out


# ----------------------------
# Enrichment + batch summaries
# ----------------------------

def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = scores.size
    k = max(1, int(math.ceil(float(stable_frac) * n)))
    idx = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[idx[:k]] = True
    return mask


def enrichment_rows(
    sigs: List[Tuple[int, int, int, int]],
    stable_mask: np.ndarray,
    alpha: float,
) -> List[Tuple[Tuple[int, int, int, int], int, int, float]]:
    overall = Counter(sigs)
    stable = Counter([sigs[i] for i in range(len(sigs)) if stable_mask[i]])

    n_all = len(sigs)
    n_stable = int(np.sum(stable_mask))
    stable_frac = n_stable / max(1, n_all)

    K = max(1, len(overall))
    rows = []
    for sig, o in overall.items():
        st = stable.get(sig, 0)
        p_all = (o + alpha) / (n_all + alpha * K)
        p_st = (st + alpha) / (n_stable + alpha * K)
        enr = (p_st / p_all) / max(1e-12, stable_frac)
        rows.append((sig, int(o), int(st), float(enr)))

    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def top_k_families(rows: List[Tuple[Tuple[int, int, int, int], int, int, float]], k: int, min_overall: int, min_stable: int):
    out = []
    for sig, o, st, enr in rows:
        if o >= min_overall and st >= min_stable:
            out.append((sig, o, st, enr))
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


# ----------------------------
# Batch runner (paired REAL + NULL)
# ----------------------------

@dataclass
class BatchResult:
    batch_id: int
    seed_offset: int
    model: str
    n_candidates: int
    score_stats: Tuple[float, float, float]
    leak_stats: Tuple[float, float, float]
    ent_stats: Tuple[float, float, float]
    dom_stats: Tuple[float, float, float]  # mean, median, max
    top_abs: List[Tuple[SigAbs, int, int, float]]
    top_qg: List[Tuple[SigQGlobal, int, int, float]]
    effect_entropy_bits: float
    effect_ci: Tuple[float, float]
    top_keys_qg: List[SigQGlobal]


def compute_batch_result(
    *,
    batch_id: int,
    seed_offset: int,
    model: str,
    cands: List[Candidate],
    stable_frac: float,
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    bootstrap: int,
) -> BatchResult:
    if not cands:
        return BatchResult(
            batch_id=batch_id, seed_offset=seed_offset, model=model,
            n_candidates=0,
            score_stats=(0.0, 0.0, 0.0),
            leak_stats=(0.0, 0.0, 0.0),
            ent_stats=(0.0, 0.0, 0.0),
            dom_stats=(0.0, 0.0, 0.0),
            top_abs=[], top_qg=[],
            effect_entropy_bits=0.0, effect_ci=(0.0, 0.0),
            top_keys_qg=[],
        )

    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak for c in cands], dtype=float)
    doms = np.array([c.dom for c in cands], dtype=float)
    ent_pool = np.array([c.ent_i for c in cands] + [c.ent_j for c in cands], dtype=float)

    mask = stable_mask_from_scores(scores, stable_frac)

    sigs_abs = [c.sig_abs for c in cands]
    rows_abs = enrichment_rows(sigs_abs, mask, alpha=alpha)
    top_abs = top_k_families(rows_abs, k=topK, min_overall=min_overall, min_stable=min_stable)

    sigs_qg = [c.sig_qg for c in cands]
    rows_qg = enrichment_rows(sigs_qg, mask, alpha=alpha)
    top_qg = top_k_families(rows_qg, k=topK, min_overall=min_overall, min_stable=min_stable)

    rng_eff = rng_from_seed(stable_hash_int(f"{model}|batch{batch_id}|eff"))
    eff, ci = summarize_entropy_effect(ent_pool, np.repeat(mask, 2), rng_eff, B=bootstrap)

    return BatchResult(
        batch_id=batch_id, seed_offset=seed_offset, model=model,
        n_candidates=len(cands),
        score_stats=(float(scores.mean()), float(np.median(scores)), float(scores.max())),
        leak_stats=(float(leaks.mean()), float(np.median(leaks)), float(leaks.min())),
        ent_stats=(float(ent_pool.mean()), float(np.median(ent_pool)), float(ent_pool.max())),
        dom_stats=(float(doms.mean()), float(np.median(doms)), float(doms.max())),
        top_abs=top_abs,
        top_qg=top_qg,
        effect_entropy_bits=eff,
        effect_ci=ci,
        top_keys_qg=[x[0] for x in top_qg],
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
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    q_bins: int,
    bootstrap: int,
) -> Tuple[List[BatchResult], List[BatchResult]]:
    real_results: List[BatchResult] = []
    null_results: List[BatchResult] = []

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
        edges = make_quantile_edges(pooled_ent, q_bins=q_bins) if pooled_ent.size else make_quantile_edges(np.array([0.0]), q_bins=q_bins)

        real_cands = assign_sig_q_global(real_cands, edges, q_bins=q_bins, leak_step=leak_step)
        null_cands = assign_sig_q_global(null_cands, edges, q_bins=q_bins, leak_step=leak_step)

        elapsed = time.time() - t0
        print(f"Batch {b+1}/{n_batches} generated: REAL={len(real_cands)} NULL={len(null_cands)} (elapsed {elapsed:.1f}s)")

        real_results.append(
            compute_batch_result(
                batch_id=b,
                seed_offset=offset,
                model="REAL",
                cands=real_cands,
                stable_frac=stable_frac,
                topK=topK,
                min_overall=min_overall,
                min_stable=min_stable,
                alpha=alpha,
                bootstrap=bootstrap,
            )
        )
        null_results.append(
            compute_batch_result(
                batch_id=b,
                seed_offset=offset,
                model="NULL_HAAR_BASIS",
                cands=null_cands,
                stable_frac=stable_frac,
                topK=topK,
                min_overall=min_overall,
                min_stable=min_stable,
                alpha=alpha,
                bootstrap=bootstrap,
            )
        )

    return real_results, null_results


# ----------------------------
# Reporting
# ----------------------------

def format_top(top: List[Tuple[Tuple[int, int, int, int], int, int, float]], label: str, show: int = 10) -> str:
    lines = []
    lines.append(f"Top signature families ({label}):")
    lines.append("Format: sig=(dom, e1, e2, leak_bin) | overall | stable | enrichment")
    for (sig, o, st, enr) in top[:show]:
        lines.append(f"  {sig} | {o:5d} | {st:5d} | {enr:10.2f}x")
    if len(top) > show:
        lines.append(f"  ... ({len(top)} total, showing {show})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0_7 convergence harness v11 (mass-based dom mask + pooled quantile bins).")
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
    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=3)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--q_bins", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--output", type=str, default="v0_7_convergence_families_v11_output.txt")
    args = ap.parse_args()

    header = []
    header.append("=== v0_7: Convergence + Baseline Calibration (v11) ===")
    header.append(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.n_terms}")
    header.append(f"Batches: {args.batches} × {args.seeds_per_batch} seeds (base_seed={args.base_seed}, stride={args.batch_stride})")
    header.append(f"Neighbor eps={args.eps_neighbor:.3f}")
    header.append(f"Dominant set: keep_mass={args.keep_mass:.2f} (mass-based; guarantees non-empty mask)")
    header.append(f"Bins (SigAbs): ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
    header.append(f"Bins (SigQ_GLOBAL): q_bins={args.q_bins} computed from pooled REAL+NULL entropies per batch")
    header.append(f"Leakage proxy times={args.times} | stable_frac={args.stable_frac:.3f} | topK={args.topK}")
    header.append("")
    header_text = "\n".join(header)

    with open(args.output, "w", encoding="utf-8") as f:
        def out(s: str = "") -> None:
            print(s)
            f.write(s + "\n")

        out(header_text)

        real, null = run_paired_batches(
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
            topK=args.topK,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            q_bins=args.q_bins,
            bootstrap=args.bootstrap,
        )

        out("")
        for b in range(args.batches):
            out("----------------------------------------------")
            out(f"Batch {b+1}/{args.batches} (seed_offset={real[b].seed_offset})")

            R = real[b]
            out(f"Model: REAL | candidates={R.n_candidates}")
            out(f"  score(mean/median/max)={R.score_stats[0]:.3f}/{R.score_stats[1]:.3f}/{R.score_stats[2]:.3f}")
            out(f"  leakage(mean/median/min)={R.leak_stats[0]:.3f}/{R.leak_stats[1]:.3f}/{R.leak_stats[2]:.3f}")
            out(f"  entropy(mean/median/max)={R.ent_stats[0]:.3f}/{R.ent_stats[1]:.3f}/{R.ent_stats[2]:.3f}")
            out(f"  dom_count(mean/median/max)={R.dom_stats[0]:.2f}/{R.dom_stats[1]:.1f}/{R.dom_stats[2]:.0f}")
            out(f"  entropy effect (stable-overall)={R.effect_entropy_bits:+.3f} bits  CI95={R.effect_ci}")
            out("")
            out(format_top(R.top_abs, "REAL, SigAbs"))
            out("")
            out(format_top(R.top_qg, "REAL, SigQ_GLOBAL"))
            out("")

            N = null[b]
            out(f"Model: NULL_HAAR_BASIS | candidates={N.n_candidates}")
            out(f"  score(mean/median/max)={N.score_stats[0]:.3f}/{N.score_stats[1]:.3f}/{N.score_stats[2]:.3f}")
            out(f"  leakage(mean/median/min)={N.leak_stats[0]:.3f}/{N.leak_stats[1]:.3f}/{N.leak_stats[2]:.3f}")
            out(f"  entropy(mean/median/max)={N.ent_stats[0]:.3f}/{N.ent_stats[1]:.3f}/{N.ent_stats[2]:.3f}")
            out(f"  dom_count(mean/median/max)={N.dom_stats[0]:.2f}/{N.dom_stats[1]:.1f}/{N.dom_stats[2]:.0f}")
            out(f"  entropy effect (stable-overall)={N.effect_entropy_bits:+.3f} bits  CI95={N.effect_ci}")
            out("")
            out(format_top(N.top_abs, "NULL, SigAbs"))
            out("")
            out(format_top(N.top_qg, "NULL, SigQ_GLOBAL"))
            out("")

            ov = jaccard(R.top_keys_qg, N.top_keys_qg)
            out(f"Batch {b+1}: Jaccard(Top-{args.topK}) REAL vs NULL (SigQ_GLOBAL) = {ov:.3f}")
            out("")

        out("----------------------------------------------")
        out("=== Convergence diagnostics (REAL, SigQ_GLOBAL) ===")
        real_sets = [set(r.top_keys_qg) for r in real]
        for i in range(len(real_sets)):
            for j in range(i + 1, len(real_sets)):
                out(f"REAL batch overlap: Jaccard(Top-{args.topK}) batch{i+1} vs batch{j+1} = {jaccard(real_sets[i], real_sets[j]):.3f}")

        out("")
        out("=== Distribution-level effect (entropy, bits): median(stable) - median(overall) ===")
        effR = np.array([r.effect_entropy_bits for r in real], dtype=float)
        effN = np.array([n.effect_entropy_bits for n in null], dtype=float)
        out(f"  REAL: mean={effR.mean():+.3f} across {len(effR)} batches")
        out(f"  NULL: mean={effN.mean():+.3f} across {len(effN)} batches")
        out(f"  REAL-NULL (batch means): {(effR.mean()-effN.mean()):+.3f}")

        out("")
        out("=== Notes (scientific reading) ===")
        out("1) v11 removes the empty-dominant-mask failure mode: dom_count>=1 and leakage is no longer forced to 1.0 by construction.")
        out("2) Cross-model comparisons should use SigQ_GLOBAL (pooled quantile bins) rather than per-model quantiles, because it aligns entropy bins across REAL and NULL.")
        out("3) If REAL-vs-NULL top-family overlap remains near zero under v11, that is informative: it suggests the stability-selected families are model-dependent, not a binning artifact.")
        out("")
        out("=== End of v11 ===")


if __name__ == "__main__":
    main()
