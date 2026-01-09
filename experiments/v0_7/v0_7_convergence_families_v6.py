#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v6.py
================================

Purpose
-------
This program is a "convergence + replicability" wrapper around the v0.6.1/v0.7
signature-family logic.

It runs multiple independent batches (e.g., 3 x 5000 seeds) and reports:

1) Top-family stability across batches (overlap / Jaccard for Top-K families)
2) Distribution-level effect sizes (e.g., median entropy-bin REAL vs NULL) with bootstrap CI
3) Cross-model enrichment with smoothing + minimum-count filtering (readable ratios)
4) Optional hold-out replication inside each batch (split by seed parity)

Core scientific logic is unchanged:
- Random Pauli-sum Hamiltonians (n qubits, fixed number of Pauli terms)
- Near-degenerate neighbor detection (|Ei - Ej| < eps)
- Eigenvector structure in computational basis:
  dominant basis-state count + amplitude entropy
- Leakage proxy from exact time evolution in eigenbasis at fixed times
- Candidate score = (1 - leakage) / sqrt(dom_count)
- Signature family key = (dom_count, ent_bin_i, ent_bin_j, leak_bin)

NULL model:
- spectrum matched: keep eigenvalues from REAL, but randomize eigenvectors using a Haar unitary

This file writes console output AND (optionally) mirrors it to a UTF-8 txt file.

Notes
-----
- ASCII-only console output to avoid Windows cp1252 issues.
- The "concentration" of a signature is P(stable | signature) with Beta smoothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np


# ----------------------------
# Output mirroring (safe)
# ----------------------------

class Tee:
    """Mirror print output to stdout and an optional UTF-8 file, safely."""
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._fh = None

    def __enter__(self):
        if self.path:
            self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None
        # do not suppress exceptions
        return False

    def write_line(self, s: str = "") -> None:
        # ASCII-safe: enforce str, but do not inject non-ascii characters from ourselves
        sys.stdout.write(s + "\n")
        if self._fh:
            self._fh.write(s + "\n")

    def banner(self, s: str) -> None:
        self.write_line(s)
        self.write_line("-" * len(s))


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class SignatureKey:
    dom: int
    ent_i_bin: int
    ent_j_bin: int
    leak_bin: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.dom, self.ent_i_bin, self.ent_j_bin, self.leak_bin)


@dataclass
class Candidate:
    sig: SignatureKey
    score: float
    leakage: float
    ent_i: float
    ent_j: float
    dom_count: int
    seed: int  # seed that generated the Hamiltonian


# ----------------------------
# Utilities
# ----------------------------

def bin_int(x: float, step: float) -> int:
    if step <= 0:
        return 0
    return int(math.floor(float(x) / float(step) + 1e-12))


def entropy_from_probs(p: np.ndarray) -> float:
    """Shannon entropy (base-2) of a probability vector."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def dominant_index_set(state_vec: np.ndarray, threshold: float) -> List[int]:
    """Return indices |x> where |amp|^2 >= threshold."""
    probs = np.abs(state_vec) ** 2
    return [int(i) for i, v in enumerate(probs) if float(v) >= threshold]


def exact_time_evolution_leakage(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    dom_set: Sequence[int],
    times: Sequence[float],
) -> float:
    """
    Leakage proxy under exact time evolution:
    - Start state: uniform superposition over dominant computational basis states.
    - Evolve in time exactly using spectral decomposition.
    - Leakage = 1 - average_{t in times} sum_{x in dom_set} |<x|psi(t)>|^2
    """
    d = eigvecs.shape[0]
    if len(dom_set) == 0:
        return 1.0

    psi0 = np.zeros((d,), dtype=complex)
    amp = 1.0 / math.sqrt(len(dom_set))
    for idx in dom_set:
        psi0[idx] = amp

    # coefficients in eigenbasis
    c = eigvecs.conj().T @ psi0

    keep = np.array(dom_set, dtype=int)
    keep = keep[(keep >= 0) & (keep < d)]
    if keep.size == 0:
        return 1.0

    keep_prob_sum = 0.0
    for t in times:
        phase = np.exp(-1j * eigvals * float(t))
        psi_t = eigvecs @ (c * phase)
        probs = np.abs(psi_t[keep]) ** 2
        keep_prob_sum += float(np.sum(probs))

    keep_prob_avg = keep_prob_sum / max(1, len(times))
    leakage = max(0.0, min(1.0, 1.0 - keep_prob_avg))
    # guard tiny negative -0.0 from float noise
    if abs(leakage) < 1e-15:
        leakage = 0.0
    return leakage


# ----------------------------
# Hamiltonian generation (Pauli-sum)
# ----------------------------

PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def kron_n(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def random_pauli_string(rng: np.random.Generator, n: int) -> str:
    letters = ["I", "X", "Y", "Z"]
    return "".join(rng.choice(letters) for _ in range(n))


def build_random_hamiltonian(n_qubits: int, num_terms: int, rng: np.random.Generator) -> np.ndarray:
    """Dense Hamiltonian H = sum_k c_k P_k (small n only; for n=3 d=8)."""
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)

    # avoid all-identity too often; allow but resample if all I
    for _ in range(num_terms):
        s = random_pauli_string(rng, n_qubits)
        if set(s) == {"I"}:
            # resample once
            s = random_pauli_string(rng, n_qubits)
        ops = [PAULI_MATS[ch] for ch in s]
        P = kron_n(ops)
        c = float(rng.normal(loc=0.0, scale=1.0))
        H = H + c * P

    # Hermitian cleanup (should already be Hermitian)
    H = 0.5 * (H + H.conj().T)
    return H


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-random unitary via QR decomposition of a complex Ginibre matrix."""
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    # make diag(r) have positive real part
    diag = np.diag(r)
    ph = diag / np.abs(diag)
    q = q * ph
    return q


# ----------------------------
# Candidate discovery
# ----------------------------

def near_degenerate_neighbor_pairs(eigvals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    """Only compare neighbors in sorted spectrum for O(d log d) behavior."""
    idx = np.argsort(eigvals)
    vals = eigvals[idx]
    pairs: List[Tuple[int, int]] = []
    for k in range(len(vals) - 1):
        i = int(idx[k])
        j = int(idx[k + 1])
        if abs(float(eigvals[i] - eigvals[j])) < float(eps):
            pairs.append((i, j))
    return pairs


def make_signature(dom_count: int, ent_i: float, ent_j: float, leakage: float, ent_step: float, leak_step: float) -> SignatureKey:
    ei = bin_int(ent_i, ent_step)
    ej = bin_int(ent_j, ent_step)
    if ej < ei:
        ei, ej = ej, ei
    return SignatureKey(dom=dom_count, ent_i_bin=ei, ent_j_bin=ej, leak_bin=bin_int(leakage, leak_step))


def scan_one_hamiltonian(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    seed: int,
    eps: float,
    dominant_threshold: float,
    ent_step: float,
    leak_step: float,
    times: Sequence[float],
) -> List[Candidate]:
    d = eigvecs.shape[0]
    assert d == eigvecs.shape[1]
    pairs = near_degenerate_neighbor_pairs(eigvals, eps)

    cands: List[Candidate] = []
    for (i, j) in pairs:
        psi_i = eigvecs[:, i]
        psi_j = eigvecs[:, j]

        pi = np.abs(psi_i) ** 2
        pj = np.abs(psi_j) ** 2
        ent_i = entropy_from_probs(pi)
        ent_j = entropy_from_probs(pj)

        dom_i = dominant_index_set(psi_i, dominant_threshold)
        dom_j = dominant_index_set(psi_j, dominant_threshold)
        dom_set = sorted(set(dom_i).union(dom_j))
        dom_count = len(dom_set)

        leakage = exact_time_evolution_leakage(eigvals, eigvecs, dom_set, times)
        stability = max(0.0, 1.0 - leakage)
        penalty = 1.0 / math.sqrt(max(1, dom_count))
        score = float(min(1.0, stability * penalty))

        sig = make_signature(dom_count, ent_i, ent_j, leakage, ent_step, leak_step)
        cands.append(Candidate(sig=sig, score=score, leakage=leakage, ent_i=ent_i, ent_j=ent_j, dom_count=dom_count, seed=seed))

    return cands


def scan_batch(
    seed_start: int,
    n_seeds: int,
    n_qubits: int,
    num_terms: int,
    eps: float,
    dominant_threshold: float,
    ent_step: float,
    leak_step: float,
    times: Sequence[float],
    rng_model: np.random.Generator,
    rng_null: np.random.Generator,
) -> Tuple[List[Candidate], List[Candidate]]:
    """Return (REAL candidates, NULL_HAAR_BASIS candidates) for a batch."""
    real_all: List[Candidate] = []
    null_all: List[Candidate] = []

    d = 2 ** n_qubits

    for s in range(seed_start, seed_start + n_seeds):
        rng_local = np.random.default_rng(int(s))
        H = build_random_hamiltonian(n_qubits, num_terms, rng_local)
        eigvals, eigvecs = np.linalg.eigh(H)

        # REAL
        real_all.extend(
            scan_one_hamiltonian(eigvals, eigvecs, s, eps, dominant_threshold, ent_step, leak_step, times)
        )

        # NULL: spectrum-matched, eigenvectors randomized (Haar basis)
        U = haar_unitary(d, rng_null)
        null_eigvecs = U  # columns are random orthonormal eigenvectors
        null_all.extend(
            scan_one_hamiltonian(eigvals, null_eigvecs, s, eps, dominant_threshold, ent_step, leak_step, times)
        )

    return real_all, null_all


# ----------------------------
# Family statistics / enrichment
# ----------------------------

def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = scores.size
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.floor(float(stable_frac) * n)))
    # higher score = more stable
    thresh = np.partition(scores, n - k)[n - k]
    return scores >= thresh


def family_table(
    sigs: Sequence[SignatureKey],
    stable_mask: np.ndarray,
    alpha: float,
    beta: float,
    min_overall: int,
) -> Dict[SignatureKey, Dict[str, float]]:
    """
    Compute counts and smoothed concentration for each signature.
    concentration = P(stable | sig), with Beta(alpha, beta) smoothing:
      conc = (k + alpha) / (n + alpha + beta)
    """
    overall: Dict[SignatureKey, int] = {}
    stable: Dict[SignatureKey, int] = {}

    for idx, sig in enumerate(sigs):
        overall[sig] = overall.get(sig, 0) + 1
        if bool(stable_mask[idx]):
            stable[sig] = stable.get(sig, 0) + 1

    out: Dict[SignatureKey, Dict[str, float]] = {}
    for sig, n in overall.items():
        if n < min_overall:
            continue
        k = stable.get(sig, 0)
        conc = (k + alpha) / (n + alpha + beta)
        out[sig] = {
            "overall": float(n),
            "stable": float(k),
            "conc": float(conc),
        }
    return out


def topk_by_conc(table: Dict[SignatureKey, Dict[str, float]], topk: int) -> List[Tuple[SignatureKey, Dict[str, float]]]:
    rows = list(table.items())
    rows.sort(key=lambda kv: (kv[1]["conc"], kv[1]["stable"], kv[1]["overall"]), reverse=True)
    return rows[: max(1, int(topk))]


def overlap_and_jaccard(a: Sequence[SignatureKey], b: Sequence[SignatureKey]) -> Tuple[int, float]:
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb) if (sa or sb) else 1
    return inter, inter / union


def bootstrap_ci_median_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha_ci: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for median(x) - median(y)."""
    if x.size == 0 or y.size == 0:
        return (float("nan"), float("nan"))
    diffs = []
    for _ in range(int(n_boot)):
        xb = x[rng.integers(0, x.size, size=x.size)]
        yb = y[rng.integers(0, y.size, size=y.size)]
        diffs.append(float(np.median(xb) - np.median(yb)))
    diffs.sort()
    lo = diffs[int((alpha_ci / 2) * len(diffs))]
    hi = diffs[int((1 - alpha_ci / 2) * len(diffs)) - 1]
    return lo, hi


def bootstrap_ci_ratio(
    sig: SignatureKey,
    sigs_real: Sequence[SignatureKey],
    stable_real: np.ndarray,
    sigs_null: Sequence[SignatureKey],
    stable_null: np.ndarray,
    alpha_smooth: float,
    beta_smooth: float,
    n_boot: int,
    rng: np.random.Generator,
    alpha_ci: float = 0.05,
) -> Tuple[float, float]:
    """
    Bootstrap CI for ratio of concentrations:
      ratio = conc_real / conc_null
    where conc_* is smoothed P(stable|sig).
    Resample candidates with replacement independently in REAL and NULL.
    """
    r = np.array([s.as_tuple() for s in sigs_real], dtype=int)
    n = np.array([s.as_tuple() for s in sigs_null], dtype=int)
    target = np.array(sig.as_tuple(), dtype=int)

    def conc_from_sample(sample_sigs: np.ndarray, sample_stable: np.ndarray) -> float:
        mask = np.all(sample_sigs == target, axis=1)
        tot = int(np.sum(mask))
        if tot == 0:
            # no evidence in this bootstrap sample; return prior mean
            return float(alpha_smooth / (alpha_smooth + beta_smooth))
        k = int(np.sum(sample_stable[mask]))
        return float((k + alpha_smooth) / (tot + alpha_smooth + beta_smooth))

    ratios = []
    for _ in range(int(n_boot)):
        idx_r = rng.integers(0, r.shape[0], size=r.shape[0])
        idx_n = rng.integers(0, n.shape[0], size=n.shape[0])

        cr = conc_from_sample(r[idx_r], stable_real[idx_r])
        cn = conc_from_sample(n[idx_n], stable_null[idx_n])

        ratios.append(float(cr / max(1e-12, cn)))

    ratios.sort()
    lo = ratios[int((alpha_ci / 2) * len(ratios))]
    hi = ratios[int((1 - alpha_ci / 2) * len(ratios)) - 1]
    return lo, hi


def cross_model_top_families(
    table_real: Dict[SignatureKey, Dict[str, float]],
    table_null: Dict[SignatureKey, Dict[str, float]],
    topk: int,
) -> List[Tuple[SignatureKey, float, float, float]]:
    """
    Return list of (sig, conc_real, conc_null, ratio) for signatures present in either table.
    """
    keys = set(table_real.keys()) | set(table_null.keys())
    rows = []
    for sig in keys:
        cr = table_real.get(sig, {}).get("conc", 0.0)
        cn = table_null.get(sig, {}).get("conc", 0.0)
        ratio = float(cr / max(1e-12, cn))
        rows.append((sig, float(cr), float(cn), ratio))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows[: max(1, int(topk))]


# ----------------------------
# Hold-out replication (inside a batch)
# ----------------------------

def holdout_topk_overlap(
    cands: Sequence[Candidate],
    stable_frac: float,
    alpha_smooth: float,
    beta_smooth: float,
    min_overall: int,
    topk: int,
) -> Tuple[int, float]:
    """
    Split by seed parity:
      train = even seeds, test = odd seeds (or vice versa depending on seed_start)
    Compute Top-K families by smoothed concentration in each split and return:
      (overlap_count, jaccard)
    """
    if not cands:
        return 0, float("nan")

    train = [c for c in cands if (c.seed % 2 == 0)]
    test = [c for c in cands if (c.seed % 2 == 1)]
    if len(train) < 10 or len(test) < 10:
        return 0, float("nan")

    def topk_sigs(sub: List[Candidate]) -> List[SignatureKey]:
        sigs = [c.sig for c in sub]
        scores = np.array([c.score for c in sub], dtype=float)
        stable = stable_mask_from_scores(scores, stable_frac)
        tab = family_table(sigs, stable, alpha_smooth, beta_smooth, min_overall)
        return [s for s, _ in topk_by_conc(tab, topk)]

    a = topk_sigs(train)
    b = topk_sigs(test)
    return overlap_and_jaccard(a, b)


# ----------------------------
# Main analysis driver
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.7 convergence analysis (v6): batch stability + cross-model calibration.")
    p.add_argument("--n_qubits", type=int, default=3)
    p.add_argument("--num_terms", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--dominant_threshold", type=float, default=0.25)
    p.add_argument("--ent_step", type=float, default=0.10)
    p.add_argument("--leak_step", type=float, default=0.05)
    p.add_argument("--times", type=str, default="0.5,1.0,1.5", help="Comma-separated times for leakage proxy.")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--seeds_per_batch", type=int, default=5000)
    p.add_argument("--seed0", type=int, default=0, help="Start seed for batch 0. Batch b starts at seed0 + b*seeds_per_batch.")
    p.add_argument("--stable_frac", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=25)
    p.add_argument("--min_overall", type=int, default=5, help="Minimum overall count per signature to be reported.")
    p.add_argument("--alpha_smooth", type=float, default=0.5, help="Beta prior alpha for P(stable|sig).")
    p.add_argument("--beta_smooth", type=float, default=0.5, help="Beta prior beta for P(stable|sig).")
    p.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples (CI).")
    p.add_argument("--bootstrap_seed", type=int, default=123)
    p.add_argument("--null_seed", type=int, default=999, help="Seed for NULL Haar unitary RNG stream.")
    p.add_argument("--output", type=str, default="v0_7_convergence_families_v6_output.txt", help="UTF-8 output txt file.")
    p.add_argument("--no_holdout", action="store_true", help="Disable hold-out replication inside each batch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    times = [float(x.strip()) for x in args.times.split(",") if x.strip()]
    d = 2 ** args.n_qubits

    rng_null = np.random.default_rng(int(args.null_seed))
    rng_ci = np.random.default_rng(int(args.bootstrap_seed))

    with Tee(args.output) as out:
        out.banner("=== v0_7 Convergence: Signature Families (v6) ===")
        out.write_line(f"Qubits: {args.n_qubits} (d={d}) | terms={args.num_terms}")
        out.write_line(f"Batches: {args.batches} x {args.seeds_per_batch} seeds | seed0={args.seed0}")
        out.write_line(f"Near-degenerate neighbor eps={args.eps:.3f}")
        out.write_line(f"Dominant threshold={args.dominant_threshold:.3f} | ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
        out.write_line(f"Leakage times={times}")
        out.write_line(f"Stable_frac={args.stable_frac:.3f} | topK={args.topk} | min_overall={args.min_overall}")
        out.write_line(f"Smoothing (Beta prior): alpha={args.alpha_smooth} beta={args.beta_smooth} | bootstrap={args.bootstrap}")
        out.write_line(f"Output file: {args.output}")
        out.write_line("")

        # Collect per-batch summaries
        top_real_per_batch: List[List[SignatureKey]] = []
        top_null_per_batch: List[List[SignatureKey]] = []
        cross_top_per_batch: List[List[SignatureKey]] = []

        median_entbin_real: List[float] = []
        median_entbin_null: List[float] = []

        # For aggregated cross-model ratios
        ratio_accum: Dict[SignatureKey, List[float]] = {}
        ratio_ci_accum: Dict[SignatureKey, List[Tuple[float, float]]] = {}

        for b in range(int(args.batches)):
            seed_start = int(args.seed0 + b * args.seeds_per_batch)
            out.banner(f"Batch {b+1}/{args.batches} (seeds {seed_start}..{seed_start + args.seeds_per_batch - 1})")

            t0 = time.time()
            rng_model = np.random.default_rng(seed_start + 1337)  # not used directly (per-seed rng used), but kept for extensibility
            real_cands, null_cands = scan_batch(
                seed_start=seed_start,
                n_seeds=int(args.seeds_per_batch),
                n_qubits=int(args.n_qubits),
                num_terms=int(args.num_terms),
                eps=float(args.eps),
                dominant_threshold=float(args.dominant_threshold),
                ent_step=float(args.ent_step),
                leak_step=float(args.leak_step),
                times=times,
                rng_model=rng_model,
                rng_null=rng_null,
            )
            elapsed = time.time() - t0
            out.write_line(f"REAL candidates: {len(real_cands)} | NULL candidates: {len(null_cands)} | elapsed={elapsed:.1f}s")

            # Stable masks
            real_scores = np.array([c.score for c in real_cands], dtype=float)
            null_scores = np.array([c.score for c in null_cands], dtype=float)
            real_stable = stable_mask_from_scores(real_scores, float(args.stable_frac))
            null_stable = stable_mask_from_scores(null_scores, float(args.stable_frac))

            # Family tables with smoothing
            real_sigs = [c.sig for c in real_cands]
            null_sigs = [c.sig for c in null_cands]
            tab_real = family_table(real_sigs, real_stable, float(args.alpha_smooth), float(args.beta_smooth), int(args.min_overall))
            tab_null = family_table(null_sigs, null_stable, float(args.alpha_smooth), float(args.beta_smooth), int(args.min_overall))

            top_real = [s for s, _ in topk_by_conc(tab_real, int(args.topk))]
            top_null = [s for s, _ in topk_by_conc(tab_null, int(args.topk))]
            top_real_per_batch.append(top_real)
            top_null_per_batch.append(top_null)

            # Cross-model ranking by conc ratio
            cross_rows = cross_model_top_families(tab_real, tab_null, int(args.topk))
            cross_top = [r[0] for r in cross_rows]
            cross_top_per_batch.append(cross_top)

            out.write_line("")
            out.write_line("Top families by concentration P(stable|sig) (REAL):")
            for sig in top_real[:10]:
                v = tab_real[sig]
                out.write_line(f"  sig={sig.as_tuple()} | overall={int(v['overall'])} stable={int(v['stable'])} conc={v['conc']:.3f}")
            out.write_line("Top families by concentration P(stable|sig) (NULL):")
            for sig in top_null[:10]:
                v = tab_null[sig]
                out.write_line(f"  sig={sig.as_tuple()} | overall={int(v['overall'])} stable={int(v['stable'])} conc={v['conc']:.3f}")

            out.write_line("")
            out.write_line("Top cross-model families by ratio conc_REAL / conc_NULL (smoothed):")
            for sig, cr, cn, ratio in cross_rows[:10]:
                out.write_line(f"  sig={sig.as_tuple()} | conc_REAL={cr:.3f} conc_NULL={cn:.3f} ratio={ratio:.2f}")

            # Distribution-level: median entropy-bin in stable candidates (REAL vs NULL)
            # Define per-candidate entropy-bin as min(ent_i_bin, ent_j_bin) (order-invariant, interpretable).
            entbin_real_stable = np.array([min(c.sig.ent_i_bin, c.sig.ent_j_bin) for i, c in enumerate(real_cands) if bool(real_stable[i])], dtype=float)
            entbin_null_stable = np.array([min(c.sig.ent_i_bin, c.sig.ent_j_bin) for i, c in enumerate(null_cands) if bool(null_stable[i])], dtype=float)

            med_r = float(np.median(entbin_real_stable)) if entbin_real_stable.size else float("nan")
            med_n = float(np.median(entbin_null_stable)) if entbin_null_stable.size else float("nan")
            median_entbin_real.append(med_r)
            median_entbin_null.append(med_n)

            ci_lo, ci_hi = bootstrap_ci_median_diff(entbin_real_stable, entbin_null_stable, int(args.bootstrap), rng_ci)
            out.write_line("")
            out.write_line("Stable-set distribution effect (entropy-bin, stable candidates):")
            out.write_line(f"  median_bin(REAL)={med_r:.2f} | median_bin(NULL)={med_n:.2f} | diff={med_r - med_n:.2f} | boot95=[{ci_lo:.2f},{ci_hi:.2f}]")

            # Optional holdout replication
            if not bool(args.no_holdout):
                ov_r, jac_r = holdout_topk_overlap(real_cands, float(args.stable_frac), float(args.alpha_smooth), float(args.beta_smooth), int(args.min_overall), int(args.topk))
                ov_n, jac_n = holdout_topk_overlap(null_cands, float(args.stable_frac), float(args.alpha_smooth), float(args.beta_smooth), int(args.min_overall), int(args.topk))
                out.write_line("")
                out.write_line("Hold-out replication inside batch (Top-K by conc, split by seed parity):")
                out.write_line(f"  REAL: overlap={ov_r}/{args.topk} | jaccard={jac_r:.3f}")
                out.write_line(f"  NULL: overlap={ov_n}/{args.topk} | jaccard={jac_n:.3f}")

            # Accumulate ratios for signatures appearing in cross topK for this batch
            for sig, cr, cn, ratio in cross_rows:
                ratio_accum.setdefault(sig, []).append(float(ratio))
                # CI per signature (ratio)
                try:
                    lo, hi = bootstrap_ci_ratio(
                        sig=sig,
                        sigs_real=real_sigs,
                        stable_real=real_stable.astype(int),
                        sigs_null=null_sigs,
                        stable_null=null_stable.astype(int),
                        alpha_smooth=float(args.alpha_smooth),
                        beta_smooth=float(args.beta_smooth),
                        n_boot=int(args.bootstrap),
                        rng=rng_ci,
                    )
                    ratio_ci_accum.setdefault(sig, []).append((float(lo), float(hi)))
                except Exception:
                    # do not fail the run because a CI failed for a rare sig
                    pass

            out.write_line("")

        # ----------------------------
        # Across-batch stability (Top-K overlap)
        # ----------------------------
        out.banner("Across-batch stability (Top-K overlap / Jaccard)")

        def report_pairwise(label: str, tops: List[List[SignatureKey]]) -> None:
            out.write_line(label)
            B = len(tops)
            for i in range(B):
                for j in range(i + 1, B):
                    inter, jac = overlap_and_jaccard(tops[i], tops[j])
                    out.write_line(f"  batches {i+1}-{j+1}: overlap={inter}/{args.topk} | jaccard={jac:.3f}")
            out.write_line("")

        report_pairwise("REAL Top-K:", top_real_per_batch)
        report_pairwise("NULL Top-K:", top_null_per_batch)
        report_pairwise("CROSS (ratio) Top-K:", cross_top_per_batch)

        # Consensus families (appear in >=2 batches)
        out.banner("Consensus families (appear in >=2 batches of CROSS Top-K)")
        freq: Dict[SignatureKey, int] = {}
        for lst in cross_top_per_batch:
            for sig in set(lst):
                freq[sig] = freq.get(sig, 0) + 1

        consensus = [(sig, c) for sig, c in freq.items() if c >= 2]
        consensus.sort(key=lambda x: (x[1], np.median(ratio_accum.get(x[0], [0.0]))), reverse=True)

        if not consensus:
            out.write_line("No CROSS Top-K families replicated in 2+ batches under current settings.")
        else:
            out.write_line("Format: sig | batches_present | ratio_median | ratio_boot95_median")
            for sig, c in consensus[:25]:
                ratios = ratio_accum.get(sig, [])
                r_med = float(np.median(np.array(ratios, dtype=float))) if ratios else float("nan")
                # summarize CIs by taking median of lower/upper bounds across batches
                cis = ratio_ci_accum.get(sig, [])
                if cis:
                    lo_med = float(np.median([x[0] for x in cis]))
                    hi_med = float(np.median([x[1] for x in cis]))
                    ci_str = f"[{lo_med:.2f},{hi_med:.2f}]"
                else:
                    ci_str = "[na,na]"
                out.write_line(f"  {sig.as_tuple()} | {c}/{args.batches} | {r_med:.2f} | {ci_str}")

        out.write_line("")
        out.banner("Across-batch distribution effect (median entropy-bin, stable set)")
        mr = np.array(median_entbin_real, dtype=float)
        mn = np.array(median_entbin_null, dtype=float)
        out.write_line(f"Per-batch median bins (REAL): {mr.tolist()}")
        out.write_line(f"Per-batch median bins (NULL): {mn.tolist()}")
        out.write_line(f"Mean median-bin: REAL={float(np.nanmean(mr)):.2f} | NULL={float(np.nanmean(mn)):.2f} | diff={float(np.nanmean(mr - mn)):.2f}")

        out.write_line("")
        out.banner("Interpretation guide (minimal)")
        out.write_line("1) Convergence is supported when CROSS Top-K overlaps are non-trivial across batches,")
        out.write_line("   and the distribution-level effect (median entropy-bin in stable set) is stable across batches.")
        out.write_line("2) Extremely large ratios can still occur when conc_NULL is very small; smoothing + min_overall")
        out.write_line("   makes ratios readable, but replicated families across batches is the key indicator.")
        out.write_line("3) If hold-out replication inside batches is low in both REAL and NULL, it usually means")
        out.write_line("   the stable set is very small and/or family keys are too fine; increase min_overall or coarsen bins.")
        out.write_line("")
        out.write_line("=== End of v6 ===")


if __name__ == "__main__":
    main()
