#!/usr/bin/env python3
"""
v0_7_convergence_families_v7.py
--------------------------------
Convergence-focused baseline calibration for signature families (REAL vs NULL_HAAR_BASIS),
built on the v0_6.1 / v0_7 SigV2 logic.

What this version adds (without changing the core discovery logic):
  1) Multi-batch replication: run multiple independent batches with seed offsets and report Top-K overlap stability.
  2) Distribution-level effect sizes (REAL vs NULL) with bootstrap uncertainty.
  3) Cross-model enrichment that is readable: smoothing + minimum-count filters + consistent log/ratio reporting.
  4) Dual signature views: "fine" (default bins) and "coarse" (optional, for replication robustness).

This script writes all console output to a UTF-8 text file as well.

Python: 3.10+ recommended.
Dependencies: numpy (required).
"""

from __future__ import annotations

import argparse
import cmath
import math
import os
import sys
import time
import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

try:
    import numpy as np
except Exception as e:
    print("ERROR: numpy is required for this script.")
    print("Install with: pip install numpy")
    raise


# ----------------------------
# Utilities: Tee output
# ----------------------------

class Tee:
    """Write to both stdout and a file (UTF-8). Safe on Windows console encodings."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._f = open(filepath, "w", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout

    def write(self, obj):
        # ensure str
        s = obj if isinstance(obj, str) else str(obj)
        self._stdout.write(s)
        self._f.write(s)

    def flush(self):
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self._f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ----------------------------
# Quantum / linear algebra helpers (3-qubit default, but n is configurable)
# ----------------------------

PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def random_pauli_string(n: int, rng: random.Random) -> str:
    # Include I to allow locality / sparsity patterns like earlier prototypes
    alphabet = ["I", "X", "Y", "Z"]
    return "".join(rng.choice(alphabet) for _ in range(n))

def build_random_hamiltonian(n_qubits: int, num_terms: int, seed: int) -> Tuple[np.ndarray, List[Tuple[complex, str]]]:
    """
    Build a dense Hamiltonian H = sum_k c_k P_k, where P_k is a Pauli string.
    Coefficients are real-valued in [-1,1] (as in earlier prototypes).
    """
    rng = random.Random(seed)
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=np.complex128)
    terms: List[Tuple[complex, str]] = []

    for _ in range(num_terms):
        pstr = random_pauli_string(n_qubits, rng)
        ck = rng.uniform(-1.0, 1.0)
        ops = [PAULI[ch] for ch in pstr]
        Pk = kron_n(ops)
        H = H + ck * Pk
        terms.append((ck, pstr))

    # Hermitian by construction (Paulis are Hermitian, real coefficients)
    return H, terms

def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Hermitian eigendecomposition
    evals, evecs = np.linalg.eigh(H)
    # Ensure deterministic ordering (ascending evals)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]

def find_near_degenerate_neighbors(evals: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    """
    Neighbor scan: only compare adjacent eigenvalues (fast, and consistent with earlier v0_6+ work).
    Returns list of (i, i+1, gap).
    """
    pairs = []
    for i in range(len(evals) - 1):
        gap = float(abs(evals[i+1] - evals[i]))
        if gap < eps:
            pairs.append((i, i+1, gap))
    return pairs

def shannon_entropy_from_probs(p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def dominant_basis_count(vec: np.ndarray, prob_threshold: float) -> int:
    probs = np.abs(vec) ** 2
    return int(np.sum(probs >= prob_threshold))

def leakage_proxy_from_projected_states(
    H: np.ndarray,
    evecs: np.ndarray,
    i: int,
    j: int,
    times: List[float],
    trotter_steps: int,
    rng: random.Random,
) -> float:
    """
    Lightweight stability proxy: prepare a random superposition inside span{|psi_i>, |psi_j>}
    and measure how much weight leaks outside after time evolution.
    Uses exact expm via eigendecomposition (since d=8 is small) to avoid algorithmic drift.
    """
    # random normalized superposition in the 2D subspace
    a = rng.uniform(-1.0, 1.0) + 1j * rng.uniform(-1.0, 1.0)
    b = rng.uniform(-1.0, 1.0) + 1j * rng.uniform(-1.0, 1.0)
    norm = math.sqrt((a.real*a.real + a.imag*a.imag) + (b.real*b.real + b.imag*b.imag)) + 1e-12
    a /= norm
    b /= norm

    psi_i = evecs[:, i]
    psi_j = evecs[:, j]
    psi0 = a * psi_i + b * psi_j

    # projector onto the candidate subspace
    # Π = |psi_i><psi_i| + |psi_j><psi_j|
    Pi = np.outer(psi_i, np.conjugate(psi_i)) + np.outer(psi_j, np.conjugate(psi_j))

    # Exact unitary via eigendecomposition of H: U(t) = V diag(exp(-i E t)) V†
    evals, V = np.linalg.eigh(H)
    Vh = np.conjugate(V.T)

    leak_values = []
    for t in times:
        phases = np.exp(-1j * evals * t)
        U = V @ (phases[:, None] * Vh)
        psit = U @ psi0
        inside = np.vdot(psit, Pi @ psit).real  # <psi|Π|psi> in [0,1]
        inside = max(0.0, min(1.0, inside))
        leak = 1.0 - inside
        leak_values.append(leak)

    # average leakage across times
    return float(sum(leak_values) / max(1, len(leak_values)))


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar random unitary via QR of complex Gaussian matrix.
    """
    Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(Z)
    # Fix phases
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * np.conjugate(ph)
    return Q

def make_null_haar_basis_evecs(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = haar_random_unitary(d, rng)
    return U  # columns are orthonormal eigenvectors


# ----------------------------
# Signature construction
# ----------------------------

@dataclass(frozen=True)
class SigKey:
    dom: int
    ent_i: int
    ent_j: int
    leak: int
    level: str  # "fine" or "coarse"

def bin_value(x: float, step: float, cap: Optional[int] = None) -> int:
    b = int(math.floor(x / step + 1e-12))
    if cap is not None:
        b = max(0, min(int(cap), b))
    return b

def make_signature(
    dom: int,
    ent1: float,
    ent2: float,
    leak: float,
    ent_step: float,
    leak_step: float,
    ent_cap: Optional[int],
    leak_cap: Optional[int],
    level: str,
) -> SigKey:
    e1 = bin_value(ent1, ent_step, ent_cap)
    e2 = bin_value(ent2, ent_step, ent_cap)
    # symmetry: signature should not depend on eigenpair ordering
    if e2 < e1:
        e1, e2 = e2, e1
    l = bin_value(leak, leak_step, leak_cap)
    return SigKey(dom=dom, ent_i=e1, ent_j=e2, leak=l, level=level)

def candidate_score(leak: float, dom: int) -> float:
    """
    Same core scoring logic as earlier iterations: prefer low leakage and low dom_count.
    Keep it monotone and bounded in [0,1] for stable_frac selection.
    """
    leak = max(0.0, min(1.0, leak))
    dom = max(1, int(dom))
    penalty = 1.0 / math.sqrt(dom)
    return max(0.0, 1.0 - leak) * penalty


# ----------------------------
# Enrichment / bootstrap
# ----------------------------

def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = len(scores)
    k = max(1, int(round(stable_frac * n)))
    # top k scores
    idx = np.argpartition(-scores, kth=min(k-1, n-1))[:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def counts_by_signature(keys: List[SigKey], mask: Optional[np.ndarray] = None) -> Dict[SigKey, int]:
    out: Dict[SigKey, int] = {}
    if mask is None:
        for k in keys:
            out[k] = out.get(k, 0) + 1
    else:
        for k, m in zip(keys, mask):
            if m:
                out[k] = out.get(k, 0) + 1
    return out

def log_enrichment(
    overall_count: int,
    stable_count: int,
    overall_total: int,
    stable_total: int,
    alpha: float,
) -> float:
    """
    log enrichment = log p(sig|stable) - log p(sig|overall), with Dirichlet smoothing.
    """
    # Smoothed proportions
    p_stable = (stable_count + alpha) / (stable_total + alpha)
    p_overall = (overall_count + alpha) / (overall_total + alpha)
    return float(math.log(p_stable) - math.log(p_overall))

def enrichment_tables(
    keys: List[SigKey],
    scores: np.ndarray,
    stable_frac: float,
    alpha: float,
) -> Tuple[Dict[SigKey, int], Dict[SigKey, int], Dict[SigKey, float], np.ndarray]:
    mask = stable_mask_from_scores(scores, stable_frac)
    overall = counts_by_signature(keys, None)
    stable = counts_by_signature(keys, mask)
    overall_total = len(keys)
    stable_total = int(mask.sum())

    logenr: Dict[SigKey, float] = {}
    for sig, oc in overall.items():
        sc = stable.get(sig, 0)
        logenr[sig] = log_enrichment(oc, sc, overall_total, stable_total, alpha)
    return overall, stable, logenr, mask

def select_top_cross_families(
    logenr_real: Dict[SigKey, float],
    overall_real: Dict[SigKey, int],
    stable_real: Dict[SigKey, int],
    logenr_null: Dict[SigKey, float],
    overall_null: Dict[SigKey, int],
    stable_null: Dict[SigKey, int],
    min_overall_both: int,
    min_stable_any: int,
    topK: int,
) -> List[Tuple[SigKey, float, Dict[str, int]]]:
    """
    Rank signatures by cross log-ratio = logenr_real - logenr_null,
    but only for signatures with sufficient support in BOTH models (overall count),
    and at least some presence in the stable sets (to avoid pure noise keys).
    """
    heap: List[Tuple[float, int, SigKey, Dict[str, int]]] = []
    tie = 0
    for sig, lr in logenr_real.items():
        if sig.level != "fine":
            continue
        if overall_real.get(sig, 0) < min_overall_both:
            continue
        if overall_null.get(sig, 0) < min_overall_both:
            continue
        if max(stable_real.get(sig, 0), stable_null.get(sig, 0)) < min_stable_any:
            continue
        ln = logenr_null.get(sig, 0.0)  # if missing, treated as 0 (but min_overall_both prevents full miss)
        cross = float(lr - ln)
        meta = {
            "overall_real": overall_real.get(sig, 0),
            "stable_real": stable_real.get(sig, 0),
            "overall_null": overall_null.get(sig, 0),
            "stable_null": stable_null.get(sig, 0),
        }
        # keep topK largest cross
        if len(heap) < topK:
            heapq.heappush(heap, (cross, tie, sig, meta))
            tie += 1
        else:
            if cross > heap[0][0]:
                heapq.heapreplace(heap, (cross, tie, sig, meta))
                tie += 1

    heap.sort(key=lambda x: x[0], reverse=True)
    return [(sig, cross, meta) for (cross, _, sig, meta) in heap]

def bootstrap_cross_logratio_ci(
    keys_real: List[SigKey],
    scores_real: np.ndarray,
    keys_null: List[SigKey],
    scores_null: np.ndarray,
    stable_frac: float,
    alpha: float,
    target_sig: SigKey,
    iters: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Bootstrap CI for cross log-ratio for one signature, resampling candidates in each model independently.
    Returns 2.5% and 97.5% quantiles.
    """
    nR = len(keys_real)
    nN = len(keys_null)
    if nR < 10 or nN < 10:
        return (float("nan"), float("nan"))

    # precompute arrays for fast indexing
    keys_real_arr = np.array(keys_real, dtype=object)
    keys_null_arr = np.array(keys_null, dtype=object)

    samples = []
    for _ in range(iters):
        idxR = rng.integers(0, nR, size=nR)
        idxN = rng.integers(0, nN, size=nN)

        kR = list(keys_real_arr[idxR])
        sR = scores_real[idxR]
        kN = list(keys_null_arr[idxN])
        sN = scores_null[idxN]

        oR, stR, logR, _ = enrichment_tables(kR, sR, stable_frac, alpha)
        oN, stN, logN, _ = enrichment_tables(kN, sN, stable_frac, alpha)

        # Only meaningful if present in both overall after resample; otherwise skip iteration
        if target_sig not in oR or target_sig not in oN:
            continue
        cross = float(logR.get(target_sig, 0.0) - logN.get(target_sig, 0.0))
        samples.append(cross)

    if len(samples) < max(30, iters // 4):
        return (float("nan"), float("nan"))

    samples = np.array(samples, dtype=float)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return lo, hi


def bootstrap_median_diff_ci(x: np.ndarray, y: np.ndarray, iters: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Bootstrap CI for median(y) - median(x). Returns (diff, lo, hi).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 10 or ny < 10:
        d = float(np.median(y) - np.median(x))
        return d, float("nan"), float("nan")
    diffs = []
    for _ in range(iters):
        bx = x[rng.integers(0, nx, size=nx)]
        by = y[rng.integers(0, ny, size=ny)]
        diffs.append(float(np.median(by) - np.median(bx)))
    diffs = np.array(diffs, dtype=float)
    return float(np.median(y) - np.median(x)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ----------------------------
# Candidate generation
# ----------------------------

@dataclass
class Candidate:
    sig_fine: SigKey
    sig_coarse: SigKey
    score: float
    leakage: float
    dom: int
    ent_pair_mean: float

def extract_candidates_for_seed(
    seed: int,
    n_qubits: int,
    num_terms: int,
    eps: float,
    dom_threshold: float,
    ent_step: float,
    leak_step: float,
    ent_cap: Optional[int],
    leak_cap: Optional[int],
    ent_step_coarse: float,
    leak_step_coarse: float,
    ent_cap_coarse: Optional[int],
    leak_cap_coarse: Optional[int],
    times: List[float],
    trotter_steps: int,
    rng_local: random.Random,
    model: str,
) -> List[Candidate]:
    """
    Produce candidates for one seed under either:
      - model="REAL": real Pauli-sum Hamiltonian
      - model="NULL_HAAR_BASIS": spectrum-matched Haar random basis (destroys computational-basis structure)
    """
    H, _terms = build_random_hamiltonian(n_qubits, num_terms, seed)
    evals, evecs_real = diagonalize(H)
    pairs = find_near_degenerate_neighbors(evals, eps)
    if not pairs:
        return []

    d = 2 ** n_qubits

    if model == "REAL":
        evecs = evecs_real
    elif model == "NULL_HAAR_BASIS":
        # Keep the spectrum, randomize eigenbasis.
        # We don't need to explicitly rebuild H; we only need eigenvectors for structure metrics,
        # and use H itself for time evolution (this intentionally tests whether structure in basis matters).
        # If you want a "fully null" dynamics too, replace H by V diag(evals) V† with the Haar basis V.
        U = make_null_haar_basis_evecs(d, seed=seed + 10_000_003)
        evecs = U  # columns are eigenvectors
    else:
        raise ValueError(f"Unknown model: {model}")

    out: List[Candidate] = []
    for (i, j, _gap) in pairs:
        vi = evecs[:, i]
        vj = evecs[:, j]

        # Dominant basis count computed over BOTH eigenvectors (conservative)
        dom_i = dominant_basis_count(vi, dom_threshold)
        dom_j = dominant_basis_count(vj, dom_threshold)
        dom = max(dom_i, dom_j)

        # Entropy over computational-basis probabilities
        ent_i = shannon_entropy_from_probs(np.abs(vi) ** 2)
        ent_j = shannon_entropy_from_probs(np.abs(vj) ** 2)
        ent_mean = 0.5 * (ent_i + ent_j)

        # Leakage proxy based on dynamics under the REAL Hamiltonian H (same for both models).
        leak = leakage_proxy_from_projected_states(
            H=H, evecs=evecs_real, i=i, j=j, times=times, trotter_steps=trotter_steps, rng=rng_local
        )
        sc = candidate_score(leak, dom)

        sig_f = make_signature(dom, ent_i, ent_j, leak, ent_step, leak_step, ent_cap, leak_cap, "fine")
        sig_c = make_signature(dom, ent_i, ent_j, leak, ent_step_coarse, leak_step_coarse, ent_cap_coarse, leak_cap_coarse, "coarse")

        out.append(Candidate(sig_fine=sig_f, sig_coarse=sig_c, score=sc, leakage=leak, dom=dom, ent_pair_mean=ent_mean))

    return out


def run_batch(
    batch_id: int,
    base_seed: int,
    n_seeds: int,
    seed_offset: int,
    args,
) -> Dict[str, Dict[str, object]]:
    """
    Returns a dict with per-model results for this batch.
    """
    t0 = time.time()
    rng_local = random.Random(base_seed + 99991 + 31 * batch_id)

    models = ["REAL", "NULL_HAAR_BASIS"]
    results: Dict[str, Dict[str, object]] = {}

    for model in models:
        cands: List[Candidate] = []
        for s in range(n_seeds):
            seed = seed_offset + s
            cands.extend(
                extract_candidates_for_seed(
                    seed=seed,
                    n_qubits=args.n_qubits,
                    num_terms=args.num_terms,
                    eps=args.eps,
                    dom_threshold=args.dom_threshold,
                    ent_step=args.ent_step,
                    leak_step=args.leak_step,
                    ent_cap=args.ent_cap,
                    leak_cap=args.leak_cap,
                    ent_step_coarse=args.ent_step_coarse,
                    leak_step_coarse=args.leak_step_coarse,
                    ent_cap_coarse=args.ent_cap_coarse,
                    leak_cap_coarse=args.leak_cap_coarse,
                    times=args.times,
                    trotter_steps=args.trotter_steps,
                    rng_local=rng_local,
                    model=model,
                )
            )

        # pack arrays
        keys_fine = [c.sig_fine for c in cands]
        scores = np.array([c.score for c in cands], dtype=float)
        leaks = np.array([c.leakage for c in cands], dtype=float)
        doms = np.array([c.dom for c in cands], dtype=int)
        entm = np.array([c.ent_pair_mean for c in cands], dtype=float)

        results[model] = {
            "cands": cands,
            "keys_fine": keys_fine,
            "scores": scores,
            "leaks": leaks,
            "doms": doms,
            "entm": entm,
            "elapsed": time.time() - t0,
        }

    return results


# ----------------------------
# Reporting
# ----------------------------

def fmt_sig(sig: SigKey) -> str:
    return f"({sig.dom}, {sig.ent_i}, {sig.ent_j}, {sig.leak})"

def print_model_summary(model: str, n_cands: int, scores: np.ndarray, leaks: np.ndarray, doms: np.ndarray) -> None:
    if n_cands == 0:
        print(f"Model: {model} | candidates=0")
        return
    print(f"Model: {model} | candidates={n_cands}")
    print(f"  score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f}")
    print(f"  leakage(mean/median/min)={leaks.mean():.3f}/{np.median(leaks):.3f}/{leaks.min():.3f}")
    print(f"  dom_count(median)={int(np.median(doms))}")

def top_families_within_model(
    overall: Dict[SigKey, int],
    stable: Dict[SigKey, int],
    logenr: Dict[SigKey, float],
    stable_frac: float,
    topK: int,
) -> List[Tuple[SigKey, float, int, int]]:
    """
    Return topK families by within-model log enrichment (descending).
    """
    items = []
    for sig, lr in logenr.items():
        if sig.level != "fine":
            continue
        oc = overall.get(sig, 0)
        sc = stable.get(sig, 0)
        items.append((sig, lr, oc, sc))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:topK]

def overlap_metrics(sets: List[set]) -> Tuple[float, float, int]:
    """
    Pairwise overlap summary over list of sets:
      - mean intersection size
      - mean Jaccard
      - size of intersection across ALL
    """
    if len(sets) < 2:
        return 0.0, 0.0, len(sets[0]) if sets else 0

    inter_all = set.intersection(*sets) if sets else set()

    inter_sizes = []
    jaccards = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            union = len(a | b) if (a | b) else 1
            inter_sizes.append(inter)
            jaccards.append(inter / union)
    return float(np.mean(inter_sizes)), float(np.mean(jaccards)), len(inter_all)


def main():
    parser = argparse.ArgumentParser(description="v0_7 convergence study for signature families (REAL vs NULL).")
    parser.add_argument("--n_qubits", type=int, default=3)
    parser.add_argument("--num_terms", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5000)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed_stride", type=int, default=100000, help="Seed offset between batches.")
    parser.add_argument("--eps", type=float, default=0.05, help="Near-degenerate neighbor threshold.")
    parser.add_argument("--dom_threshold", type=float, default=0.25, help="Probability threshold for dominant basis support.")
    parser.add_argument("--ent_step", type=float, default=0.1)
    parser.add_argument("--leak_step", type=float, default=0.05)
    parser.add_argument("--ent_cap", type=int, default=None)
    parser.add_argument("--leak_cap", type=int, default=None)

    # Coarse signature view for replicability (optional)
    parser.add_argument("--ent_step_coarse", type=float, default=0.25)
    parser.add_argument("--leak_step_coarse", type=float, default=0.10)
    parser.add_argument("--ent_cap_coarse", type=int, default=None)
    parser.add_argument("--leak_cap_coarse", type=int, default=None)

    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--trotter_steps", type=int, default=8)  # kept for interface; exact evolution used internally

    parser.add_argument("--stable_frac", type=float, default=0.01)
    parser.add_argument("--topK", type=int, default=25)

    # Enrichment stabilization
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing pseudocount (Dirichlet).")
    parser.add_argument("--min_overall_both", type=int, default=10, help="Min overall count in BOTH models to score cross-model.")
    parser.add_argument("--min_stable_any", type=int, default=2, help="Min stable count in at least one model to score cross-model.")

    # Bootstrap
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap_seed", type=int, default=123)

    # Output
    parser.add_argument("--out", type=str, default="v0_7_convergence_families_v7_output.txt")
    args = parser.parse_args()

    tee = Tee(args.out)
    sys.stdout = tee

    print("=== v0_7 Convergence: Signature Families (v7) ===")
    print(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.num_terms} | seeds/batch={args.seeds} | batches={args.batches}")
    print(f"Near-degenerate neighbor eps={args.eps:.3f}")
    print(f"Dominant threshold={args.dom_threshold:.3f} | fine bins: ent_step={args.ent_step:.3f}, leak_step={args.leak_step:.3f}")
    print(f"Coarse bins (replication view): ent_step={args.ent_step_coarse:.3f}, leak_step={args.leak_step_coarse:.3f}")
    print(f"Evidence: stable_frac={args.stable_frac:.3f} | topK={args.topK} | smoothing alpha={args.alpha:.2f}")
    print(f"Cross-model filters: min_overall_both={args.min_overall_both} | min_stable_any={args.min_stable_any}")
    print(f"Bootstrap: iters={args.bootstrap} | seed={args.bootstrap_seed}")
    print(f"Output file: {os.path.abspath(args.out)}")
    print()

    rng_ci = np.random.default_rng(args.bootstrap_seed)

    # Store per-batch top sets for convergence metrics
    cross_top_sets: List[set] = []
    real_top_sets: List[set] = []
    null_top_sets: List[set] = []

    # Store distribution-level effect sizes per batch
    dist_effects = []

    for b in range(args.batches):
        seed_offset = args.base_seed + b * args.seed_stride
        print("----------------------------------------------")
        print(f"Batch {b+1}/{args.batches} | seed_offset={seed_offset}")
        t0 = time.time()
        batch_res = run_batch(batch_id=b, base_seed=args.base_seed, n_seeds=args.seeds, seed_offset=seed_offset, args=args)
        dt = time.time() - t0

        # Summaries
        for model in ["REAL", "NULL_HAAR_BASIS"]:
            scores = batch_res[model]["scores"]
            leaks = batch_res[model]["leaks"]
            doms = batch_res[model]["doms"]
            print_model_summary(model, len(scores), scores, leaks, doms)
        print(f"Batch elapsed: {dt:.1f}s")
        print()

        # Enrichment tables
        keys_real = batch_res["REAL"]["keys_fine"]
        scores_real = batch_res["REAL"]["scores"]
        keys_null = batch_res["NULL_HAAR_BASIS"]["keys_fine"]
        scores_null = batch_res["NULL_HAAR_BASIS"]["scores"]

        if len(keys_real) < 50 or len(keys_null) < 50:
            print("WARNING: too few candidates in one of the models; skipping detailed stats for this batch.")
            continue

        oR, stR, logR, maskR = enrichment_tables(keys_real, scores_real, args.stable_frac, args.alpha)
        oN, stN, logN, maskN = enrichment_tables(keys_null, scores_null, args.stable_frac, args.alpha)

        # Top within-model families (fine)
        top_real = top_families_within_model(oR, stR, logR, args.stable_frac, topK=args.topK)
        top_null = top_families_within_model(oN, stN, logN, args.stable_frac, topK=args.topK)

        real_set = set([t[0] for t in top_real])
        null_set = set([t[0] for t in top_null])
        real_top_sets.append(real_set)
        null_top_sets.append(null_set)

        print("Top within-model families (REAL, by log-enrichment):")
        print("  sig=(dom, ent_i, ent_j, leak) | overall | stable | log_enr | enr")
        for sig, lr, oc, sc in top_real[:10]:
            enr = math.exp(lr)
            print(f"  {fmt_sig(sig)} | {oc:5d} | {sc:5d} | {lr:7.3f} | {enr:7.2f}x")
        print("  ...")
        print()

        print("Top within-model families (NULL_HAAR_BASIS, by log-enrichment):")
        print("  sig=(dom, ent_i, ent_j, leak) | overall | stable | log_enr | enr")
        for sig, lr, oc, sc in top_null[:10]:
            enr = math.exp(lr)
            print(f"  {fmt_sig(sig)} | {oc:5d} | {sc:5d} | {lr:7.3f} | {enr:7.2f}x")
        print("  ...")
        print()

        # Cross-model top families with readable filters
        cross_top = select_top_cross_families(
            logenr_real=logR, overall_real=oR, stable_real=stR,
            logenr_null=logN, overall_null=oN, stable_null=stN,
            min_overall_both=args.min_overall_both,
            min_stable_any=args.min_stable_any,
            topK=args.topK,
        )

        cross_set = set([t[0] for t in cross_top])
        cross_top_sets.append(cross_set)

        print("Cross-model calibration (REAL vs NULL) on supported families:")
        print("  sig=(dom, ent_i, ent_j, leak) | log_ratio | ratio | (overallR,stableR) vs (overallN,stableN) | CI95_log_ratio")
        for sig, cross_lr, meta in cross_top[:min(10, len(cross_top))]:
            lo, hi = bootstrap_cross_logratio_ci(
                keys_real, scores_real, keys_null, scores_null,
                stable_frac=args.stable_frac, alpha=args.alpha,
                target_sig=sig, iters=args.bootstrap, rng=rng_ci
            )
            ratio = math.exp(cross_lr)
            # CI in ratio space if finite
            ratio_lo = math.exp(lo) if (not math.isnan(lo)) else float("nan")
            ratio_hi = math.exp(hi) if (not math.isnan(hi)) else float("nan")
            print(
                f"  {fmt_sig(sig)} | {cross_lr:7.3f} | {ratio:7.2f}x | "
                f"({meta['overall_real']},{meta['stable_real']}) vs ({meta['overall_null']},{meta['stable_null']}) | "
                f"[{lo:7.3f},{hi:7.3f}] -> [{ratio_lo:7.2f}x,{ratio_hi:7.2f}x]"
            )
        if len(cross_top) == 0:
            print("  (No cross-model families passed the minimum-count filters. Consider lowering --min_overall_both or increasing --seeds.)")
        print()

        # Distribution-level effects (stable set only)
        entR = batch_res["REAL"]["entm"][maskR]
        entN = batch_res["NULL_HAAR_BASIS"]["entm"][maskN]
        leakR = batch_res["REAL"]["leaks"][maskR]
        leakN = batch_res["NULL_HAAR_BASIS"]["leaks"][maskN]
        domR = batch_res["REAL"]["doms"][maskR].astype(float)
        domN = batch_res["NULL_HAAR_BASIS"]["doms"][maskN].astype(float)

        d_ent, lo_ent, hi_ent = bootstrap_median_diff_ci(entR, entN, args.bootstrap, rng_ci)
        d_leak, lo_leak, hi_leak = bootstrap_median_diff_ci(leakR, leakN, args.bootstrap, rng_ci)
        d_dom, lo_dom, hi_dom = bootstrap_median_diff_ci(domR, domN, args.bootstrap, rng_ci)

        dist_effects.append({
            "batch": b+1,
            "median_ent_REAL": float(np.median(entR)),
            "median_ent_NULL": float(np.median(entN)),
            "diff_ent_NULL_minus_REAL": d_ent,
            "diff_ent_CI": (lo_ent, hi_ent),
            "median_leak_REAL": float(np.median(leakR)),
            "median_leak_NULL": float(np.median(leakN)),
            "diff_leak_NULL_minus_REAL": d_leak,
            "diff_leak_CI": (lo_leak, hi_leak),
            "median_dom_REAL": float(np.median(domR)),
            "median_dom_NULL": float(np.median(domN)),
            "diff_dom_NULL_minus_REAL": d_dom,
            "diff_dom_CI": (lo_dom, hi_dom),
            "cross_top_count": len(cross_top),
        })

        print("Distribution-level effects (stable set only; median with bootstrap CI):")
        print(f"  entropy median: REAL={np.median(entR):.3f}, NULL={np.median(entN):.3f}, NULL-REAL={d_ent:.3f}  CI95=[{lo_ent:.3f},{hi_ent:.3f}]")
        print(f"  leakage  median: REAL={np.median(leakR):.3f}, NULL={np.median(leakN):.3f}, NULL-REAL={d_leak:.3f}  CI95=[{lo_leak:.3f},{hi_leak:.3f}]")
        print(f"  dom_cnt  median: REAL={np.median(domR):.3f}, NULL={np.median(domN):.3f}, NULL-REAL={d_dom:.3f}  CI95=[{lo_dom:.3f},{hi_dom:.3f}]")
        print()

    # ----------------------------
    # Convergence summary across batches
    # ----------------------------
    print("==============================================")
    print("Convergence summary across batches")
    print()

    if cross_top_sets:
        mean_inter, mean_jacc, inter_all = overlap_metrics(cross_top_sets)
        print("Cross-model Top-K overlap stability (supported families):")
        print(f"  mean pairwise intersection size: {mean_inter:.2f}")
        print(f"  mean pairwise Jaccard:          {mean_jacc:.3f}")
        print(f"  intersection across ALL runs:   {inter_all}")
        print()
    else:
        print("No cross-model Top-K sets computed (insufficient candidates or strict min-count filters).")
        print()

    mean_inter_r, mean_jacc_r, inter_all_r = overlap_metrics(real_top_sets) if real_top_sets else (0.0, 0.0, 0)
    mean_inter_n, mean_jacc_n, inter_all_n = overlap_metrics(null_top_sets) if null_top_sets else (0.0, 0.0, 0)
    print("Within-model Top-K overlap stability:")
    print(f"  REAL: mean intersection={mean_inter_r:.2f}, mean Jaccard={mean_jacc_r:.3f}, intersection_all={inter_all_r}")
    print(f"  NULL: mean intersection={mean_inter_n:.2f}, mean Jaccard={mean_jacc_n:.3f}, intersection_all={inter_all_n}")
    print()

    if dist_effects:
        # summarize effect stability across batches
        ent_diffs = np.array([d["diff_ent_NULL_minus_REAL"] for d in dist_effects], dtype=float)
        leak_diffs = np.array([d["diff_leak_NULL_minus_REAL"] for d in dist_effects], dtype=float)
        dom_diffs = np.array([d["diff_dom_NULL_minus_REAL"] for d in dist_effects], dtype=float)

        print("Distribution-level effect stability across batches (stable set):")
        print(f"  entropy diff (NULL-REAL): mean={ent_diffs.mean():.3f}, std={ent_diffs.std(ddof=1) if len(ent_diffs)>1 else 0.0:.3f}")
        print(f"  leakage  diff (NULL-REAL): mean={leak_diffs.mean():.3f}, std={leak_diffs.std(ddof=1) if len(leak_diffs)>1 else 0.0:.3f}")
        print(f"  dom_cnt  diff (NULL-REAL): mean={dom_diffs.mean():.3f}, std={dom_diffs.std(ddof=1) if len(dom_diffs)>1 else 0.0:.3f}")
        print()

        print("Interpretation (method-level, no hype):")
        print("  - If Top-K overlaps are high AND distribution-level effects are stable across batches,")
        print("    then the result is unlikely to be a single-run artifact.")
        print("  - If overlaps are low but distribution effects are stable,")
        print("    then the binning is still too fine for family labels to replicate;")
        print("    in that case, increase --min_overall_both, coarsen ent/leak steps, or report effects at distribution level.")
        print("  - If both overlaps and distribution effects are unstable,")
        print("    then increase seeds, tighten the stability score definition, or adjust eps/dom_threshold for a higher-signal candidate set.")
        print()

    print("=== End of v0_7 convergence v7 ===")
    tee.flush()
    tee.close()


if __name__ == "__main__":
    main()
