#!/usr/bin/env python3
"""
v0_7_convergence_families_v9.py

Goal (v0.7 series): Calibrate "chance structure" under the same pipeline, and test convergence
(i.e., that results are not single-run artifacts) using multiple batches with different seed offsets.

What v9 adds on top of v8:
  1) Explicit multi-batch convergence metrics for REAL and NULL:
       - Top-K family overlap (Jaccard; and intersection size across all runs)
       - Rank stability (Spearman on log-enrichment for common families)
  2) Distribution-level effect size with uncertainty across batches:
       - median entropy-bin, median leakage-bin, median dom_count (REAL vs NULL)
       - paired batch-differences + bootstrap CI over batches
  3) Cross-model enrichment calibration with:
       - smoothing (pseudocounts) to avoid 0/0 or infinite ratios
       - minimum-count filters so ratios are readable and statistically meaningful
       - bootstrap CI over batches for REAL/NULL ratio on selected families

Design constraints:
  - Keep the scientific logic consistent with the v0.6.1/v0.7 style:
    (near-degenerate neighbor pairs) + (structure descriptors) + (leakage proxy) + (stable subset).
  - Use ASCII-only output (avoid Unicode arrows etc.) for maximum Windows console compatibility.
  - Write console output to a UTF-8 text file (Tee).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


# --------------------------
# I/O: Tee (stdout -> file)
# --------------------------
class Tee:
    def __init__(self, filepath: str, mode: str = "w", encoding: str = "utf-8"):
        self.filepath = filepath
        self._f = open(filepath, mode, encoding=encoding, newline="\n")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, s: str) -> int:
        # Always write to both stdout and file.
        try:
            self._stdout.write(s)
        except Exception:
            pass
        try:
            self._f.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.close()


# --------------------------
# Quantum utilities
# --------------------------
PAULI_1Q = {
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


def pauli_term_matrix(pauli: str) -> np.ndarray:
    """Return the 2^n x 2^n matrix for a Pauli string like 'YZX'."""
    mats = [PAULI_1Q[p] for p in pauli]
    return kron_n(mats)


def random_pauli_string(n: int, rng: np.random.Generator) -> str:
    alphabet = ["I", "X", "Y", "Z"]
    return "".join(rng.choice(alphabet) for _ in range(n))


def build_random_hamiltonian(n_qubits: int, num_terms: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    Build an Hermitian Pauli-sum Hamiltonian:
        H = sum_k c_k P_k
    with random Pauli strings and real coefficients.
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    terms: List[Tuple[str, float]] = []
    for _ in range(num_terms):
        p = random_pauli_string(n_qubits, rng)
        c = float(rng.uniform(-1.0, 1.0))
        terms.append((p, c))
        H = H + c * pauli_term_matrix(p)
    # Numerical hermiticity guard (should already be Hermitian)
    H = 0.5 * (H + H.conj().T)
    return H, terms


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-random unitary via QR of complex normal matrix.
    """
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph


def spectrum_matched_null(evals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    NULL model: preserve the eigenvalues but randomize the eigenbasis (Haar basis).
      H_null = U diag(evals) U^\dagger
    """
    dim = evals.shape[0]
    U = haar_random_unitary(dim, rng)
    return (U * evals) @ U.conj().T


def shannon_entropy_probs(p: np.ndarray) -> float:
    """Shannon entropy in bits; p must be nonnegative and sum to 1."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def dominant_support(psi_i: np.ndarray, psi_j: np.ndarray, threshold: float) -> List[int]:
    """
    Dominant basis support = indices where |amp|^2 > threshold
    in either eigenvector (union).
    """
    pi = np.abs(psi_i) ** 2
    pj = np.abs(psi_j) ** 2
    idx = np.where((pi > threshold) | (pj > threshold))[0]
    return idx.tolist()


def trotter_step_unitary(terms: List[Tuple[str, float]], dt: float) -> np.ndarray:
    """
    First-order Trotter step: U(dt) = prod_k exp(-i c_k P_k dt).
    Uses exp(-i a P) = cos(a) I - i sin(a) P (since P^2 = I).
    """
    n = len(terms[0][0])
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    I = np.eye(dim, dtype=complex)
    for p, c in terms:
        P = pauli_term_matrix(p)
        a = c * dt
        U_k = math.cos(a) * I - 1j * math.sin(a) * P
        U = U_k @ U
    return U


def leakage_proxy_from_terms(
    terms: List[Tuple[str, float]],
    support: List[int],
    times: List[float],
    trotter_steps: int,
) -> float:
    """
    Leakage proxy: prepare a uniform state on the dominant computational basis support,
    evolve via Trotterized dynamics, and measure probability mass outside the support.
    Returns average leakage over 'times'.

    Leakage(t) = 1 - sum_{x in support} |<x|psi(t)>|^2
    """
    if len(support) == 0:
        return 1.0
    n = len(terms[0][0])
    dim = 2 ** n

    psi0 = np.zeros((dim,), dtype=complex)
    amp = 1.0 / math.sqrt(len(support))
    for idx in support:
        psi0[idx] = amp

    leaks: List[float] = []
    support_mask = np.zeros((dim,), dtype=float)
    support_mask[support] = 1.0

    for t in times:
        m = max(1, int(trotter_steps))
        dt = float(t) / m
        U_step = trotter_step_unitary(terms, dt)
        psi = psi0.copy()
        for _ in range(m):
            psi = U_step @ psi
        p = np.abs(psi) ** 2
        inside = float(np.sum(p * support_mask))
        leaks.append(max(0.0, min(1.0, 1.0 - inside)))

    return float(np.mean(leaks))


def bin_index(x: float, step: float) -> int:
    if step <= 0:
        return 0
    return int(math.floor(max(0.0, x) / step))


def signature_key(dom_count: int, ent_i: float, ent_j: float, leak: float, ent_step: float, leak_step: float) -> Tuple[int, int, int, int]:
    """
    Primary family signature (integer bins):
      (dom_count, ent_bin_low, ent_bin_high, leak_bin)

    Sorting ent bins makes the key invariant to swapping i/j.
    """
    bi = bin_index(ent_i, ent_step)
    bj = bin_index(ent_j, ent_step)
    b_low, b_high = (bi, bj) if bi <= bj else (bj, bi)
    bl = bin_index(leak, leak_step)
    return (int(dom_count), int(b_low), int(b_high), int(bl))


def interestingness_score(dom_count: int, leak_avg: float, ent_i: float, ent_j: float,
                          w_leak: float = 0.70, w_dom: float = 0.20, w_ent: float = 0.10) -> float:
    leak_term = 1.0 - max(0.0, min(1.0, leak_avg))
    dom_term = 1.0 / (1.0 + max(0, dom_count - 1))
    ent_term = 1.0 / (1.0 + 0.5 * (ent_i + ent_j))
    s = w_leak * leak_term + w_dom * dom_term + w_ent * ent_term
    return float(max(0.0, min(1.0, s)))


# --------------------------
# Candidate extraction
# --------------------------
@dataclass(frozen=True)
class Candidate:
    sig: Tuple[int, int, int, int]
    score: float
    leak: float
    dom: int
    ent_i: float
    ent_j: float


def near_degenerate_neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    """
    Neighbor scan: sort eigenvalues and test adjacent spacings.
    Returns pairs of original indices (i,j) with |E_i - E_j| < eps.
    """
    order = np.argsort(evals)
    pairs: List[Tuple[int, int]] = []
    for k in range(len(order) - 1):
        i = int(order[k])
        j = int(order[k + 1])
        if abs(float(evals[i] - evals[j])) < eps:
            pairs.append((i, j))
    return pairs


def extract_candidates(
    H: np.ndarray,
    terms: List[Tuple[str, float]],
    eps: float,
    threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    trotter_steps: int,
) -> List[Candidate]:
    evals, evecs = np.linalg.eigh(H)
    pairs = near_degenerate_neighbor_pairs(evals, eps)

    cands: List[Candidate] = []
    for i, j in pairs:
        psi_i = evecs[:, i]
        psi_j = evecs[:, j]
        dom_idx = dominant_support(psi_i, psi_j, threshold)
        dom_count = len(dom_idx)

        pi = np.abs(psi_i) ** 2
        pj = np.abs(psi_j) ** 2
        ent_i = shannon_entropy_probs(pi)
        ent_j = shannon_entropy_probs(pj)

        leak = leakage_proxy_from_terms(terms, dom_idx, times, trotter_steps)
        score = interestingness_score(dom_count, leak, ent_i, ent_j)
        sig = signature_key(dom_count, ent_i, ent_j, leak, ent_step, leak_step)

        cands.append(Candidate(sig=sig, score=score, leak=leak, dom=dom_count, ent_i=ent_i, ent_j=ent_j))

    return cands


# --------------------------
# Enrichment and stability
# --------------------------
def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = int(scores.size)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.ceil(stable_frac * n)))
    # stable = top-k by score
    idx = np.argsort(-scores)  # descending
    mask = np.zeros((n,), dtype=bool)
    mask[idx[:k]] = True
    return mask


def enrichment_table(
    sigs: List[Tuple[int, int, int, int]],
    stable_mask: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[Dict[Tuple[int, int, int, int], float], Dict[Tuple[int, int, int, int], int], Dict[Tuple[int, int, int, int], int], int, int]:
    """
    Compute smoothed enrichment per signature:
      enr = (p_stable / p_overall)
    with:
      p_stable  = (c_stable + alpha) / (N_stable + alpha)
      p_overall = (c_overall + alpha) / (N + alpha)
    Returns:
      enr_dict, overall_counts, stable_counts, N, N_stable
    """
    N = len(sigs)
    N_stable = int(np.sum(stable_mask)) if N > 0 else 0
    overall: Dict[Tuple[int, int, int, int], int] = {}
    stable: Dict[Tuple[int, int, int, int], int] = {}

    for k, s in enumerate(sigs):
        overall[s] = overall.get(s, 0) + 1
        if stable_mask[k]:
            stable[s] = stable.get(s, 0) + 1

    enr: Dict[Tuple[int, int, int, int], float] = {}
    if N == 0 or N_stable == 0:
        return enr, overall, stable, N, N_stable

    for s, c_all in overall.items():
        c_st = stable.get(s, 0)
        p_st = (c_st + alpha) / (N_stable + alpha)
        p_all = (c_all + alpha) / (N + alpha)
        enr[s] = float(p_st / p_all)

    return enr, overall, stable, N, N_stable


def topK_families_by_enrichment(
    enr: Dict[Tuple[int, int, int, int], float],
    overall: Dict[Tuple[int, int, int, int], int],
    stable: Dict[Tuple[int, int, int, int], int],
    topK: int,
    min_overall: int,
    min_stable: int,
) -> List[Tuple[Tuple[int, int, int, int], float, int, int]]:
    """
    Select topK families by enrichment, with count filters.
    Returns list of (sig, enrichment, overall_count, stable_count).
    """
    rows = []
    for s, e in enr.items():
        o = overall.get(s, 0)
        st = stable.get(s, 0)
        if o < min_overall:
            continue
        if st < min_stable:
            continue
        rows.append((s, float(e), int(o), int(st)))

    rows.sort(key=lambda t: (-t[1], -t[3], -t[2], t[0]))
    return rows[:topK]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def spearman_rank_corr(x: Dict, y: Dict) -> Optional[float]:
    """
    Spearman correlation on overlapping keys.
    Values are mapped to ranks (average ranks for ties).
    """
    keys = list(set(x.keys()) & set(y.keys()))
    if len(keys) < 3:
        return None

    xv = np.array([x[k] for k in keys], dtype=float)
    yv = np.array([y[k] for k in keys], dtype=float)

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            # average rank for ties
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx = ranks(xv)
    ry = ranks(yv)
    # Pearson on ranks
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = float(np.linalg.norm(rxm) * np.linalg.norm(rym))
    if denom == 0.0:
        return None
    return float(np.dot(rxm, rym) / denom)


# --------------------------
# Batch runner
# --------------------------
@dataclass
class BatchResult:
    model: str
    batch_id: int
    n_candidates: int
    sigs: List[Tuple[int, int, int, int]]
    scores: np.ndarray
    ent_bins: np.ndarray
    leak_bins: np.ndarray
    dom_counts: np.ndarray
    enr: Dict[Tuple[int, int, int, int], float]
    overall: Dict[Tuple[int, int, int, int], int]
    stable: Dict[Tuple[int, int, int, int], int]
    top_rows: List[Tuple[Tuple[int, int, int, int], float, int, int]]


def run_batch(
    batch_id: int,
    model: str,
    n_qubits: int,
    num_terms: int,
    seeds: int,
    seed_offset: int,
    eps: float,
    threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    trotter_steps: int,
    stable_frac: float,
    topK: int,
    alpha: float,
    min_overall: int,
    min_stable: int,
) -> BatchResult:
    rng = np.random.default_rng(seed_offset)

    all_sigs: List[Tuple[int, int, int, int]] = []
    all_scores: List[float] = []
    all_ent_bins: List[int] = []
    all_leak_bins: List[int] = []
    all_dom: List[int] = []

    t0 = time.time()
    for s in range(seeds):
        # Seed each trial deterministically from (seed_offset + s)
        trial_rng = np.random.default_rng(seed_offset + s)

        H_real, terms = build_random_hamiltonian(n_qubits, num_terms, trial_rng)
        if model == "REAL":
            H = H_real
        elif model == "NULL_HAAR_BASIS":
            evals, _ = np.linalg.eigh(H_real)
            H = spectrum_matched_null(evals, trial_rng)
        else:
            raise ValueError(f"Unknown model: {model}")

        cands = extract_candidates(H, terms, eps, threshold, ent_step, leak_step, times, trotter_steps)
        for c in cands:
            all_sigs.append(c.sig)
            all_scores.append(c.score)
            # For distribution-level summaries, store binned entropy/leakage:
            # use mean entropy bin and leakage bin from signature.
            dom, e1, e2, bl = c.sig
            all_ent_bins.append(int(round(0.5 * (e1 + e2))))  # symmetric entropy bin
            all_leak_bins.append(int(bl))
            all_dom.append(int(dom))

    elapsed = time.time() - t0
    scores = np.array(all_scores, dtype=float)
    sigs = list(all_sigs)

    stable_mask = stable_mask_from_scores(scores, stable_frac)
    enr, overall, stable, N, N_stable = enrichment_table(sigs, stable_mask, alpha=alpha)
    top_rows = topK_families_by_enrichment(enr, overall, stable, topK, min_overall, min_stable)

    ent_bins = np.array(all_ent_bins, dtype=int) if all_ent_bins else np.zeros((0,), dtype=int)
    leak_bins = np.array(all_leak_bins, dtype=int) if all_leak_bins else np.zeros((0,), dtype=int)
    dom_counts = np.array(all_dom, dtype=int) if all_dom else np.zeros((0,), dtype=int)

    print(f"Model: {model:<14} | batch={batch_id} | seeds={seeds} | candidates={len(sigs)} | elapsed={elapsed:.1f}s")
    return BatchResult(
        model=model,
        batch_id=batch_id,
        n_candidates=len(sigs),
        sigs=sigs,
        scores=scores,
        ent_bins=ent_bins,
        leak_bins=leak_bins,
        dom_counts=dom_counts,
        enr=enr,
        overall=overall,
        stable=stable,
        top_rows=top_rows,
    )


# --------------------------
# Cross-batch aggregation
# --------------------------
def aggregate_counts(batch_results: List[BatchResult]) -> Tuple[Dict[Tuple[int,int,int,int], int], Dict[Tuple[int,int,int,int], int], int, int]:
    overall: Dict[Tuple[int,int,int,int], int] = {}
    stable: Dict[Tuple[int,int,int,int], int] = {}
    N = 0
    N_stable = 0

    for br in batch_results:
        for s, c in br.overall.items():
            overall[s] = overall.get(s, 0) + int(c)
        for s, c in br.stable.items():
            stable[s] = stable.get(s, 0) + int(c)

        N += sum(br.overall.values())
        N_stable += sum(br.stable.values())

    return overall, stable, N, N_stable


def smoothed_enrichment_from_counts(overall: Dict, stable: Dict, N: int, N_stable: int, alpha: float) -> Dict:
    enr: Dict = {}
    if N == 0 or N_stable == 0:
        return enr
    for s, c_all in overall.items():
        c_st = stable.get(s, 0)
        p_st = (c_st + alpha) / (N_stable + alpha)
        p_all = (c_all + alpha) / (N + alpha)
        enr[s] = float(p_st / p_all)
    return enr


def bootstrap_ratio_ci_over_batches(
    fam: Tuple[int,int,int,int],
    real_batches: List[BatchResult],
    null_batches: List[BatchResult],
    alpha: float,
    min_overall: int,
    min_stable: int,
    B: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI by resampling batches with replacement.
    Returns (ratio_point, lo, hi) on REAL/NULL enrichment ratio for a given family.
    """
    # Point estimate from pooled counts
    def pooled_enr(batches: List[BatchResult]) -> Optional[float]:
        o = 0
        st = 0
        N = 0
        Ns = 0
        for br in batches:
            o += br.overall.get(fam, 0)
            st += br.stable.get(fam, 0)
            N += sum(br.overall.values())
            Ns += sum(br.stable.values())
        if o < min_overall or st < min_stable:
            return None
        if N == 0 or Ns == 0:
            return None
        p_st = (st + alpha) / (Ns + alpha)
        p_all = (o + alpha) / (N + alpha)
        return float(p_st / p_all)

    enr_r = pooled_enr(real_batches)
    enr_n = pooled_enr(null_batches)
    if enr_r is None or enr_n is None:
        return (float("nan"), float("nan"), float("nan"))
    ratio_point = enr_r / enr_n if enr_n > 0 else float("inf")

    # Bootstrap
    ratios: List[float] = []
    nb = len(real_batches)
    for _ in range(B):
        idx = rng.integers(0, nb, size=nb)
        rb = [real_batches[int(i)] for i in idx]
        nb2 = len(null_batches)
        idx2 = rng.integers(0, nb2, size=nb2)
        nbatch = [null_batches[int(i)] for i in idx2]
        er = pooled_enr(rb)
        en = pooled_enr(nbatch)
        if er is None or en is None or en <= 0:
            continue
        ratios.append(er / en)

    if len(ratios) < 10:
        return (ratio_point, float("nan"), float("nan"))

    ratios.sort()
    lo = ratios[int(0.025 * (len(ratios) - 1))]
    hi = ratios[int(0.975 * (len(ratios) - 1))]
    return (ratio_point, float(lo), float(hi))


def print_top_rows(title: str, rows: List[Tuple[Tuple[int,int,int,int], float, int, int]]) -> None:
    print(title)
    print("Format: sig=(dom, ent_bin_lo, ent_bin_hi, leak_bin) | enrichment | overall | stable")
    for s, e, o, st in rows:
        print(f"  {s} | {e:7.2f}x | {o:7d} | {st:6d}")
    if not rows:
        print("  (no rows; try lowering min_overall/min_stable or increasing seeds)")
    print("")


def summarize_distribution_effects(real_batches: List[BatchResult], null_batches: List[BatchResult], B: int, rng: np.random.Generator) -> None:
    """
    Report batch-level medians and a bootstrap CI over the paired batch differences.
    """
    def batch_median(br: BatchResult):
        return (
            float(np.median(br.ent_bins)) if br.ent_bins.size else float("nan"),
            float(np.median(br.leak_bins)) if br.leak_bins.size else float("nan"),
            float(np.median(br.dom_counts)) if br.dom_counts.size else float("nan"),
        )

    pairs = []
    for r, n in zip(real_batches, null_batches):
        re = batch_median(r)
        nu = batch_median(n)
        pairs.append((re, nu))

    diffs = []
    for (re, nu) in pairs:
        diffs.append((re[0] - nu[0], re[1] - nu[1], re[2] - nu[2]))

    diffs = np.array(diffs, dtype=float) if diffs else np.zeros((0,3), dtype=float)
    if diffs.shape[0] == 0:
        print("Distribution-level effects: no data.")
        return

    point = np.nanmedian(diffs, axis=0)

    # bootstrap over batches
    boots = []
    nb = diffs.shape[0]
    for _ in range(B):
        idx = rng.integers(0, nb, size=nb)
        boots.append(np.nanmedian(diffs[idx], axis=0))
    boots = np.array(boots, dtype=float)

    def ci(col: int):
        v = np.sort(boots[:, col])
        lo = v[int(0.025 * (len(v) - 1))]
        hi = v[int(0.975 * (len(v) - 1))]
        return float(lo), float(hi)

    ci_ent = ci(0)
    ci_leak = ci(1)
    ci_dom = ci(2)

    print("=== Distribution-level effect sizes (REAL - NULL) across batches ===")
    print("Metric: median(entropy_bin_mean), median(leak_bin), median(dom_count)")
    print(f"Point (median of batch diffs): ent={point[0]:+.2f}, leak={point[1]:+.2f}, dom={point[2]:+.2f}")
    print(f"Bootstrap 95% CI (over batches): ent=[{ci_ent[0]:+.2f}, {ci_ent[1]:+.2f}]")
    print(f"                           leak=[{ci_leak[0]:+.2f}, {ci_leak[1]:+.2f}]")
    print(f"                            dom=[{ci_dom[0]:+.2f}, {ci_dom[1]:+.2f}]")
    print("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--num_terms", type=int, default=5)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--seed0", type=int, default=12345, help="base RNG seed for batch 0")
    ap.add_argument("--seed_stride", type=int, default=100000, help="seed offset between batches")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--ent_step", type=float, default=0.10)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--times", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--trotter_steps", type=int, default=8)
    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.5, help="pseudocount smoothing for enrichment")
    ap.add_argument("--min_overall", type=int, default=10)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--bootstrap_batches", type=int, default=500)
    ap.add_argument("--bootstrap_ci", type=int, default=200)
    ap.add_argument("--out", type=str, default="v0_7_convergence_families_v9_output.txt")
    args = ap.parse_args()

    times = [float(x.strip()) for x in args.times.split(",") if x.strip()]

    header = (
        "=== v0_7: Convergence + Baseline Calibration (v9) ===\n"
        f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.num_terms}\n"
        f"Batches: {args.batches} | seeds_per_batch={args.seeds_per_batch} | stable_frac={args.stable_frac:.3f} | topK={args.topK}\n"
        f"Near-degenerate neighbor eps={args.eps:.3f}\n"
        f"Dominant threshold={args.threshold:.3f} | ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}\n"
        f"Leakage proxy: times={times} | trotter_steps={args.trotter_steps}\n"
        f"Enrichment smoothing alpha={args.alpha:.2f} | min_overall={args.min_overall} | min_stable={args.min_stable}\n"
        f"Bootstrap: batch_CI={args.bootstrap_batches} | ratio_CI={args.bootstrap_ci}\n"
        f"Output: {args.out}\n"
    )

    with Tee(args.out):
        print(header)

        real_batches: List[BatchResult] = []
        null_batches: List[BatchResult] = []

        for b in range(args.batches):
            offset = int(args.seed0 + b * args.seed_stride)
            br_real = run_batch(
                batch_id=b,
                model="REAL",
                n_qubits=args.n_qubits,
                num_terms=args.num_terms,
                seeds=args.seeds_per_batch,
                seed_offset=offset,
                eps=args.eps,
                threshold=args.threshold,
                ent_step=args.ent_step,
                leak_step=args.leak_step,
                times=times,
                trotter_steps=args.trotter_steps,
                stable_frac=args.stable_frac,
                topK=args.topK,
                alpha=args.alpha,
                min_overall=args.min_overall,
                min_stable=args.min_stable,
            )
            br_null = run_batch(
                batch_id=b,
                model="NULL_HAAR_BASIS",
                n_qubits=args.n_qubits,
                num_terms=args.num_terms,
                seeds=args.seeds_per_batch,
                seed_offset=offset + 777,  # independent stream, still paired per batch
                eps=args.eps,
                threshold=args.threshold,
                ent_step=args.ent_step,
                leak_step=args.leak_step,
                times=times,
                trotter_steps=args.trotter_steps,
                stable_frac=args.stable_frac,
                topK=args.topK,
                alpha=args.alpha,
                min_overall=args.min_overall,
                min_stable=args.min_stable,
            )

            real_batches.append(br_real)
            null_batches.append(br_null)

        print("\n=== (1) Top-family stability across batches ===")
        # Top sets and Jaccard
        def top_set(br: BatchResult) -> set:
            return set([r[0] for r in br.top_rows])

        real_sets = [top_set(br) for br in real_batches]
        null_sets = [top_set(br) for br in null_batches]

        def pairwise_jaccard(sets: List[set]) -> Tuple[float, int]:
            """Mean pairwise Jaccard, skipping pairs where at least one set is empty."""
            if len(sets) < 2:
                return 0.0, 0
            js = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    if not sets[i] or not sets[j]:
                        continue
                    js.append(jaccard(sets[i], sets[j]))
            if not js:
                return 0.0, 0
            return float(np.mean(js)), len(js)

        real_j, real_pairs = pairwise_jaccard(real_sets)
        null_j, null_pairs = pairwise_jaccard(null_sets)
        real_all = set.intersection(*real_sets) if real_sets else set()
        null_all = set.intersection(*null_sets) if null_sets else set()

        print(f"REAL: Top-{args.topK} sizes per batch = {[len(s) for s in real_sets]}")
        if real_pairs>0:
            print(f"REAL: pairwise Jaccard mean (non-empty pairs) = {real_j:.3f} | pairs_used={real_pairs} | intersection(all batches) = {len(real_all)}")
        else:
            print(f"REAL: pairwise Jaccard mean = n/a (Top sets empty under filters) | intersection(all batches) = {len(real_all)}")
        print(f"NULL: Top-{args.topK} sizes per batch = {[len(s) for s in null_sets]}")
        if null_pairs>0:
            print(f"NULL: pairwise Jaccard mean (non-empty pairs) = {null_j:.3f} | pairs_used={null_pairs} | intersection(all batches) = {len(null_all)}")
        else:
            print(f"NULL: pairwise Jaccard mean = n/a (Top sets empty under filters) | intersection(all batches) = {len(null_all)}")
        print("Note: If Top-K lists are very small (due to min-count filters), overlaps become less informative.\n")

        # Rank stability (Spearman) on log-enrichment for overlap keys
        def log_enr_map(br: BatchResult) -> Dict:
            # Use full enrichment dict but avoid log(0).
            return {k: math.log(max(v, 1e-12)) for k, v in br.enr.items()}

        if args.batches >= 2:
            rs = []
            ns = []
            for i in range(args.batches - 1):
                r = spearman_rank_corr(log_enr_map(real_batches[i]), log_enr_map(real_batches[i + 1]))
                n = spearman_rank_corr(log_enr_map(null_batches[i]), log_enr_map(null_batches[i + 1]))
                if r is not None:
                    rs.append(r)
                if n is not None:
                    ns.append(n)
            if rs:
                print(f"REAL: Spearman(log enrichment) adjacent-batch mean = {float(np.mean(rs)):.3f} (n={len(rs)})")
            else:
                print("REAL: Spearman(log enrichment) not available (too few common families).")
            if ns:
                print(f"NULL: Spearman(log enrichment) adjacent-batch mean = {float(np.mean(ns)):.3f} (n={len(ns)})")
            else:
                print("NULL: Spearman(log enrichment) not available (too few common families).")
            print("")

        print("\n=== (2) Distribution-level effects (REAL vs NULL) ===")
        rng_eff = np.random.default_rng(20250101)
        summarize_distribution_effects(real_batches, null_batches, args.bootstrap_batches, rng_eff)

        print("\n=== (3) Pooled top families per model (after smoothing + min-count filter) ===")
        # Pooled counts
        real_overall, real_stable, N_r, Ns_r = aggregate_counts(real_batches)
        null_overall, null_stable, N_n, Ns_n = aggregate_counts(null_batches)

        enr_r = smoothed_enrichment_from_counts(real_overall, real_stable, N_r, Ns_r, args.alpha)
        enr_n = smoothed_enrichment_from_counts(null_overall, null_stable, N_n, Ns_n, args.alpha)

        top_real = topK_families_by_enrichment(enr_r, real_overall, real_stable, args.topK, args.min_overall, args.min_stable)
        top_null = topK_families_by_enrichment(enr_n, null_overall, null_stable, args.topK, args.min_overall, args.min_stable)

        print_top_rows("Pooled REAL Top families:", top_real)
        print_top_rows("Pooled NULL Top families:", top_null)

        print("\n=== (4) Cross-model calibration on pooled Top families ===")
        print("Interpretation: focus on families that are enriched in REAL AND present with adequate counts in NULL.")
        print("Format: sig | enr_REAL | enr_NULL | ratio | ratio_CI_95 (bootstrap over batches)\n")

        # Candidate families to compare: union of pooled top lists
        fams = sorted(set([r[0] for r in top_real] + [r[0] for r in top_null]))

        rng_ci = np.random.default_rng(20251215)
        shown = 0
        for fam in fams:
            # Filter families with sufficient support in BOTH models
            if real_overall.get(fam, 0) < args.min_overall or null_overall.get(fam, 0) < args.min_overall:
                continue
            if real_stable.get(fam, 0) < args.min_stable or null_stable.get(fam, 0) < args.min_stable:
                continue

            er = enr_r.get(fam, float("nan"))
            en = enr_n.get(fam, float("nan"))
            if not (er > 0 and en > 0):
                continue

            ratio_point, lo, hi = bootstrap_ratio_ci_over_batches(
                fam, real_batches, null_batches,
                alpha=args.alpha,
                min_overall=args.min_overall,
                min_stable=args.min_stable,
                B=args.bootstrap_ci,
                rng=rng_ci,
            )
            if math.isnan(ratio_point):
                continue

            if math.isnan(lo) or math.isnan(hi):
                ci_str = "[insufficient bootstrap support]"
            else:
                ci_str = f"[{lo:.2f}, {hi:.2f}]"

            print(f"  {fam} | {er:7.2f}x | {en:7.2f}x | {ratio_point:7.2f}x | {ci_str}")
            shown += 1

        if shown == 0:
            print("  (no cross-model families passed the support filters; increase seeds or relax min_* thresholds)")
        print("")

        print("=== Notes (scientific reading) ===")
        print("1) Within-model enrichment has a saturation ceiling near 1/stable_frac when a family appears almost only inside the stable subset.")
        print("   This is why cross-model calibration + support filters are essential.")
        print("2) Convergence is supported when (a) Top-K overlap is non-trivial across batches and (b) distribution-level effects are stable.")
        print("3) Cross-model ratios are only interpretable when the same family has non-trivial support in BOTH models; otherwise, ratios are dominated by sparsity.")
        print("\n=== End of v0_7 (v9) ===")


if __name__ == "__main__":
    main()
