#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_batches.py
Batch-convergence harness for "matched, fair" baseline calibration.

What it does (3× batches, e.g. 3 × 5000 seeds):
  (1) Top-family stability: overlap(Top-25) between runs (pairwise + all-three)
  (2) Distribution effect-size: median(entropy_bin) REAL vs NULL + bootstrap CI
  (3) Cross-model enrichment with smoothing + minimum-count filter (readable ratios)

Design constraints:
  - Keeps the core discovery logic: near-degenerate neighbor pairs -> structure metrics -> leakage proxy -> score -> stable-set -> enrichment.
  - Uses only NumPy (no SciPy), and prints + writes the same output to a .txt file (UTF-8).
  - 3 qubits (d=8) for speed and interpretability.

Notes:
  - "NULL_HAAR_BASIS" is spectrum-matched: we keep eigenvalues and the same Hamiltonian,
    but replace eigenvectors used for signature extraction with a Haar-random orthonormal basis.
  - Cross-model ratios are computed only for families that pass minimum-count filters.

Author: Tom Stevns project support
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Utilities: Tee output to file
# ----------------------------

class Tee:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._f = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s: str):
        # Always write to both outputs
        self._stdout.write(s)
        self._f.write(s)

    def flush(self):
        self._stdout.flush()
        self._f.flush()

    def close(self):
        try:
            self._f.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


# ----------------------------
# Quantum / Linear algebra core
# ----------------------------

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


@dataclass(frozen=True)
class PauliTerm:
    label: str          # e.g. "XZI"
    coeff: float        # real coefficient

    def matrix(self) -> np.ndarray:
        return kron_n([PAULI_1Q[p] for p in self.label])


@dataclass
class Candidate:
    sig: Tuple[int, int, int, int]   # (dom, ent_bin_i, ent_bin_j, leak_bin)
    score: float
    ent_bin_med: float               # for distribution-level metrics
    leakage_avg: float


def random_pauli_label(n: int, rng: np.random.Generator) -> str:
    # Avoid the all-identity label to keep non-trivial Hamiltonians
    while True:
        label = "".join(rng.choice(["I", "X", "Y", "Z"], size=n))
        if set(label) != {"I"}:
            return label


def build_random_hamiltonian(n_qubits: int, num_terms: int, seed: int) -> List[PauliTerm]:
    rng = np.random.default_rng(seed)
    labels = [random_pauli_label(n_qubits, rng) for _ in range(num_terms)]
    coeffs = rng.uniform(-1.0, 1.0, size=num_terms).astype(float)
    terms = [PauliTerm(label=labels[i], coeff=float(coeffs[i])) for i in range(num_terms)]
    return terms


def hamiltonian_matrix(terms: List[PauliTerm]) -> np.ndarray:
    d = PAULI_1Q["I"].shape[0] ** len(terms[0].label)
    H = np.zeros((d, d), dtype=complex)
    for t in terms:
        H += t.coeff * t.matrix()
    # Hermitian safeguard
    H = 0.5 * (H + H.conj().T)
    return H


def diag_hermitian(H: np.ndarray):
    # Returns eigenvalues (ascending) and eigenvectors (columns)
    evals, evecs = np.linalg.eigh(H)
    return evals.real, evecs


def near_degenerate_neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(len(evals) - 1):
        if abs(evals[i + 1] - evals[i]) < eps:
            pairs.append((i, i + 1))
    return pairs


def probs_from_state(vec: np.ndarray) -> np.ndarray:
    p = np.abs(vec) ** 2
    s = p.sum()
    return p / s if s > 0 else p


def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * (np.log(p) / np.log(base))).sum())


def dominant_basis_count(p: np.ndarray, threshold: float) -> int:
    return int((p >= threshold).sum())


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    # Haar via QR on complex Gaussian
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    # Fix phases
    diag = np.diag(r)
    ph = diag / np.abs(diag)
    q = q * ph.conj()
    return q


def expm_pauli_term(term: PauliTerm, theta: float) -> np.ndarray:
    # For a Pauli string P with P^2 = I: exp(-i * theta * P) = cos(theta) I - i sin(theta) P
    P = term.matrix()
    d = P.shape[0]
    I = np.eye(d, dtype=complex)
    return (math.cos(theta) * I) - 1j * (math.sin(theta) * P)


def trotter_unitary(terms: List[PauliTerm], t: float, steps: int) -> np.ndarray:
    d = 2 ** len(terms[0].label)
    U = np.eye(d, dtype=complex)
    dt = t / steps
    # First-order Trotter: (prod_k exp(-i c_k P_k dt))^steps
    step_U = np.eye(d, dtype=complex)
    for term in terms:
        step_U = expm_pauli_term(term, term.coeff * dt) @ step_U
    for _ in range(steps):
        U = step_U @ U
    return U


def prepare_initial_state_from_basis_union(vec_i: np.ndarray, vec_j: np.ndarray, top_k: int = 2) -> np.ndarray:
    # Build a simple superposition from the union of the top-k basis states (by probability)
    p_i = probs_from_state(vec_i)
    p_j = probs_from_state(vec_j)
    idx_i = np.argsort(-p_i)[:top_k]
    idx_j = np.argsort(-p_j)[:top_k]
    idx = sorted(set(idx_i.tolist() + idx_j.tolist()))
    d = vec_i.shape[0]
    psi0 = np.zeros((d,), dtype=complex)
    for k in idx:
        psi0[k] = 1.0
    psi0 = psi0 / np.linalg.norm(psi0)
    return psi0


def leakage_proxy(terms: List[PauliTerm],
                  vec_i: np.ndarray,
                  vec_j: np.ndarray,
                  times: List[float],
                  trotter_steps: int) -> float:
    # Projector onto candidate subspace span{vec_i, vec_j}
    # Pi = |i><i| + |j><j|
    Pi = np.outer(vec_i, vec_i.conj()) + np.outer(vec_j, vec_j.conj())

    psi0 = prepare_initial_state_from_basis_union(vec_i, vec_j, top_k=2)

    leaks = []
    for t in times:
        U = trotter_unitary(terms, t=t, steps=trotter_steps)
        psit = U @ psi0
        in_sub = float(np.vdot(psit, Pi @ psit).real)
        leak = 1.0 - max(0.0, min(1.0, in_sub))
        leaks.append(leak)
    return float(np.mean(leaks))


# ----------------------------
# Signature + scoring + enrichment
# ----------------------------

def bin_int(x: float, step: float) -> int:
    return int(np.floor(x / step + 1e-12))


def signature_for_pair(vec_i: np.ndarray,
                       vec_j: np.ndarray,
                       dom_threshold: float,
                       ent_step: float,
                       leak_bin_step: float,
                       leak_value: float) -> Tuple[int, int, int, int, float]:
    p_i = probs_from_state(vec_i)
    p_j = probs_from_state(vec_j)

    ent_i = shannon_entropy(p_i)
    ent_j = shannon_entropy(p_j)
    dom_i = dominant_basis_count(p_i, dom_threshold)
    dom_j = dominant_basis_count(p_j, dom_threshold)

    dom = int(round(0.5 * (dom_i + dom_j)))
    ent_bi = bin_int(ent_i, ent_step)
    ent_bj = bin_int(ent_j, ent_step)
    leak_b = bin_int(leak_value, leak_bin_step)

    ent_med = 0.5 * (ent_bi + ent_bj)
    return (dom, ent_bi, ent_bj, leak_b, ent_med)


def interestingness_score(leakage_avg: float, dom: int) -> float:
    # Small dom and small leakage should score high.
    # Score in [0,1] approximately.
    # (1-leak) is in [0,1]; divide by (1+dom) to penalize large supports.
    base = max(0.0, min(1.0, 1.0 - leakage_avg))
    return float(base / (1.0 + max(0, dom)))


def stable_mask(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = scores.size
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.ceil(stable_frac * n)))
    # Top-k by score
    idx = np.argpartition(-scores, k - 1)[:k]
    mask = np.zeros((n,), dtype=bool)
    mask[idx] = True
    return mask


def enrichment_rows(sig_list: List[Tuple[int, int, int, int]],
                    stable: np.ndarray,
                    alpha: float,
                    min_overall: int,
                    min_stable: int) -> List[Tuple[Tuple[int, int, int, int], int, int, float]]:
    """
    Returns rows: (sig, overall_count, stable_count, enrichment_smoothed)
    with smoothing: add alpha pseudocount to both stable and non-stable buckets.
    """
    overall: Dict[Tuple[int, int, int, int], int] = {}
    stable_ct: Dict[Tuple[int, int, int, int], int] = {}

    for s, m in zip(sig_list, stable):
        overall[s] = overall.get(s, 0) + 1
        if bool(m):
            stable_ct[s] = stable_ct.get(s, 0) + 1

    N = len(sig_list)
    S = int(stable.sum())
    if N == 0 or S == 0:
        return []

    rows = []
    for s, oc in overall.items():
        sc = stable_ct.get(s, 0)
        if oc < min_overall or sc < min_stable:
            continue

        # Smoothed proportions
        p_stable = (sc + alpha) / (S + 2.0 * alpha)
        p_overall = (oc + alpha) / (N + 2.0 * alpha)
        enr = p_stable / p_overall
        rows.append((s, oc, sc, float(enr)))

    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def topK_families_by_ratio(rows_real, rows_null, topK: int) -> List[Tuple[Tuple[int, int, int, int], float, float, float]]:
    null_map = {s: enr for (s, _, _, enr) in rows_null}
    out = []
    for (s, _, _, enr_r) in rows_real:
        enr_n = null_map.get(s, 0.0)
        # Ratio with a tiny floor to avoid inf; we still want readable
        ratio = enr_r / max(1e-12, enr_n)
        out.append((s, enr_r, enr_n, float(ratio)))
    out.sort(key=lambda x: x[3], reverse=True)
    return out[:topK]


def bootstrap_ci_median_diff(x: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Bootstrap CI for median(x) - median(y).
    Returns (point, lo, hi).
    """
    if x.size == 0 or y.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = float(np.median(x) - np.median(y))
    diffs = []
    nx, ny = x.size, y.size
    for _ in range(B):
        bx = x[rng.integers(0, nx, size=nx)]
        by = y[rng.integers(0, ny, size=ny)]
        diffs.append(float(np.median(bx) - np.median(by)))
    diffs = np.sort(np.array(diffs, dtype=float))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return point, lo, hi


# ----------------------------
# Batch runner
# ----------------------------

def scan_model_for_batch(model: str,
                         n_seeds: int,
                         seed_offset: int,
                         n_qubits: int,
                         num_terms: int,
                         eps: float,
                         dom_threshold: float,
                         ent_step: float,
                         leak_step: float,
                         times: List[float],
                         trotter_steps: int,
                         rng_null: np.random.Generator) -> List[Candidate]:

    candidates: List[Candidate] = []

    for s in range(n_seeds):
        seed = seed_offset + s
        terms = build_random_hamiltonian(n_qubits=n_qubits, num_terms=num_terms, seed=seed)
        H = hamiltonian_matrix(terms)
        evals, evecs_real = diag_hermitian(H)

        pairs = near_degenerate_neighbor_pairs(evals, eps=eps)
        if not pairs:
            continue

        if model == "REAL":
            evecs = evecs_real
        elif model == "NULL_HAAR_BASIS":
            d = evecs_real.shape[0]
            evecs = haar_random_unitary(d, rng_null)
        else:
            raise ValueError(f"Unknown model: {model}")

        for (i, j) in pairs:
            vec_i = evecs[:, i]
            vec_j = evecs[:, j]

            leak = leakage_proxy(terms, vec_i, vec_j, times=times, trotter_steps=trotter_steps)
            sig_dom, bi, bj, bl, ent_med = signature_for_pair(
                vec_i, vec_j, dom_threshold=dom_threshold, ent_step=ent_step,
                leak_bin_step=leak_step, leak_value=leak
            )
            sig = (sig_dom, bi, bj, bl)
            score = interestingness_score(leakage_avg=leak, dom=sig_dom)

            candidates.append(Candidate(sig=sig, score=score, ent_bin_med=ent_med, leakage_avg=leak))

    return candidates


def run_one_batch(batch_id: int, args, seed_offset: int) -> dict:
    t0 = time.time()

    rng_null = np.random.default_rng(args.null_seed + 100000 * batch_id)

    cands_real = scan_model_for_batch(
        model="REAL",
        n_seeds=args.seeds,
        seed_offset=seed_offset,
        n_qubits=args.qubits,
        num_terms=args.terms,
        eps=args.eps,
        dom_threshold=args.dom_threshold,
        ent_step=args.ent_step,
        leak_step=args.leak_step,
        times=args.times,
        trotter_steps=args.trotter_steps,
        rng_null=rng_null,
    )

    cands_null = scan_model_for_batch(
        model="NULL_HAAR_BASIS",
        n_seeds=args.seeds,
        seed_offset=seed_offset,
        n_qubits=args.qubits,
        num_terms=args.terms,
        eps=args.eps,
        dom_threshold=args.dom_threshold,
        ent_step=args.ent_step,
        leak_step=args.leak_step,
        times=args.times,
        trotter_steps=args.trotter_steps,
        rng_null=rng_null,
    )

    elapsed = time.time() - t0

    def summarize(model_name: str, cands: List[Candidate]):
        scores = np.array([c.score for c in cands], dtype=float)
        leaks = np.array([c.leakage_avg for c in cands], dtype=float)
        doms = np.array([c.sig[0] for c in cands], dtype=int)
        return {
            "model": model_name,
            "candidates": len(cands),
            "score_mean": float(scores.mean()) if scores.size else float("nan"),
            "score_median": float(np.median(scores)) if scores.size else float("nan"),
            "score_max": float(scores.max()) if scores.size else float("nan"),
            "leak_mean": float(leaks.mean()) if leaks.size else float("nan"),
            "leak_median": float(np.median(leaks)) if leaks.size else float("nan"),
            "leak_min": float(leaks.min()) if leaks.size else float("nan"),
            "dom_median": int(np.median(doms)) if doms.size else -1,
        }

    s_real = summarize("REAL", cands_real)
    s_null = summarize("NULL_HAAR_BASIS", cands_null)

    # Stable sets & enrichment
    scores_r = np.array([c.score for c in cands_real], dtype=float)
    scores_n = np.array([c.score for c in cands_null], dtype=float)
    stable_r = stable_mask(scores_r, args.stable_frac)
    stable_n = stable_mask(scores_n, args.stable_frac)

    sigs_r = [c.sig for c in cands_real]
    sigs_n = [c.sig for c in cands_null]

    rows_r = enrichment_rows(sigs_r, stable_r, alpha=args.smooth_alpha,
                            min_overall=args.min_overall, min_stable=args.min_stable)
    rows_n = enrichment_rows(sigs_n, stable_n, alpha=args.smooth_alpha,
                            min_overall=args.min_overall, min_stable=args.min_stable)

    top_ratio = topK_families_by_ratio(rows_r, rows_n, topK=args.topK)

    # Distribution effect size: median(entropy_bin_med) REAL - NULL + bootstrap CI
    ent_r = np.array([c.ent_bin_med for c in cands_real], dtype=float)
    ent_n = np.array([c.ent_bin_med for c in cands_null], dtype=float)
    ci = bootstrap_ci_median_diff(ent_r, ent_n, B=args.bootstrap, rng=np.random.default_rng(args.bootstrap_seed + batch_id))

    return {
        "batch_id": batch_id,
        "seed_offset": seed_offset,
        "elapsed_s": float(elapsed),
        "summary_real": s_real,
        "summary_null": s_null,
        "top_ratio": top_ratio,         # list of (sig, enr_real, enr_null, ratio)
        "median_diff_entropybin": ci,    # (point, lo, hi)
        "rows_real_count": len(rows_r),
        "rows_null_count": len(rows_n),
    }


def overlap(set_a: set, set_b: set) -> Tuple[int, float]:
    inter = len(set_a & set_b)
    uni = len(set_a | set_b) if (set_a | set_b) else 1
    return inter, inter / uni


def format_sig(sig: Tuple[int, int, int, int]) -> str:
    return f"({sig[0]}, {sig[1]}, {sig[2]}, {sig[3]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--terms", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seed_stride", type=int, default=100000)

    ap.add_argument("--eps", type=float, default=0.050)

    ap.add_argument("--dom_threshold", type=float, default=0.250)
    ap.add_argument("--ent_step", type=float, default=0.100)
    ap.add_argument("--leak_step", type=float, default=0.050)

    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--trotter_steps", type=int, default=8)

    ap.add_argument("--stable_frac", type=float, default=0.010)
    ap.add_argument("--topK", type=int, default=25)

    ap.add_argument("--smooth_alpha", type=float, default=1.0)
    ap.add_argument("--min_overall", type=int, default=10)
    ap.add_argument("--min_stable", type=int, default=3)

    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap_seed", type=int, default=12345)
    ap.add_argument("--null_seed", type=int, default=777)

    ap.add_argument("--out", type=str, default="v0_7_convergence_batches_output_1.txt")
    args = ap.parse_args()

    tee = Tee(args.out)
    sys.stdout = tee

    try:
        print("=== v0_7: Convergence Batches (Matched, FAIR) ===")
        print(f"Qubits: {args.qubits} (d={2**args.qubits}) | terms={args.terms} | seeds_per_batch={args.seeds} | batches={args.batches}")
        print(f"Near-degenerate neighbor eps={args.eps:.3f}")
        print(f"Dominant threshold={args.dom_threshold:.3f} | ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
        print(f"Leakage proxy: times={args.times} | trotter_steps={args.trotter_steps}")
        print(f"Evidence: stable_frac={args.stable_frac:.3f} | topK={args.topK} | smoothing_alpha={args.smooth_alpha:.2f}")
        print(f"Min-count filter: min_overall={args.min_overall} | min_stable={args.min_stable}")
        print(f"Bootstrap: B={args.bootstrap} | seed={args.bootstrap_seed}")
        print(f"Output: {args.out}")
        print("")

        results = []
        for b in range(args.batches):
            seed_offset = args.seed0 + b * args.seed_stride
            r = run_one_batch(batch_id=b, args=args, seed_offset=seed_offset)
            results.append(r)

            sr = r["summary_real"]
            sn = r["summary_null"]
            print("----------------------------------------------")
            print(f"Batch {b+1}/{args.batches} | seed_offset={seed_offset} | elapsed={r['elapsed_s']:.1f}s")
            print(f"REAL candidates={sr['candidates']} | score(mean/median/max)={sr['score_mean']:.3f}/{sr['score_median']:.3f}/{sr['score_max']:.3f} | leakage(mean/median/min)={sr['leak_mean']:.3f}/{sr['leak_median']:.3f}/{sr['leak_min']:.3f}")
            print(f"NULL candidates={sn['candidates']} | score(mean/median/max)={sn['score_mean']:.3f}/{sn['score_median']:.3f}/{sn['score_max']:.3f} | leakage(mean/median/min)={sn['leak_mean']:.3f}/{sn['leak_median']:.3f}/{sn['leak_min']:.3f}")

            point, lo, hi = r["median_diff_entropybin"]
            print(f"Effect size: median(entropy_bin_med) REAL - NULL = {point:.3f}  (bootstrap 95% CI [{lo:.3f}, {hi:.3f}])")
            print(f"Enrichment rows (after filters): REAL={r['rows_real_count']} | NULL={r['rows_null_count']}")
            print("")
            print("Top families by cross-model ratio (REAL_enr / NULL_enr), smoothed + filtered:")
            print("Format: sig=(dom, ent_i_bin, ent_j_bin, leak_bin) | enr_REAL | enr_NULL | ratio")
            for (sig, er, en, ratio) in r["top_ratio"][:min(args.topK, 10)]:
                print(f"  {format_sig(sig)} | {er:7.2f}x | {en:7.2f}x | {ratio:9.2f}x")
            if len(r["top_ratio"]) > 10:
                print("  ... (truncated display)")
            print("")

        # ----------------------------
        # (1) Top-family stability: overlap between batches
        # ----------------------------
        top_sets = []
        for r in results:
            s = set([sig for (sig, _, _, _) in r["top_ratio"]])
            top_sets.append(s)

        print("==============================================")
        print("Convergence diagnostics across batches")
        print("")
        print("(1) Top-family stability: overlap(Top-25) between runs")
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                inter, jac = overlap(top_sets[i], top_sets[j])
                print(f"  Batch {i+1} vs {j+1}: intersection={inter} | Jaccard={jac:.3f}")

        if len(top_sets) >= 3:
            inter_all = set.intersection(*top_sets)
            uni_all = set.union(*top_sets)
            print(f"  All-batches: intersection={len(inter_all)} | union={len(uni_all)} | Jaccard={len(inter_all)/max(1,len(uni_all)):.3f}")
            if len(inter_all) > 0:
                show = list(sorted(inter_all))[:10]
                print("  Example shared families (up to 10): " + ", ".join(format_sig(s) for s in show))
        print("")

        # ----------------------------
        # (2) Distribution-level effect size aggregation
        # ----------------------------
        print("(2) Distribution effect-size: median(entropy_bin_med) REAL - NULL per batch")
        points = []
        for r in results:
            point, lo, hi = r["median_diff_entropybin"]
            points.append(point)
            print(f"  Batch {r['batch_id']+1}: {point:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")
        if len(points) > 0:
            print(f"  Across-batch mean(point)={float(np.mean(points)):.3f} | std(point)={float(np.std(points)):.3f}")
        print("")

        # ----------------------------
        # (3) Readability guardrails explanation
        # ----------------------------
        print("(3) Cross-model enrichment: smoothing + minimum-count filter")
        print("  - smoothing_alpha controls pseudocount smoothing in enrichment computation (prevents infinities).")
        print("  - min_overall/min_stable remove families that are too rare to interpret robustly.")
        print("")
        print("Interpretation guideline:")
        print("  - Convergence in (1) means the *same families* keep showing up as top-ranked under matched settings.")
        print("  - Stability in (2) means the *distribution-level* structure difference is not a single-run artifact.")
        print("  - If both hold, the result is substantially more credible than a single scan.")
        print("")
        print("=== End of v0_7 convergence batches ===")

    finally:
        try:
            sys.stdout = tee._stdout
        except Exception:
            pass
        tee.close()


if __name__ == "__main__":
    main()
