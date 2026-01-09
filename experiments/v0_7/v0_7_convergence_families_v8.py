#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v8.py
Convergence + baseline calibration with a *non-trivial* stability proxy.

What v8 changes (without changing the high-level scientific intent):
- Fixes the "leakage collapses to ~0" failure mode by redefining leakage as
  *computational-subspace leakage* (dominant basis support), NOT eigen-subspace leakage.
  (Eigen-subspaces are invariant by definition -> trivial leakage for both REAL and NULL.)
- Uses deterministic dominant-basis selection (top-m basis states per eigenvector) so
  signatures are more stable and comparable between REAL and NULL.
- Adds smoothing + minimum-count filtering so cross-model ratios are readable and not
  dominated by zero-count divisions.
- Keeps the v0.7 convergence structure: 3 batches, overlap stability, distribution-level
  effect size, and cross-model enrichment.

Outputs:
- A single UTF-8 text report (path printed at start).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from collections import Counter
from typing import Dict, List, Tuple, Iterable

import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def shannon_entropy(p: np.ndarray, eps: float = 1e-15) -> float:
    """Shannon entropy (natural log) of a probability vector."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def bin_index(x: float, step: float) -> int:
    if step <= 0:
        return 0
    return int(math.floor(x / step))


def jaccard(a: Iterable, b: Iterable) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# -----------------------------
# Hamiltonians
# -----------------------------

PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


def random_pauli_string(n: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("IXYZ")) for _ in range(n))


def build_random_hamiltonian(n_qubits: int, num_terms: int, seed: int) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Build a dense Hamiltonian H = sum_k c_k P_k with Pauli strings."""
    rng = np.random.default_rng(seed)
    terms: List[Tuple[str, float]] = []
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)

    for _ in range(num_terms):
        p = random_pauli_string(n_qubits, rng)
        if set(p) == {"I"}:
            continue
        c = float(rng.normal(loc=0.0, scale=1.0))
        P = kron_n([PAULI_MATS[ch] for ch in p])
        H += c * P
        terms.append((p, c))

    H = 0.5 * (H + H.conj().T)
    return H, terms


def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    E, V = np.linalg.eigh(H)
    return E.real, V


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    ph = np.diag(R) / np.abs(np.diag(R))
    Q = Q * ph.conj()
    return Q


def spectrum_matched_null(E: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    d = len(E)
    U = haar_random_unitary(d, rng)
    H_null = U @ np.diag(E) @ U.conj().T
    V_null = U
    return H_null, V_null


# -----------------------------
# Candidate definition (non-trivial leakage)
# -----------------------------

@dataclasses.dataclass
class Candidate:
    model: str
    seed: int
    i: int
    j: int
    dE: float
    dom_idx: Tuple[int, ...]
    dom_count: int
    ent_i: float
    ent_j: float
    leak_avg: float
    score: float
    sig_key: Tuple[int, int, int, int]


def top_m_support(vec: np.ndarray, m: int) -> List[int]:
    p = np.abs(vec) ** 2
    order = np.argsort(-p)
    return list(map(int, order[:m]))


def evolve_state_from_eigendecomp(V: np.ndarray, E: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    phases = np.exp(-1j * E * t)
    coeff = V.conj().T @ psi0
    coeff_t = phases * coeff
    return V @ coeff_t


def computational_leakage(V: np.ndarray, E: np.ndarray, dom_idx: Tuple[int, ...], times: List[float]) -> float:
    d = V.shape[0]
    if len(dom_idx) == 0:
        return 1.0
    psi0 = np.zeros((d,), dtype=complex)
    amp = 1.0 / math.sqrt(len(dom_idx))
    for k in dom_idx:
        psi0[k] = amp

    leaks = []
    for t in times:
        psi_t = evolve_state_from_eigendecomp(V, E, psi0, t)
        p_in = float(np.sum(np.abs(psi_t[list(dom_idx)]) ** 2))
        leaks.append(max(0.0, 1.0 - p_in))
    return float(np.mean(leaks))


def interestingness_score(dom_count: int, leak_avg: float, ent_i: float, ent_j: float,
                          w_leak: float = 0.70, w_dom: float = 0.20, w_ent: float = 0.10) -> float:
    leak_term = 1.0 - max(0.0, min(1.0, leak_avg))
    dom_term = 1.0 / (1.0 + max(0, dom_count - 1))
    ent_term = 1.0 / (1.0 + 0.5 * (ent_i + ent_j))
    s = w_leak * leak_term + w_dom * dom_term + w_ent * ent_term
    return float(max(0.0, min(1.0, s)))


def enumerate_near_degenerate_pairs(E: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    for k in range(len(E) - 1):
        de = float(abs(E[k+1] - E[k]))
        if de < eps:
            pairs.append((k, k+1, de))
    return pairs


def build_candidates_for_seed(model: str,
                              seed: int,
                              n_qubits: int,
                              num_terms: int,
                              eps: float,
                              top_m: int,
                              max_union: int,
                              ent_step: float,
                              leak_step: float,
                              times: List[float],
                              rng_null: np.random.Generator) -> List[Candidate]:
    H_real, _ = build_random_hamiltonian(n_qubits, num_terms, seed=seed)
    E_real, V_real = diagonalize(H_real)

    if model == "REAL":
        E, V = E_real, V_real
    elif model == "NULL_HAAR_BASIS":
        _, Vn = spectrum_matched_null(E_real, rng_null)
        E, V = E_real, Vn
    else:
        raise ValueError(f"Unknown model: {model}")

    pairs = enumerate_near_degenerate_pairs(E, eps=eps)
    if not pairs:
        return []

    cands: List[Candidate] = []
    for (i, j, de) in pairs:
        psi_i = V[:, i]
        psi_j = V[:, j]

        supp = top_m_support(psi_i, top_m) + top_m_support(psi_j, top_m)
        seen = set()
        union: List[int] = []
        for idx in supp:
            if idx not in seen:
                seen.add(idx)
                union.append(idx)
        union = union[:max_union]
        dom_idx = tuple(sorted(union))
        dom_count = len(dom_idx)

        ent_i = shannon_entropy(np.abs(psi_i) ** 2)
        ent_j = shannon_entropy(np.abs(psi_j) ** 2)

        leak = computational_leakage(V, E, dom_idx, times=times)
        score = interestingness_score(dom_count, leak, ent_i, ent_j)

        sig = (dom_count,
               bin_index(ent_i, ent_step),
               bin_index(ent_j, ent_step),
               bin_index(leak, leak_step))

        cands.append(Candidate(
            model=model,
            seed=seed,
            i=i,
            j=j,
            dE=de,
            dom_idx=dom_idx,
            dom_count=dom_count,
            ent_i=ent_i,
            ent_j=ent_j,
            leak_avg=leak,
            score=score,
            sig_key=sig,
        ))

    return cands


# -----------------------------
# Enrichment + calibration
# -----------------------------

@dataclasses.dataclass
class EnrichmentRow:
    sig: Tuple[int, int, int, int]
    overall: int
    stable: int
    enrichment: float


def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(round(stable_frac * n)))
    thr = np.partition(scores, n - k)[n - k]
    return scores >= thr


def enrichment_rows(sigs: List[Tuple[int, int, int, int]],
                    stable_mask: np.ndarray,
                    alpha: float = 0.5,
                    min_overall: int = 10,
                    min_stable: int = 2) -> List[EnrichmentRow]:
    overall = Counter(sigs)
    stable = Counter([s for s, m in zip(sigs, stable_mask) if m])

    n_all = len(sigs)
    n_stable = int(stable_mask.sum())
    if n_all == 0 or n_stable == 0:
        return []

    rows: List[EnrichmentRow] = []
    for s, o in overall.items():
        st = stable.get(s, 0)
        if o < min_overall or st < min_stable:
            continue
        p_stable = (st + alpha) / (n_stable + alpha * 2)
        p_all = (o + alpha) / (n_all + alpha * 2)
        enr = p_stable / p_all
        rows.append(EnrichmentRow(sig=s, overall=o, stable=st, enrichment=float(enr)))

    rows.sort(key=lambda r: r.enrichment, reverse=True)
    return rows


def topk_sigs(rows: List[EnrichmentRow], k: int) -> List[Tuple[int, int, int, int]]:
    return [r.sig for r in rows[:k]]


def bootstrap_effect_size(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, Tuple[float, float]]:
    if len(x) == 0 or len(y) == 0:
        return float("nan"), (float("nan"), float("nan"))
    est = float(np.median(x) - np.median(y))
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(float(np.median(xb) - np.median(yb)))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return est, (float(lo), float(hi))


# -----------------------------
# Batch runner + reporting
# -----------------------------

def run_batch(batch_id: int, seed_offset: int, args: argparse.Namespace, rng_null: np.random.Generator) -> Dict[str, List[Candidate]]:
    out: Dict[str, List[Candidate]] = {"REAL": [], "NULL_HAAR_BASIS": []}
    t0 = time.time()
    for s in range(args.seeds):
        seed = seed_offset + s
        out["REAL"].extend(build_candidates_for_seed(
            model="REAL",
            seed=seed,
            n_qubits=args.qubits,
            num_terms=args.terms,
            eps=args.eps,
            top_m=args.top_m,
            max_union=args.max_union,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=args.times,
            rng_null=rng_null,
        ))
        out["NULL_HAAR_BASIS"].extend(build_candidates_for_seed(
            model="NULL_HAAR_BASIS",
            seed=seed,
            n_qubits=args.qubits,
            num_terms=args.terms,
            eps=args.eps,
            top_m=args.top_m,
            max_union=args.max_union,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=args.times,
            rng_null=rng_null,
        ))
    elapsed = time.time() - t0
    print(f"Batch {batch_id}: seeds={args.seeds} offset={seed_offset} | REAL_cands={len(out['REAL'])} | NULL_cands={len(out['NULL_HAAR_BASIS'])} | elapsed={elapsed:.1f}s")
    return out


def summarize_model(cands: List[Candidate]) -> Dict[str, float]:
    if not cands:
        return {"candidates": 0}
    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak_avg for c in cands], dtype=float)
    doms = np.array([c.dom_count for c in cands], dtype=int)
    entm = np.array([0.5*(c.ent_i+c.ent_j) for c in cands], dtype=float)
    return {
        "candidates": len(cands),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
        "leak_mean": float(leaks.mean()),
        "leak_median": float(np.median(leaks)),
        "leak_min": float(leaks.min()),
        "dom_median": float(np.median(doms)),
        "ent_median": float(np.median(entm)),
    }


def print_top_rows(label: str, rows: List[EnrichmentRow], topk: int) -> None:
    print(f"\nTop {topk} signature families for {label} (stable vs overall) [smoothed, filtered]:")
    print("Format: sig=(dom, ent_bin_i, ent_bin_j, leak_bin) | overall | stable | enrichment")
    for r in rows[:topk]:
        print(f"  {r.sig} | {r.overall:5d} | {r.stable:5d} | {r.enrichment:8.2f}x")
    if len(rows) > topk:
        print(f"  ... ({len(rows)} total rows after filters)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--terms", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5000, help="Seeds per batch")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--batch_stride", type=int, default=100000)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--top_m", type=int, default=2)
    ap.add_argument("--max_union", type=int, default=4)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--ent_step", type=float, default=0.10)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=15)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="v0_7_convergence_families_v8_output.txt")
    args = ap.parse_args()

    # UTF-8 output (avoid Windows cp1252 issues).
    f = open(args.out, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s: str):
            for ff in self.files:
                ff.write(s)
        def flush(self):
            for ff in self.files:
                ff.flush()

    import sys
    sys.stdout = Tee(sys.stdout, f)

    print("=== v0.7 Convergence (v8): Non-trivial stability + matched null ===")
    print(f"Qubits: {args.qubits} (d={2**args.qubits}) | terms={args.terms}")
    print(f"Batches: {args.batches} × {args.seeds} seeds | eps={args.eps}")
    print(f"Dominant support: top_m={args.top_m} per eigenvector | max_union={args.max_union}")
    print(f"Leakage proxy: times={args.times} (computational-subspace leakage)")
    print(f"Signature binning: ent_step={args.ent_step} | leak_step={args.leak_step}")
    print(f"Evidence: stable_frac={args.stable_frac:.3f} | topK={args.topK} | min_overall={args.min_overall} | min_stable={args.min_stable} | alpha={args.alpha}")
    print(f"Bootstrap: {args.bootstrap} (median-diff CI) | seed={args.bootstrap_seed}")
    print(f"Output: {args.out}\n")

    rng_null = np.random.default_rng(999)
    rng_boot = np.random.default_rng(args.bootstrap_seed)

    batches: List[Dict[str, List[Candidate]]] = []
    for b in range(args.batches):
        offset = args.seed0 + b * args.batch_stride
        batches.append(run_batch(b, offset, args, rng_null=rng_null))

    top_fams: Dict[str, List[List[Tuple[int, int, int, int]]]] = {"REAL": [], "NULL_HAAR_BASIS": []}

    for b, data in enumerate(batches):
        for model_key in ["REAL", "NULL_HAAR_BASIS"]:
            cands = data[model_key]
            sigs = [c.sig_key for c in cands]
            scores = np.array([c.score for c in cands], dtype=float)
            m = stable_mask_from_scores(scores, args.stable_frac)
            rows = enrichment_rows(sigs, m, alpha=args.alpha, min_overall=args.min_overall, min_stable=args.min_stable)
            top_fams[model_key].append(topk_sigs(rows, args.topK))

            s = summarize_model(cands)
            print(f"\n--- Batch {b} | Model {model_key} ---")
            print(f"candidates={s.get('candidates',0)} | score(mean/median/max)={s.get('score_mean',0):.3f}/{s.get('score_median',0):.3f}/{s.get('score_max',0):.3f} | "
                  f"leak(mean/median/min)={s.get('leak_mean',0):.3f}/{s.get('leak_median',0):.3f}/{s.get('leak_min',0):.3f} | dom(median)={s.get('dom_median',0):.1f} | ent(median)={s.get('ent_median',0):.2f}")
            print_top_rows(model_key, rows, min(args.topK, 10))

    # (1) Top-family stability
    print("\n=== (1) Top-family stability across batches (Jaccard overlap) ===")
    for model_key in ["REAL", "NULL_HAAR_BASIS"]:
        fams = top_fams[model_key]
        overlaps = []
        for i in range(len(fams)):
            for j in range(i+1, len(fams)):
                overlaps.append(jaccard(fams[i], fams[j]))
        print(f"{model_key}: Top-{args.topK} Jaccard overlaps = {[round(x,3) for x in overlaps]} | mean={np.mean(overlaps):.3f}")

    # (2) Distribution-level effect sizes (pooled)
    print("\n=== (2) Distribution-level effect sizes (REAL vs NULL) ===")
    real_all = [c for b in batches for c in b["REAL"]]
    null_all = [c for b in batches for c in b["NULL_HAAR_BASIS"]]
    real_leak = np.array([c.leak_avg for c in real_all], dtype=float)
    null_leak = np.array([c.leak_avg for c in null_all], dtype=float)
    real_ent = np.array([0.5*(c.ent_i+c.ent_j) for c in real_all], dtype=float)
    null_ent = np.array([0.5*(c.ent_i+c.ent_j) for c in null_all], dtype=float)

    est_leak, ci_leak = bootstrap_effect_size(real_leak, null_leak, args.bootstrap, rng_boot)
    est_ent, ci_ent = bootstrap_effect_size(real_ent, null_ent, args.bootstrap, rng_boot)
    print(f"Leakage median difference (REAL - NULL): {est_leak:.4f} | 95% CI [{ci_leak[0]:.4f}, {ci_leak[1]:.4f}]")
    print(f"Entropy  median difference (REAL - NULL): {est_ent:.4f} | 95% CI [{ci_ent[0]:.4f}, {ci_ent[1]:.4f}]")
    print("(Interpretation: a non-zero, reproducible difference supports non-chance structure under matched settings.)")

    # (3) Cross-model enrichment (pooled)
    print("\n=== (3) Cross-model enrichment (smoothed; filtered; pooled) ===")
    def pooled_rows(cands: List[Candidate]) -> List[EnrichmentRow]:
        sigs = [c.sig_key for c in cands]
        scores = np.array([c.score for c in cands], dtype=float)
        m = stable_mask_from_scores(scores, args.stable_frac)
        return enrichment_rows(sigs, m, alpha=args.alpha, min_overall=args.min_overall, min_stable=args.min_stable)

    rows_real = pooled_rows(real_all)
    rows_null = pooled_rows(null_all)
    enr_real = {r.sig: r.enrichment for r in rows_real}
    enr_null = {r.sig: r.enrichment for r in rows_null}
    all_sigs = set(enr_real.keys()) | set(enr_null.keys())

    cross = []
    for s in all_sigs:
        r = enr_real.get(s, 1.0)
        n = enr_null.get(s, 1.0)
        cross.append((r / n, s, r, n))
    cross.sort(reverse=True, key=lambda x: x[0])

    print("Format: sig | enr_REAL | enr_NULL | ratio(REAL/NULL)")
    for ratio, s, r, n in cross[:args.topK]:
        print(f"  {s} | {r:7.2f}x | {n:7.2f}x | {ratio:7.2f}x")
    if len(cross) > args.topK:
        print(f"  ... ({len(cross)} total cross-sigs after filters)")

    print("\n=== Notes (scientific reading) ===")
    print("1) Leakage is now defined relative to a computational-basis encoding support, not eigen-subspace invariance.")
    print("2) Strong claims require convergence (stable overlaps) + reproducible REAL vs NULL separation (effect sizes / ratios).")
    print("3) If effects are near-zero with tight CIs, the proxy/encoding may still be insufficiently sensitive at n=3.")

    try:
        f.flush()
        f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
