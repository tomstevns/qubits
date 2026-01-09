#!/usr/bin/env python3
"""
v0_7_convergence_families_v5.py
================================

Purpose
-------
This script is a *convergence / replicability* harness for the v0_6.1 -> v0_7 line of work.

It implements, in one place, the three evidence-oriented checks discussed previously:

1) Three independent batches (e.g., 3 × 5,000 seeds with different seed offsets)
   - measures overlap of Top-K signature families between runs.

2) Distribution-level effect sizes (REAL vs NULL) with uncertainty
   - reports medians of entropy/leakage/score, plus bootstrap confidence intervals.

3) Readable cross-model enrichment for signature families
   - uses:
       * integer family keys (binned descriptors)
       * smoothing (Laplace prior)
       * minimum-count filtering
       * and reports REAL/NULL ratios with CIs that do not explode numerically.

Design choices (kept consistent with the "SigV2 matched FAIR" philosophy)
------------------------------------------------------------------------
- REAL model:
    Random 3-qubit Pauli-sum Hamiltonians (SparsePauliOp-like but implemented directly as matrices).
- NULL model (spectrum-matched, Haar-basis scrambled):
    Uses the *same* eigenvalues as REAL, but replaces eigenvectors by Haar-random unitary columns.

- Candidate definition:
    Near-degenerate NEIGHBOR pairs in the sorted spectrum (i, i+1) with |E[i+1]-E[i]| < eps.

- Structural descriptors:
    * dom_count: number of computational basis states in the union-dominant set
    * entropy_i, entropy_j: Shannon entropy (base 2) of |psi|^2 for each eigenvector
    * leakage: mean leakage under exact time evolution for a simple prepared state

- Leakage proxy:
    Let D be the union of dominant basis indices (from the two eigenvectors).
    Prepare |psi0> = uniform superposition over D.
    Evolve exactly with psi(t) = exp(-i H t) |psi0>, using eigen-decomposition.
    Leakage(t) = 1 - sum_{x in D} |<x|psi(t)>|^2
    Mean leakage across times is used.

Important: This file avoids any Unicode output to prevent Windows cp1252 failures.
All output is written as UTF-8.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# I/O helper: tee to file (utf-8)
# -----------------------------

class Tee:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout

    def write(self, s: str) -> None:
        self._stdout.write(s)
        self._f.write(s)

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
            self._f.close()
        except Exception:
            pass


# -----------------------------
# Physics helpers (small n=3)
# -----------------------------

PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

PAULI_LABELS = ["I", "X", "Y", "Z"]


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for a in ops[1:]:
        out = np.kron(out, a)
    return out


def random_pauli_string(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(PAULI_LABELS, size=n))


def build_random_hamiltonian(n_qubits: int, num_terms: int, rng: np.random.Generator) -> np.ndarray:
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for _ in range(num_terms):
        p = random_pauli_string(rng, n_qubits)
        ops = [PAULI[ch] for ch in p]
        Pk = kron_n(ops)
        ck = float(rng.uniform(-1.0, 1.0))
        H += ck * Pk
    H = 0.5 * (H + H.conj().T)
    return H


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph
    return q


def shannon_entropy(p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p.real, 0.0, 1.0)
    p = p / (p.sum() + eps)
    m = p > eps
    return float(-np.sum(p[m] * np.log2(p[m])))


def dominant_index_set(psi: np.ndarray, dominant_threshold: float, cum_prob_target: float = 0.90) -> List[int]:
    p = np.abs(psi) ** 2
    idx = np.where(p >= dominant_threshold)[0].tolist()
    if len(idx) > 0:
        return sorted(idx)
    order = np.argsort(-p)
    s = 0.0
    out = []
    for i in order:
        out.append(int(i))
        s += float(p[i])
        if s >= cum_prob_target:
            break
    return sorted(out)


def exact_time_evolution_leakage(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    dom_set: List[int],
    times: List[float],
) -> float:
    dim = eigvecs.shape[0]
    if len(dom_set) == 0:
        return 1.0

    psi0 = np.zeros(dim, dtype=complex)
    amp = 1.0 / math.sqrt(len(dom_set))
    for k in dom_set:
        psi0[k] = amp

    a = eigvecs.conj().T @ psi0
    leaks = []
    mask = np.zeros(dim, dtype=bool)
    mask[dom_set] = True
    for t in times:
        phase = np.exp(-1j * eigvals * t)
        psi_t = eigvecs @ (phase * a)
        p = np.abs(psi_t) ** 2
        stay = float(np.sum(p[mask]))
        leaks.append(max(0.0, 1.0 - stay))
    return float(np.mean(leaks))


# -----------------------------
# Candidate + signature representation
# -----------------------------

@dataclass(frozen=True)
class SignatureKey:
    dom_bin: int
    ent_i_bin: int
    ent_j_bin: int
    leak_bin: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.dom_bin, self.ent_i_bin, self.ent_j_bin, self.leak_bin)


@dataclass
class Candidate:
    sig: SignatureKey
    score: float
    leakage: float
    ent_i: float
    ent_j: float
    dom_count: int


def bin_int(x: float, step: float) -> int:
    return int(math.floor(x / step + 1e-12))


def make_signature(
    dom_count: int,
    ent_i: float,
    ent_j: float,
    leakage: float,
    ent_step: float,
    leak_step: float,
) -> SignatureKey:
    ei = bin_int(ent_i, ent_step)
    ej = bin_int(ent_j, ent_step)
    if ej < ei:
        ei, ej = ej, ei
    return SignatureKey(dom_bin=dom_count, ent_i_bin=ei, ent_j_bin=ej, leak_bin=bin_int(leakage, leak_step))


def scan_one_hamiltonian(
    H: np.ndarray,
    eps: float,
    dominant_threshold: float,
    times: List[float],
    ent_step: float,
    leak_step: float,
) -> List[Candidate]:
    eigvals, eigvecs = np.linalg.eigh(H)
    dim = H.shape[0]
    cands: List[Candidate] = []

    for i in range(dim - 1):
        if abs(float(eigvals[i + 1] - eigvals[i])) >= eps:
            continue

        psi_i = eigvecs[:, i]
        psi_j = eigvecs[:, i + 1]

        p_i = np.abs(psi_i) ** 2
        p_j = np.abs(psi_j) ** 2
        ent_i = shannon_entropy(p_i)
        ent_j = shannon_entropy(p_j)

        dom_i = dominant_index_set(psi_i, dominant_threshold)
        dom_j = dominant_index_set(psi_j, dominant_threshold)
        dom_set = sorted(set(dom_i).union(dom_j))
        dom_count = len(dom_set)

        leakage = exact_time_evolution_leakage(eigvals, eigvecs, dom_set, times)
        stability = max(0.0, 1.0 - leakage)
        penalty = 1.0 / math.sqrt(max(1, dom_count))
        score = float(min(1.0, stability * penalty))

        sig = make_signature(dom_count, ent_i, ent_j, leakage, ent_step, leak_step)
        cands.append(Candidate(sig=sig, score=score, leakage=leakage, ent_i=ent_i, ent_j=ent_j, dom_count=dom_count))

    return cands


def scan_batch(
    n_qubits: int,
    num_terms: int,
    seeds: int,
    seed_offset: int,
    eps: float,
    dominant_threshold: float,
    times: List[float],
    ent_step: float,
    leak_step: float,
    rng_seed: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    dim = 2 ** n_qubits
    _rng = np.random.default_rng(rng_seed)

    real_all: List[Candidate] = []
    null_all: List[Candidate] = []

    for s in range(seeds):
        seed = seed_offset + s
        rng_s = np.random.default_rng(int(seed))

        H_real = build_random_hamiltonian(n_qubits, num_terms, rng_s)
        eigvals, _eigvecs = np.linalg.eigh(H_real)

        real_all.extend(scan_one_hamiltonian(H_real, eps, dominant_threshold, times, ent_step, leak_step))

        U = haar_unitary(dim, rng_s)
        H_null = U @ np.diag(eigvals) @ U.conj().T
        null_all.extend(scan_one_hamiltonian(H_null, eps, dominant_threshold, times, ent_step, leak_step))

    return real_all, null_all


# -----------------------------
# Enrichment + overlap + effects
# -----------------------------

def stable_mask(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(math.ceil(stable_frac * scores.size)))
    thr = np.partition(scores, -k)[-k]
    return scores >= thr


def enrichment_table(
    sigs: List[SignatureKey],
    stable: np.ndarray,
    alpha: float,
    min_overall: int,
) -> Dict[SignatureKey, Tuple[int, int, float]]:
    overall: Dict[SignatureKey, int] = {}
    st: Dict[SignatureKey, int] = {}
    for i, s in enumerate(sigs):
        overall[s] = overall.get(s, 0) + 1
        if stable[i]:
            st[s] = st.get(s, 0) + 1

    out: Dict[SignatureKey, Tuple[int, int, float]] = {}
    for s, o in overall.items():
        if o < min_overall:
            continue
        k = st.get(s, 0)
        conc = (k + alpha) / (o + alpha)
        out[s] = (o, k, float(conc))
    return out


def top_families_by_ratio(
    real_tab: Dict[SignatureKey, Tuple[int, int, float]],
    null_tab: Dict[SignatureKey, Tuple[int, int, float]],
    topK: int,
    smooth_floor: float = 1e-12,
) -> List[Tuple[float, SignatureKey, Tuple[int, int, float], Tuple[int, int, float]]]:
    ranked = []
    for sig, (o_r, k_r, conc_r) in real_tab.items():
        if sig not in null_tab:
            continue
        o_n, k_n, conc_n = null_tab[sig]
        ratio = (conc_r + smooth_floor) / (conc_n + smooth_floor)
        ranked.append((ratio, sig, (o_r, k_r, conc_r), (o_n, k_n, conc_n)))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:topK]


def bootstrap_ci_median_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if x.size == 0 or y.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        diffs.append(float(np.median(xb) - np.median(yb)))
    diffs = np.sort(np.array(diffs))
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return (float(np.median(x) - np.median(y)), lo, hi)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a.intersection(b)) / max(1, len(a.union(b)))


def summarize_candidates(cands: List[Candidate]) -> Dict[str, float]:
    if len(cands) == 0:
        return {"n": 0.0, "score_mean": float("nan"), "score_median": float("nan"), "score_max": float("nan"),
                "leak_mean": float("nan"), "leak_median": float("nan"), "leak_min": float("nan"),
                "ent_mean": float("nan"), "dom_median": float("nan")}
    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leakage for c in cands], dtype=float)
    ents = np.array([(c.ent_i + c.ent_j) * 0.5 for c in cands], dtype=float)
    doms = np.array([c.dom_count for c in cands], dtype=float)
    return {
        "n": float(len(cands)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "leak_mean": float(np.mean(leaks)),
        "leak_median": float(np.median(leaks)),
        "leak_min": float(np.min(leaks)),
        "ent_mean": float(np.mean(ents)),
        "dom_median": float(np.median(doms)),
    }


def print_summary_block(name: str, stats: Dict[str, float]) -> None:
    print(f"Model: {name}")
    print(
        f"  candidates={int(stats['n'])} | "
        f"score(mean/median/max)={stats['score_mean']:.3f}/{stats['score_median']:.3f}/{stats['score_max']:.3f} | "
        f"leakage(mean/median/min)={stats['leak_mean']:.3f}/{stats['leak_median']:.3f}/{stats['leak_min']:.3f} | "
        f"dom_count(median)={stats['dom_median']:.1f} | ent(mean)={stats['ent_mean']:.3f}"
    )


def format_sig(sig: SignatureKey) -> str:
    a, b, c, d = sig.as_tuple()
    return f"({a}, {b}, {c}, {d})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0_7 convergence: replicable signature families (v5).")
    p.add_argument("--n_qubits", type=int, default=3)
    p.add_argument("--num_terms", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5000)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--seed_stride", type=int, default=100000)

    p.add_argument("--eps", type=float, default=0.050)
    p.add_argument("--dominant_threshold", type=float, default=0.25)

    p.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--ent_step", type=float, default=0.25)
    p.add_argument("--leak_step", type=float, default=0.10)

    p.add_argument("--stable_frac", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--min_overall", type=int, default=20)
    p.add_argument("--topK", type=int, default=25)

    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--bootstrap_seed", type=int, default=123)

    p.add_argument("--output", type=str, default="v0_7_convergence_families_v5_output.txt")
    return p.parse_args()


def batch_report(
    batch_id: int,
    real: List[Candidate],
    null: List[Candidate],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, object]:
    print("")
    print("==============================================")
    print(f"Batch {batch_id+1}/{args.batches}")
    print("==============================================")

    print_summary_block("REAL", summarize_candidates(real))
    print_summary_block("NULL_HAAR_BASIS", summarize_candidates(null))

    real_scores = np.array([c.score for c in real], dtype=float)
    null_scores = np.array([c.score for c in null], dtype=float)
    m_real = stable_mask(real_scores, args.stable_frac)
    m_null = stable_mask(null_scores, args.stable_frac)

    real_sigs = [c.sig for c in real]
    null_sigs = [c.sig for c in null]

    real_tab = enrichment_table(real_sigs, m_real, alpha=args.alpha, min_overall=args.min_overall)
    null_tab = enrichment_table(null_sigs, m_null, alpha=args.alpha, min_overall=args.min_overall)

    ranked = top_families_by_ratio(real_tab, null_tab, args.topK)

    print("")
    print("Top signature families by REAL/NULL concentration ratio")
    print("Format: sig=(dom, ent_bin_i, ent_bin_j, leak_bin) | overall_R | stable_R | conc_R | overall_N | stable_N | conc_N | ratio")
    for ratio, sig, (o_r, k_r, conc_r), (o_n, k_n, conc_n) in ranked[: min(args.topK, 10)]:
        print(
            f"  {format_sig(sig)} | {o_r:5d} | {k_r:5d} | {conc_r:6.3f} | "
            f"{o_n:5d} | {k_n:5d} | {conc_n:6.3f} | {ratio:7.3f}"
        )
    print(f"  ... (topK={args.topK} computed; showing up to 10)")

    def stable_values(cands: List[Candidate], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ent = np.array([(c.ent_i + c.ent_j) * 0.5 for c in cands], dtype=float)
        leak = np.array([c.leakage for c in cands], dtype=float)
        return ent[mask], leak[mask]

    ent_r, leak_r = stable_values(real, m_real)
    ent_n, leak_n = stable_values(null, m_null)

    md_ent, lo_ent, hi_ent = bootstrap_ci_median_diff(ent_r, ent_n, args.bootstrap, rng)
    md_leak, lo_leak, hi_leak = bootstrap_ci_median_diff(leak_r, leak_n, args.bootstrap, rng)

    print("")
    print("Distribution-level effects (stable candidates): medians with bootstrap CI for REAL-NULL")
    print(f"  median_entropy_diff = {md_ent:.4f}  CI95=[{lo_ent:.4f}, {hi_ent:.4f}]")
    print(f"  median_leakage_diff = {md_leak:.4f}  CI95=[{lo_leak:.4f}, {hi_leak:.4f}]")

    top_set = set([sig for _, sig, _, _ in ranked])
    return {"ranked": ranked, "top_set": top_set, "md_ent": md_ent, "md_leak": md_leak}


def main() -> None:
    args = parse_args()
    out_path = args.output

    tee = Tee(out_path)
    sys.stdout = tee  # type: ignore

    try:
        print("=== v0_7 Convergence: Replicable Signature Families (v5) ===")
        print(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.num_terms}")
        print(f"Batches: {args.batches} x seeds={args.seeds} | seed_stride={args.seed_stride}")
        print(f"Neighbor eps={args.eps:.3f} | dominant_threshold={args.dominant_threshold:.3f}")
        print(f"Times={args.times} | ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
        print(f"stable_frac={args.stable_frac:.3f} | min_overall={args.min_overall} | alpha={args.alpha:.2f} | topK={args.topK}")
        print(f"bootstrap={args.bootstrap} | output={os.path.abspath(out_path)}")

        rng = np.random.default_rng(args.bootstrap_seed)
        batch_results = []
        t0 = time.time()

        for b in range(args.batches):
            seed_offset = b * args.seed_stride
            start = time.time()
            real, null = scan_batch(
                n_qubits=args.n_qubits,
                num_terms=args.num_terms,
                seeds=args.seeds,
                seed_offset=seed_offset,
                eps=args.eps,
                dominant_threshold=args.dominant_threshold,
                times=args.times,
                ent_step=args.ent_step,
                leak_step=args.leak_step,
                rng_seed=args.bootstrap_seed + 1000 * b,
            )
            elapsed = time.time() - start
            print(f"\nBatch {b+1} scan complete: REAL candidates={len(real)}, NULL candidates={len(null)} | elapsed={elapsed:.1f}s")
            batch_results.append(batch_report(b, real, null, args, rng))

        print("\n==============================================")
        print("Cross-batch convergence summary")
        print("==============================================")

        top_sets = [br["top_set"] for br in batch_results]  # type: ignore
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                jac = jaccard(top_sets[i], top_sets[j])
                inter = len(top_sets[i].intersection(top_sets[j]))
                print(f"Top-{args.topK} overlap batch {i+1} vs {j+1}: intersection={inter} | jaccard={jac:.3f}")

        freq: Dict[SignatureKey, int] = {}
        for s in top_sets:
            for sig in s:
                freq[sig] = freq.get(sig, 0) + 1

        common = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0].as_tuple()))
        print("")
        print("Most repeatable families across batches (frequency in Top-K):")
        for sig, c in common[: min(15, len(common))]:
            print(f"  {format_sig(sig)} : {c}/{args.batches}")

        md_ent = np.array([br["md_ent"] for br in batch_results], dtype=float)  # type: ignore
        md_leak = np.array([br["md_leak"] for br in batch_results], dtype=float)  # type: ignore

        print("")
        print("Distribution-level effects across batches:")
        print(f"  median_entropy_diff (REAL-NULL): mean={np.mean(md_ent):.4f}  std={np.std(md_ent):.4f}  per_batch={md_ent}")
        print(f"  median_leakage_diff (REAL-NULL): mean={np.mean(md_leak):.4f}  std={np.std(md_leak):.4f}  per_batch={md_leak}")

        total_elapsed = time.time() - t0
        print(f"\n=== End of v5 convergence run ===  total_elapsed={total_elapsed:.1f}s")

    finally:
        try:
            sys.stdout = tee._stdout  # type: ignore
        except Exception:
            pass
        tee.close()


if __name__ == "__main__":
    main()
