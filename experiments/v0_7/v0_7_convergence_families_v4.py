#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_convergence_families_v4.py

Purpose (matched to the "fix replication" advice):
- Run 3 independent batches (e.g., 3 × 5000 seeds with seed offsets)
- Measure:
  (1) Stability of top families: overlap(Top-K) across batches
  (2) Distribution-level effect size: REAL vs NULL entropy-bin (median/mean + bootstrap CI)
  (3) Cross-model enrichment with smoothing + minimum-count filtering (readable ratios)

Key changes vs. "empty eligible families" outputs:
- Coarser binning (ent_step, leak_step) to increase counts per label
- Larger stable set (stable_frac) to increase power
- Relaxed minimum-count filter (min_overall, min_stable)
- Smoothing (pseudocount) to avoid infinite / zero ratios

This script is self-contained and runs locally (NumPy only).
No special unicode characters are used in output (Windows cp1252 safe).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# IO helper: write console + txt
# -----------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


# -----------------------------
# Pauli utilities
# -----------------------------
PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def kron_n(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


class PauliCache:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self._cache: Dict[str, np.ndarray] = {}

    def mat(self, s: str) -> np.ndarray:
        if s in self._cache:
            return self._cache[s]
        mats = [PAULI_1Q[ch] for ch in s]
        M = kron_n(mats)
        self._cache[s] = M
        return M


def random_pauli_string(rng: np.random.Generator, n: int) -> str:
    alphabet = ["I", "X", "Y", "Z"]
    while True:
        s = "".join(rng.choice(alphabet) for _ in range(n))
        if s != ("I" * n):
            return s


def pauli_exp(P: np.ndarray, theta: float) -> np.ndarray:
    d = P.shape[0]
    I = np.eye(d, dtype=np.complex128)
    return (math.cos(theta) * I) + (-1j * math.sin(theta)) * P


# -----------------------------
# Candidates and signatures
# -----------------------------
@dataclass(frozen=True)
class SigKey:
    dom: int
    ent_bin: int
    leak_bin: int

    def as_tuple(self):
        return (self.dom, self.ent_bin, self.leak_bin)


@dataclass
class Candidate:
    sig: SigKey
    entropy_mean: float
    leakage_mean: float
    score: float


# -----------------------------
# Core metrics
# -----------------------------
def shannon_entropy(p: np.ndarray) -> float:
    p = np.clip(p.real, 0.0, 1.0)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def dominant_indices(p: np.ndarray, threshold: float) -> List[int]:
    idx = np.where(p >= threshold)[0].tolist()
    if not idx:
        idx = [int(np.argmax(p))]
    return idx


def build_random_hamiltonian(
    rng: np.random.Generator,
    cache: PauliCache,
    n_qubits: int,
    num_terms: int,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    terms = []
    d = 2**n_qubits
    H = np.zeros((d, d), dtype=np.complex128)
    for _ in range(num_terms):
        s = random_pauli_string(rng, n_qubits)
        c = float(rng.uniform(-1.0, 1.0))
        H = H + c * cache.mat(s)
        terms.append((s, c))
    return H, terms


def haar_random_unitary(rng: np.random.Generator, d: int) -> np.ndarray:
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph
    return Q


def spectrum_matched_null(H: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    evals, _ = np.linalg.eigh(H)
    d = H.shape[0]
    U = haar_random_unitary(rng, d)
    return U @ np.diag(evals) @ U.conj().T


def neighbor_near_degenerate_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    pairs = []
    for k in range(len(evals) - 1):
        if abs(evals[k + 1] - evals[k]) < eps:
            pairs.append((k, k + 1))
    return pairs


def trotter_evolve_state(
    psi0: np.ndarray,
    terms: List[Tuple[np.ndarray, float]],
    t: float,
    steps: int,
) -> np.ndarray:
    psi = psi0
    dt = t / steps
    for _ in range(steps):
        for P, c in terms:
            psi = pauli_exp(P, c * dt) @ psi
    return psi


def leakage_proxy_real(
    cache: PauliCache,
    terms_sc: List[Tuple[str, float]],
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    dom_threshold: float,
    times: List[float],
    trotter_steps: int,
) -> Tuple[float, float, int]:
    vi = psi_i.reshape(-1, 1)
    vj = psi_j.reshape(-1, 1)
    Psub = (vi @ vi.conj().T) + (vj @ vj.conj().T)

    pi = np.abs(psi_i) ** 2
    pj = np.abs(psi_j) ** 2
    idx = sorted(set(dominant_indices(pi, dom_threshold) + dominant_indices(pj, dom_threshold)))
    d = psi_i.shape[0]
    psi0 = np.zeros((d,), dtype=np.complex128)
    for k in idx:
        psi0[k] = 1.0
    psi0 = psi0 / np.linalg.norm(psi0)

    H_terms = [(cache.mat(s), c) for (s, c) in terms_sc]

    leaks = []
    for t in times:
        psi_t = trotter_evolve_state(psi0, H_terms, t, trotter_steps)
        in_sub = float(np.real(psi_t.conj().T @ (Psub @ psi_t)))
        in_sub = min(max(in_sub, 0.0), 1.0)
        leaks.append(1.0 - in_sub)

    leak_mean = float(np.mean(leaks))
    ent_mean = 0.5 * (shannon_entropy(pi) + shannon_entropy(pj))
    dom = len(idx)
    return ent_mean, leak_mean, dom


def leakage_proxy_null(
    evals: np.ndarray,
    evecs: np.ndarray,
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    dom_threshold: float,
    times: List[float],
) -> Tuple[float, float, int]:
    vi = psi_i.reshape(-1, 1)
    vj = psi_j.reshape(-1, 1)
    Psub = (vi @ vi.conj().T) + (vj @ vj.conj().T)

    pi = np.abs(psi_i) ** 2
    pj = np.abs(psi_j) ** 2
    idx = sorted(set(dominant_indices(pi, dom_threshold) + dominant_indices(pj, dom_threshold)))
    d = psi_i.shape[0]
    psi0 = np.zeros((d,), dtype=np.complex128)
    for k in idx:
        psi0[k] = 1.0
    psi0 = psi0 / np.linalg.norm(psi0)

    V = evecs
    E = evals

    def exact_evolve(psi: np.ndarray, t: float) -> np.ndarray:
        phases = np.exp(-1j * E * t)
        return V @ (phases * (V.conj().T @ psi))

    leaks = []
    for t in times:
        psi_t = exact_evolve(psi0, t)
        in_sub = float(np.real(psi_t.conj().T @ (Psub @ psi_t)))
        in_sub = min(max(in_sub, 0.0), 1.0)
        leaks.append(1.0 - in_sub)

    leak_mean = float(np.mean(leaks))
    ent_mean = 0.5 * (shannon_entropy(pi) + shannon_entropy(pj))
    dom = len(idx)
    return ent_mean, leak_mean, dom


def bin_int(x: float, step: float) -> int:
    return int(round(x / step)) if step > 0 else 0


# -----------------------------
# Enrichment with smoothing
# -----------------------------
def enrichment_table(
    sigs: List[SigKey],
    scores: np.ndarray,
    stable_frac: float,
    min_overall: int,
    min_stable: int,
    alpha: float,
    topK: int,
) -> List[Tuple[SigKey, int, int, float]]:
    n = len(sigs)
    if n == 0:
        return []
    k = max(1, int(math.ceil(stable_frac * n)))
    idx_sorted = np.argsort(-scores)
    stable_set = set(idx_sorted[:k].tolist())

    overall: Dict[SigKey, int] = {}
    stable: Dict[SigKey, int] = {}
    for i, s in enumerate(sigs):
        overall[s] = overall.get(s, 0) + 1
        if i in stable_set:
            stable[s] = stable.get(s, 0) + 1

    U = len(overall)
    rows = []
    for s, oc in overall.items():
        sc = stable.get(s, 0)
        if oc < min_overall or sc < min_stable:
            continue
        p_stable = (sc + alpha) / (k + alpha * U)
        p_overall = (oc + alpha) / (n + alpha * U)
        enr = p_stable / p_overall if p_overall > 0 else float("nan")
        rows.append((s, oc, sc, float(enr)))

    rows.sort(key=lambda r: (r[3], r[2], r[1]), reverse=True)
    return rows[:topK]


def bootstrap_ci_median_diff(x_real: np.ndarray, x_null: np.ndarray, n_boot: int, rng: np.random.Generator):
    obs = float(np.median(x_real) - np.median(x_null))
    diffs = []
    nR, nN = len(x_real), len(x_null)
    for _ in range(n_boot):
        bR = x_real[rng.integers(0, nR, size=nR)]
        bN = x_null[rng.integers(0, nN, size=nN)]
        diffs.append(float(np.median(bR) - np.median(bN)))
    diffs = np.array(diffs, dtype=float)
    return obs, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# -----------------------------
# Batch scanning
# -----------------------------
def scan_batch(
    batch_seed: int,
    n_seeds: int,
    n_qubits: int,
    num_terms: int,
    eps: float,
    dom_threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    trotter_steps: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    rrng = np.random.default_rng(batch_seed)
    cache = PauliCache(n_qubits)

    real_cands: List[Candidate] = []
    null_cands: List[Candidate] = []

    for s in range(n_seeds):
        rng = np.random.default_rng(batch_seed + s * 1009)
        H, terms_sc = build_random_hamiltonian(rng, cache, n_qubits, num_terms)

        evals, evecs = np.linalg.eigh(H)
        pairs = neighbor_near_degenerate_pairs(evals, eps)
        for (i, j) in pairs:
            ent, leak, dom = leakage_proxy_real(cache, terms_sc, evecs[:, i], evecs[:, j],
                                               dom_threshold, times, trotter_steps)
            score = (1.0 - leak) / (1.0 + dom)
            sig = SigKey(dom=dom, ent_bin=bin_int(ent, ent_step), leak_bin=bin_int(leak, leak_step))
            real_cands.append(Candidate(sig=sig, entropy_mean=ent, leakage_mean=leak, score=score))

        Hn = spectrum_matched_null(H, rng)
        evals_n, evecs_n = np.linalg.eigh(Hn)
        pairs_n = neighbor_near_degenerate_pairs(evals_n, eps)
        for (i, j) in pairs_n:
            ent, leak, dom = leakage_proxy_null(evals_n, evecs_n, evecs_n[:, i], evecs_n[:, j],
                                               dom_threshold, times)
            score = (1.0 - leak) / (1.0 + dom)
            sig = SigKey(dom=dom, ent_bin=bin_int(ent, ent_step), leak_bin=bin_int(leak, leak_step))
            null_cands.append(Candidate(sig=sig, entropy_mean=ent, leakage_mean=leak, score=score))

    return real_cands, null_cands


def topK_sigs(rows: List[Tuple[SigKey, int, int, float]]) -> List[SigKey]:
    return [r[0] for r in rows]


def jaccard(a: List[SigKey], b: List[SigKey]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def print_rows(title: str, rows: List[Tuple[SigKey, int, int, float]]):
    print(title)
    print("Format: sig=(dom, ent_bin, leak_bin) | overall | stable | enrichment")
    if not rows:
        print("  (no eligible families under current filters)")
        return
    for s, oc, sc, enr in rows:
        print(f"  {s.as_tuple()} | {oc:5d} | {sc:5d} | {enr:8.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="v0_7_convergence_families_v4_output.txt")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--seed_stride", type=int, default=1000000)

    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--num_terms", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--dom_threshold", type=float, default=0.25)

    ap.add_argument("--ent_step", type=float, default=0.25)
    ap.add_argument("--leak_step", type=float, default=0.10)
    ap.add_argument("--stable_frac", type=float, default=0.05)
    ap.add_argument("--min_overall", type=int, default=10)
    ap.add_argument("--min_stable", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--topK", type=int, default=25)

    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap_seed", type=int, default=777)

    ap.add_argument("--times", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--trotter_steps", type=int, default=8)

    args = ap.parse_args()

    times = [float(x.strip()) for x in args.times.split(",") if x.strip()]

    f = open(args.out, "w", encoding="utf-8", newline="\n")
    sys.stdout = Tee(sys.__stdout__, f)

    print("=== v0_7 Convergence (Families v4) ===")
    print(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.num_terms}")
    print(f"Batches: {args.batches} | seeds_per_batch={args.seeds_per_batch} | seed0={args.seed0} | stride={args.seed_stride}")
    print(f"Near-degenerate neighbor eps={args.eps:.3f}")
    print(f"Signature binning: ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f} | dom_threshold={args.dom_threshold:.3f}")
    print(f"Evidence: stable_frac={args.stable_frac:.3f} | min_overall={args.min_overall} | min_stable={args.min_stable} | alpha={args.alpha:.2f} | topK={args.topK}")
    print(f"Leakage proxy: times={times} | trotter_steps={args.trotter_steps}")
    print(f"Bootstrap: n={args.bootstrap} | seed={args.bootstrap_seed}")
    print(f"Output: {args.out}")
    print("")

    rng_ci = np.random.default_rng(args.bootstrap_seed)
    batch_tops_real = []
    batch_tops_null = []

    for b in range(args.batches):
        batch_seed = args.seed0 + b * args.seed_stride
        t0 = time.time()
        real_cands, null_cands = scan_batch(
            batch_seed=batch_seed,
            n_seeds=args.seeds_per_batch,
            n_qubits=args.n_qubits,
            num_terms=args.num_terms,
            eps=args.eps,
            dom_threshold=args.dom_threshold,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=times,
            trotter_steps=args.trotter_steps,
        )
        elapsed = time.time() - t0

        def summarize(label: str, cands: List[Candidate]):
            if not cands:
                print(f"Batch {b} | {label}: candidates=0")
                return
            scores = np.array([c.score for c in cands], dtype=float)
            leaks = np.array([c.leakage_mean for c in cands], dtype=float)
            ents = np.array([c.entropy_mean for c in cands], dtype=float)
            doms = np.array([c.sig.dom for c in cands], dtype=int)
            print(f"Batch {b} | {label}: candidates={len(cands)} | score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f} | leakage(mean/median/min)={leaks.mean():.3f}/{np.median(leaks):.3f}/{leaks.min():.3f} | entropy(mean/median)={ents.mean():.3f}/{np.median(ents):.3f} | dom(median)={int(np.median(doms))}")

        print(f"--- Batch {b} (seed={batch_seed}) | elapsed={elapsed:.1f}s ---")
        summarize("REAL", real_cands)
        summarize("NULL_HAAR_BASIS", null_cands)
        print("")

        rowsR = enrichment_table(
            sigs=[c.sig for c in real_cands],
            scores=np.array([c.score for c in real_cands], dtype=float),
            stable_frac=args.stable_frac,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            topK=args.topK,
        )
        rowsN = enrichment_table(
            sigs=[c.sig for c in null_cands],
            scores=np.array([c.score for c in null_cands], dtype=float),
            stable_frac=args.stable_frac,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            topK=args.topK,
        )

        print_rows("REAL: Top signature families (stable-set vs overall)", rowsR)
        print("")
        print_rows("NULL: Top signature families (stable-set vs overall)", rowsN)
        print("")

        if real_cands and null_cands:
            entsR = np.array([c.entropy_mean for c in real_cands], dtype=float)
            entsN = np.array([c.entropy_mean for c in null_cands], dtype=float)
            md, lo, hi = bootstrap_ci_median_diff(entsR, entsN, args.bootstrap, rng_ci)
            print(f"Effect size (median entropy REAL - NULL): {md:.3f} | 95% CI [{lo:.3f}, {hi:.3f}]")

            leaksR = np.array([c.leakage_mean for c in real_cands], dtype=float)
            leaksN = np.array([c.leakage_mean for c in null_cands], dtype=float)
            md, lo, hi = bootstrap_ci_median_diff(leaksR, leaksN, args.bootstrap, rng_ci)
            print(f"Effect size (median leakage REAL - NULL): {md:.3f} | 95% CI [{lo:.3f}, {hi:.3f}]")
            print("")

        batch_tops_real.append(topK_sigs(rowsR))
        batch_tops_null.append(topK_sigs(rowsN))

    print("=== Across-batch replication diagnostics ===")
    if args.batches >= 2:
        print("Top-K overlap across batches (REAL):")
        for i in range(args.batches):
            for j in range(i + 1, args.batches):
                ov = len(set(batch_tops_real[i]) & set(batch_tops_real[j]))
                jac = jaccard(batch_tops_real[i], batch_tops_real[j])
                print(f"  batches {i} vs {j}: overlap={ov} | jaccard={jac:.3f}")
        print("")
        print("Top-K overlap across batches (NULL):")
        for i in range(args.batches):
            for j in range(i + 1, args.batches):
                ov = len(set(batch_tops_null[i]) & set(batch_tops_null[j]))
                jac = jaccard(batch_tops_null[i], batch_tops_null[j])
                print(f"  batches {i} vs {j}: overlap={ov} | jaccard={jac:.3f}")
        print("")
    else:
        print("Not enough batches to compute overlaps.")
        print("")

    print("=== Notes ===")
    print("1) If you still get 'no eligible families', coarsen bins further (ent_step=0.35-0.50, leak_step=0.15-0.25) or raise stable_frac to 0.10.")
    print("2) If effect sizes are stable but overlaps remain low, the effect is population-level but not concentrated into a small set of coarse families under this key definition.")
    print("3) If overlaps become stable, you can tighten bins again to localize the families more precisely.")
    print("")
    print("=== End ===")

    sys.stdout = sys.__stdout__
    try:
        f.flush()
        f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
