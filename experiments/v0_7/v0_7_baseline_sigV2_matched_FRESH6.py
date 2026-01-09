#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7 SigV2 (Matched, FAIR) — Baseline / Null-Model Calibration
-------------------------------------------------------------
Goal
  Calibrate what "chance structure" looks like under the *same* v0_6.1 SigV2 pipeline,
  by comparing REAL candidates against a spectrum-matched NULL model.

Key additions vs FRESH5 (without changing the underlying discovery logic):
  1) Parameter matching is explicit (eps, dominance threshold, binning, time grid).
  2) NULL model is spectrum-matched (same eigenvalues) and uses Haar-random eigenbasis.
  3) Cross-model comparison table: enrichment_REAL, enrichment_NULL, ratio, and ratio CI.
     This addresses the "HAAR_BASIS signal" concern: you only claim "non-chance" if
     REAL consistently exceeds NULL under matched settings.

Outputs
  - Writes a complete console transcript to a UTF-8 text file.
  - Keeps output ASCII-clean (no special arrow characters) for Windows consoles.

No graphics. No external dependencies beyond numpy.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------
# Utilities: logging / tee
# ----------------------------

class Tee:
    """Mirror stdout to both console and a UTF-8 file, without encoding surprises."""
    def __init__(self, filename: str):
        self.filename = filename
        self._file = open(filename, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout

    def write(self, obj):
        s = str(obj)
        self._stdout.write(s)
        self._file.write(s)

    def flush(self):
        # guard against closed file during interpreter shutdown
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


# ----------------------------
# Quantum building blocks
# ----------------------------

_PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def kron_n(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for a in ops[1:]:
        out = np.kron(out, a)
    return out

def build_random_pauli_sum_hamiltonian(n_qubits: int, num_terms: int, rng: np.random.Generator) -> np.ndarray:
    """Random Pauli-sum Hamiltonian H = sum_k c_k P_k with real coefficients."""
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)
    paulis = ["I", "X", "Y", "Z"]
    for _ in range(num_terms):
        word = [paulis[int(rng.integers(0, 4))] for _ in range(n_qubits)]
        # avoid all-identity term (no effect)
        if all(p == "I" for p in word):
            word[int(rng.integers(0, n_qubits))] = "Z"
        P = kron_n([_PAULI_MATS[p] for p in word])
        c = float(rng.uniform(-1.0, 1.0))
        H = H + c * P
    # Ensure Hermitian
    H = 0.5 * (H + H.conj().T)
    return H

def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-random unitary via QR on complex Gaussian matrix.
    """
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    # Normalize diagonal of R to have unit magnitude
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph
    return Q

def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues (ascending) and eigenvectors (columns)."""
    evals, evecs = np.linalg.eigh(H)
    return evals.real, evecs

def find_near_degenerate_neighbors(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    """Neighbor scan for near-degenerate pairs in sorted eigenvalues."""
    pairs = []
    for i in range(len(evals) - 1):
        if abs(float(evals[i+1] - evals[i])) < eps:
            pairs.append((i, i+1))
    return pairs

def state_probs(vec: np.ndarray) -> np.ndarray:
    p = np.abs(vec) ** 2
    s = float(p.sum())
    if s <= 0:
        return p
    return p / s

def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def dominant_basis_count(p: np.ndarray, thr: float) -> int:
    return int(np.sum(p >= thr))

def pick_subspace_basis_from_pair(evecs: np.ndarray, i: int, j: int, topk: int = 2) -> List[int]:
    """
    Define the candidate subspace in the computational basis using the *sum projector*
    of the near-degenerate pair. We then select the top-k basis indices by probability mass.
    """
    vi = evecs[:, i]
    vj = evecs[:, j]
    pi = state_probs(vi)
    pj = state_probs(vj)
    proj = pi + pj
    idx = np.argsort(-proj)[:topk]
    return [int(x) for x in idx]

def pauli_exp_gate(P: np.ndarray, theta: float) -> np.ndarray:
    """
    exp(-i * theta * P) for Hermitian Pauli product P with P^2 = I.
    exp(-i theta P) = cos(theta) I - i sin(theta) P
    """
    d = P.shape[0]
    return math.cos(theta) * np.eye(d, dtype=complex) - 1j * math.sin(theta) * P

def trotter_evolution_operator(H_terms: List[Tuple[np.ndarray, float]], t: float, steps: int) -> np.ndarray:
    """First-order Trotter: U(t) ≈ [Π_k exp(-i c_k P_k Δt)]^steps"""
    d = H_terms[0][0].shape[0]
    dt = t / float(steps)
    U_step = np.eye(d, dtype=complex)
    for P, c in H_terms:
        U_step = pauli_exp_gate(P, c * dt) @ U_step
    U = np.eye(d, dtype=complex)
    for _ in range(steps):
        U = U_step @ U
    return U

def extract_terms_from_H(n_qubits: int, num_terms: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, float]]:
    """Return a list of (PauliProductMatrix, coefficient) used to build H."""
    paulis = ["I", "X", "Y", "Z"]
    terms = []
    for _ in range(num_terms):
        word = [paulis[int(rng.integers(0, 4))] for _ in range(n_qubits)]
        if all(p == "I" for p in word):
            word[int(rng.integers(0, n_qubits))] = "Z"
        P = kron_n([_PAULI_MATS[p] for p in word])
        c = float(rng.uniform(-1.0, 1.0))
        terms.append((P, c))
    return terms

def build_H_from_terms(terms: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    d = terms[0][0].shape[0]
    H = np.zeros((d, d), dtype=complex)
    for P, c in terms:
        H = H + c * P
    H = 0.5 * (H + H.conj().T)
    return H

def leakage_proxy(subspace_basis: List[int], H_terms: List[Tuple[np.ndarray, float]], times: Sequence[float], steps: int) -> float:
    """
    Lightweight stability proxy:
      - initialize uniform superposition over selected basis states
      - evolve under Trotterized U(t)
      - measure probability remaining in the same basis subset
    Return leakage averaged over times.
    """
    d = H_terms[0][0].shape[0]
    psi0 = np.zeros((d,), dtype=complex)
    for b in subspace_basis:
        psi0[b] += 1.0
    psi0 = psi0 / np.linalg.norm(psi0)

    basis_set = set(subspace_basis)
    leak_vals = []
    for t in times:
        U = trotter_evolution_operator(H_terms, t=float(t), steps=int(steps))
        psi_t = U @ psi0
        p = np.abs(psi_t) ** 2
        stay = float(sum(p[k] for k in basis_set))
        leak = 1.0 - stay
        leak_vals.append(leak)
    return float(np.mean(leak_vals))

# ----------------------------
# Signature + scoring
# ----------------------------

def signature_key(dom_count: int, ent_i: float, ent_j: float, leak: float,
                  ent_step: float, leak_step: float) -> Tuple[int, int, int, int]:
    """Integer signature for hashing/aggregation."""
    ebi = int(math.floor(ent_i / ent_step + 1e-12))
    ebj = int(math.floor(ent_j / ent_step + 1e-12))
    lb  = int(math.floor(leak  / leak_step + 1e-12))
    return (int(dom_count), ebi, ebj, lb)

def interestingness_score(dom_count: int, ent_i: float, ent_j: float, leak: float) -> float:
    """
    Conservative score: stable + compact + structured.
    Range ~ [0, 1] after clipping.
      - (1 - leak) rewards stability
      - dom_count penalizes diffuse basis support
      - entropy penalizes diffuse eigenvectors
    """
    s_stab = max(0.0, 1.0 - leak)
    s_comp = 1.0 / max(1.0, float(dom_count))     # smaller dom_count => larger score
    s_ent  = 1.0 / (1.0 + 0.5 * (ent_i + ent_j))  # smaller entropy => larger score
    score = s_stab * s_comp * s_ent
    return float(max(0.0, min(1.0, score)))

@dataclass(frozen=True)
class Candidate:
    sig: Tuple[int, int, int, int]
    score: float
    leak: float

# ----------------------------
# Enrichment + bootstrap
# ----------------------------

def stable_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask for top-frac highest scores."""
    n = scores.size
    k = max(1, int(math.ceil(frac * n)))
    thr = float(np.partition(scores, n - k)[n - k])
    return scores >= thr

def enrichment_rows(sig_arr: np.ndarray, mask: np.ndarray) -> Tuple[List[Tuple[Tuple[int,int,int,int], int, int, float]], Dict, Dict]:
    """
    Return rows: (sig, overall_count, stable_count, enrichment)
    enrichment = (stable_frac_sig / overall_frac_sig), with small eps safety.
    """
    overall: Dict[Tuple[int,int,int,int], int] = {}
    stable: Dict[Tuple[int,int,int,int], int] = {}
    for s in sig_arr:
        s = tuple(int(x) for x in s)  # canonicalize
        overall[s] = overall.get(s, 0) + 1
    for s, m in zip(sig_arr, mask):
        if not m:
            continue
        s = tuple(int(x) for x in s)
        stable[s] = stable.get(s, 0) + 1

    n_total = float(len(sig_arr))
    n_stable = float(np.sum(mask))
    rows = []
    for s, o in overall.items():
        st = stable.get(s, 0)
        p_overall = o / n_total
        p_stable = st / max(1.0, n_stable)
        enr = p_stable / max(1e-12, p_overall)
        rows.append((s, o, st, float(enr)))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows, overall, stable

def bootstrap_ratio_ci(
    target_sigs: List[Tuple[int,int,int,int]],
    real_sig_arr: np.ndarray,
    real_scores: np.ndarray,
    null_sig_arr: np.ndarray,
    null_scores: np.ndarray,
    frac: float,
    B: int,
    rng: np.random.Generator
) -> Dict[Tuple[int,int,int,int], Tuple[float, float]]:
    """
    Bootstrap CI for enrichment ratio (REAL/NULL) for a list of signatures.
    Returns dict: sig -> (lo, hi) percentile CI (2.5%, 97.5%).
    """
    # pre-canonicalize to tuple ints for safety
    real_sig_arr = np.array([tuple(int(x) for x in s) for s in real_sig_arr], dtype=object)
    null_sig_arr = np.array([tuple(int(x) for x in s) for s in null_sig_arr], dtype=object)

    target_sigs = [tuple(int(x) for x in s) for s in target_sigs]
    ratios: Dict[Tuple[int,int,int,int], List[float]] = {s: [] for s in target_sigs}

    nR = len(real_scores)
    nN = len(null_scores)
    idxR = np.arange(nR)
    idxN = np.arange(nN)

    for _ in range(int(B)):
        sampR = rng.choice(idxR, size=nR, replace=True)
        sampN = rng.choice(idxN, size=nN, replace=True)

        r_scores = real_scores[sampR]
        n_scores = null_scores[sampN]
        r_mask = stable_mask(r_scores, frac)
        n_mask = stable_mask(n_scores, frac)

        r_rows, r_over, r_st = enrichment_rows(real_sig_arr[sampR], r_mask)
        n_rows, n_over, n_st = enrichment_rows(null_sig_arr[sampN], n_mask)

        # Build enrichment dicts for quick lookup
        def to_enr_map(over, st, n_total, n_stable):
            out = {}
            for s, o in over.items():
                p_over = o / n_total
                p_st = st.get(s, 0) / max(1.0, n_stable)
                out[s] = float(p_st / max(1e-12, p_over))
            return out

        r_enr = to_enr_map(r_over, r_st, float(len(sampR)), float(np.sum(r_mask)))
        n_enr = to_enr_map(n_over, n_st, float(len(sampN)), float(np.sum(n_mask)))

        for s in target_sigs:
            rr = r_enr.get(s, 0.0)
            nn = n_enr.get(s, 0.0)
            ratio = rr / max(1e-12, nn)
            ratios[s].append(float(ratio))

    ci = {}
    for s, vals in ratios.items():
        if len(vals) == 0:
            ci[s] = (float("nan"), float("nan"))
            continue
        lo = float(np.percentile(vals, 2.5))
        hi = float(np.percentile(vals, 97.5))
        ci[s] = (lo, hi)
    return ci

# ----------------------------
# Data generation (REAL / NULL)
# ----------------------------

def generate_candidates(
    model: str,
    n_qubits: int,
    num_terms: int,
    seeds: int,
    eps: float,
    dom_thr: float,
    ent_step: float,
    leak_step: float,
    times: Sequence[float],
    trotter_steps: int,
    rng: np.random.Generator,
) -> List[Candidate]:
    """
    Generate candidates for a given model.
    REAL:
      - random Pauli-sum Hamiltonian
      - eigenpairs from diagonalization
    NULL_HAAR_BASIS (spectrum-matched):
      - build REAL Hamiltonian terms to get spectrum
      - replace eigenbasis by Haar random unitary (keeps eigenvalues)
      - proceed identically (neighbor-degenerate detection, structural extraction, leakage proxy)
    """
    d = 2 ** n_qubits
    out: List[Candidate] = []

    for s in range(seeds):
        # seed-local RNG to make runs stable across models when desired
        rng_s = np.random.default_rng(int(rng.integers(0, 2**31-1)))

        # Build terms first (so NULL can share same spectrum)
        terms = extract_terms_from_H(n_qubits, num_terms, rng_s)
        H = build_H_from_terms(terms)
        evals, evecs_real = diagonalize(H)

        if model == "REAL":
            evecs_use = evecs_real
        elif model == "NULL_HAAR_BASIS":
            U = haar_unitary(d, rng_s)
            # spectrum-matched Hamiltonian: H_null = U diag(evals) U†
            # evecs are columns of U
            evecs_use = U
            # evals unchanged
        else:
            raise ValueError(f"Unknown model: {model}")

        pairs = find_near_degenerate_neighbors(evals, eps=eps)
        if not pairs:
            continue

        for (i, j) in pairs:
            vi = evecs_use[:, i]
            vj = evecs_use[:, j]
            pi = state_probs(vi)
            pj = state_probs(vj)

            ent_i = shannon_entropy(pi)
            ent_j = shannon_entropy(pj)
            dom_i = dominant_basis_count(pi, dom_thr)
            dom_j = dominant_basis_count(pj, dom_thr)
            dom = int(max(dom_i, dom_j))

            basis = pick_subspace_basis_from_pair(evecs_use, i, j, topk=2)
            leak = leakage_proxy(basis, terms, times=times, steps=trotter_steps)

            sig = signature_key(dom, ent_i, ent_j, leak, ent_step=ent_step, leak_step=leak_step)
            score = interestingness_score(dom, ent_i, ent_j, leak)

            out.append(Candidate(sig=sig, score=score, leak=leak))

    return out

# ----------------------------
# Reporting
# ----------------------------

def report_model(label: str, cands: List[Candidate], stable_frac: float, topK: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int,int,int,int]]]:
    sigs = np.array([c.sig for c in cands], dtype=object)
    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak for c in cands], dtype=float)

    mask = stable_mask(scores, stable_frac)
    rows, overall, stable = enrichment_rows(sigs, mask)

    sat = 1.0 / max(1e-12, stable_frac)
    print("")
    print("----------------------------------------------")
    print(f"Model: {label}")
    print(f"candidates={len(cands)} | score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f} | "
          f"leakage(mean/median/min)={leaks.mean():.3f}/{np.median(leaks):.3f}/{leaks.min():.3f} | dom_count(median)={int(np.median([s[0] for s in sigs]))}")
    print("")
    print("Top signature families (stable-set vs overall):")
    print("Format: sig=(dom, ent_bin_i, ent_bin_j, leak_bin) | overall | stable | enrichment | saturation_ceiling")
    for (sig, o, st, enr) in rows[:topK]:
        print(f"  {sig} | {o:5d} | {st:5d} | {enr:8.2f}x | {sat:8.2f}x")
    if len(rows) > topK:
        print(f"  ... ({len(rows)} total, showing topK={topK})")

    top_sigs = [r[0] for r in rows[:topK]]
    return sigs, scores, leaks, top_sigs

def cross_model_comparison(
    real_sigs: np.ndarray,
    real_scores: np.ndarray,
    null_sigs: np.ndarray,
    null_scores: np.ndarray,
    stable_frac: float,
    top_sigs_union: List[Tuple[int,int,int,int]],
    B: int,
    rng: np.random.Generator,
):
    # enrichment maps on full data
    r_mask = stable_mask(real_scores, stable_frac)
    n_mask = stable_mask(null_scores, stable_frac)
    r_rows, r_over, r_st = enrichment_rows(real_sigs, r_mask)
    n_rows, n_over, n_st = enrichment_rows(null_sigs, n_mask)

    def enr_map(over, st, n_total, n_stable):
        out = {}
        for s, o in over.items():
            p_over = o / n_total
            p_st = st.get(s, 0) / max(1.0, n_stable)
            out[s] = float(p_st / max(1e-12, p_over))
        return out

    r_enr = enr_map(r_over, r_st, float(len(real_scores)), float(np.sum(r_mask)))
    n_enr = enr_map(n_over, n_st, float(len(null_scores)), float(np.sum(n_mask)))

    # ratio CIs
    ci = bootstrap_ratio_ci(
        target_sigs=top_sigs_union,
        real_sig_arr=real_sigs,
        real_scores=real_scores,
        null_sig_arr=null_sigs,
        null_scores=null_scores,
        frac=stable_frac,
        B=B,
        rng=rng,
    )

    print("")
    print("=== Cross-model calibration (REAL vs NULL) ===")
    print("Interpretation: claim 'non-chance' only when REAL/NULL ratio is consistently > 1 under matched settings.")
    print("Format: sig | enr_REAL | enr_NULL | ratio | ratio_CI_95")
    for s in top_sigs_union:
        er = r_enr.get(s, 0.0)
        en = n_enr.get(s, 0.0)
        ratio = er / max(1e-12, en)
        lo, hi = ci.get(s, (float("nan"), float("nan")))
        print(f"  {s} | {er:7.2f}x | {en:7.2f}x | {ratio:7.2f}x | [{lo:6.2f}, {hi:6.2f}]")
    print("")

# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0_7 SigV2 matched baseline calibration (REAL vs NULL_HAAR_BASIS).")
    p.add_argument("--n_qubits", type=int, default=3)
    p.add_argument("--num_terms", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5000)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--dom_thr", type=float, default=0.25)
    p.add_argument("--ent_step", type=float, default=0.10)
    p.add_argument("--leak_step", type=float, default=0.05)
    p.add_argument("--stable_frac", type=float, default=0.01)
    p.add_argument("--topK", type=int, default=25)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--trotter_steps", type=int, default=8)
    p.add_argument("--times", type=str, default="0.5,1.0,1.5", help="Comma-separated times for leakage proxy.")
    p.add_argument("--out", type=str, default="v0_7_baseline_sigV2_matched_FRESH6_output.txt")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    times = [float(x.strip()) for x in args.times.split(",") if x.strip()]
    rng = np.random.default_rng(int(args.seed))

    tee = Tee(args.out)
    sys.stdout = tee

    try:
        print(f"=== v0_7 SigV2: Baseline / Null-Model Calibration (Matched, FAIR) ===")
        print(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.num_terms} | seeds={args.seeds}")
        print(f"Near-degenerate neighbor eps={args.eps:.3f}")
        print(f"Dominant threshold={args.dom_thr:.3f} | ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
        print(f"Leakage proxy: times={times} | trotter_steps={args.trotter_steps}")
        print(f"Evidence: stable_frac={args.stable_frac:.3f} | topK={args.topK} | bootstrap={args.bootstrap}")
        print("Models: REAL + NULL_HAAR_BASIS (spectrum-matched)")
        print(f"Output: {os.path.abspath(args.out)}")
        print("")

        # Generate REAL + NULL with matched pipeline settings
        t0 = time.time()
        real_cands = generate_candidates(
            model="REAL",
            n_qubits=args.n_qubits,
            num_terms=args.num_terms,
            seeds=args.seeds,
            eps=args.eps,
            dom_thr=args.dom_thr,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=times,
            trotter_steps=args.trotter_steps,
            rng=rng,
        )
        t1 = time.time()
        null_cands = generate_candidates(
            model="NULL_HAAR_BASIS",
            n_qubits=args.n_qubits,
            num_terms=args.num_terms,
            seeds=args.seeds,
            eps=args.eps,
            dom_thr=args.dom_thr,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=times,
            trotter_steps=args.trotter_steps,
            rng=rng,
        )
        t2 = time.time()

        print(f"Model: REAL           | candidates={len(real_cands)} | elapsed={t1-t0:.1f}s")
        print(f"Model: NULL_HAAR_BASIS| candidates={len(null_cands)} | elapsed={t2-t1:.1f}s")

        # If either side has no candidates, bail with explanation
        if len(real_cands) < 10 or len(null_cands) < 10:
            print("")
            print("WARNING: Too few candidates produced for stable calibration.")
            print("Increase --seeds or relax --eps.")
            return

        real_sigs, real_scores, real_leaks, top_real = report_model("REAL", real_cands, args.stable_frac, args.topK)
        null_sigs, null_scores, null_leaks, top_null = report_model("NULL_HAAR_BASIS", null_cands, args.stable_frac, args.topK)

        # Cross-model comparison on union of top signatures (kept small for readability)
        union = []
        seen = set()
        for s in (top_real + top_null):
            if s not in seen:
                union.append(s)
                seen.add(s)
            if len(union) >= args.topK:
                break

        rng_ci = np.random.default_rng(int(args.seed) + 999)
        cross_model_comparison(
            real_sigs=real_sigs,
            real_scores=real_scores,
            null_sigs=null_sigs,
            null_scores=null_scores,
            stable_frac=args.stable_frac,
            top_sigs_union=union,
            B=args.bootstrap,
            rng=rng_ci,
        )

        print("=== Notes (scientific reading) ===")
        print("1) Enrichment saturates at ~1/stable_frac when a signature occurs only inside the stable set.")
        print("   In that case, within-model enrichment alone is not strong evidence.")
        print("2) The cross-model REAL/NULL ratio (with CI) is the relevant calibration: it asks whether")
        print("   the same signature family is *more* concentrated in stable candidates than expected under chance.")
        print("3) If ratios are consistently ~1 (or CI overlaps 1), the observed families are compatible with")
        print("   pipeline-defined selection effects and do not yet justify non-chance claims.")
        print("")
        print("=== End of v0_7 SigV2 (FRESH6) ===")

    finally:
        # restore stdout even if something fails
        try:
            sys.stdout = tee._stdout
        except Exception:
            pass
        tee.close()

if __name__ == "__main__":
    main()
