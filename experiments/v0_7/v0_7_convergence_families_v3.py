#!/usr/bin/env python3
"""
v0_7_convergence_families_v3.py

Purpose (SigV2/SigV3-style, no hype):
- Run multiple seed batches and test whether "signature families" are:
  (1) stable across batches (Top-K overlap),
  (2) separable at the distribution level (REAL vs NULL_HAAR_BASIS),
  (3) meaningfully calibrated under a matched null (spectrum-matched Haar basis),
      using smoothing + minimum-count filters so ratios are readable.

This script is self-contained (numpy-only). It does not require Qiskit.
It uses exact spectral time evolution (U(t)=V exp(-iEt) V†), which is fast for d=2^n (here default n=3).

Outputs:
- A single readable TXT report (UTF-8).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


# ----------------------------
# IO helper: tee to stdout + file (UTF-8)
# ----------------------------
class Tee:
    def __init__(self, filename: str):
        self._stdout = sys.stdout
        self._f = open(filename, "w", encoding="utf-8")

    def write(self, s: str):
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
# Basic quantum primitives (numpy)
# ----------------------------
_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def kron_n(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def pauli_string_matrix(ps: str) -> np.ndarray:
    return kron_n([_PAULI[c] for c in ps])


def random_pauli_strings(rng: np.random.Generator, n_qubits: int, num_terms: int) -> List[str]:
    """
    Sample unique Pauli strings length n from {I,X,Y,Z}^n.
    We allow I, but avoid the all-identity string (it only shifts energy).
    """
    seen = set()
    out: List[str] = []
    alphabet = np.array(["I", "X", "Y", "Z"])
    while len(out) < num_terms:
        s = "".join(rng.choice(alphabet, size=n_qubits))
        if s == "I" * n_qubits:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


class PauliCache:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.cache: Dict[str, np.ndarray] = {}

    def mat(self, ps: str) -> np.ndarray:
        if ps not in self.cache:
            self.cache[ps] = pauli_string_matrix(ps)
        return self.cache[ps]


def build_random_hamiltonian(
    rng: np.random.Generator,
    n_qubits: int,
    num_terms: int,
    cache: PauliCache,
    coeff_scale: float = 1.0,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    H = sum_k c_k P_k, where c_k are real -> Hermitian.
    Returns matrix H and list of (pauli_string, coeff).
    """
    strings = random_pauli_strings(rng, n_qubits, num_terms)
    coeffs = rng.uniform(-coeff_scale, coeff_scale, size=num_terms).astype(float)
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    terms: List[Tuple[str, float]] = []
    for s, c in zip(strings, coeffs):
        H = H + c * cache.mat(s)
        terms.append((s, float(c)))
    return H, terms


def haar_unitary(rng: np.random.Generator, d: int) -> np.ndarray:
    """
    Haar random unitary via QR of complex Gaussian matrix.
    """
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phase = diag / np.abs(diag)
    q = q * phase
    return q


def shannon_entropy_bits(p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0)
    return float(-np.sum(p * np.log2(p)))


def exact_evolve_from_eig(V: np.ndarray, E: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """
    psi(t) = V exp(-i E t) V† psi0
    """
    phases = np.exp(-1j * E * t)
    # work in eigenbasis:
    a = V.conj().T @ psi0
    a = phases * a
    return V @ a


# ----------------------------
# Candidate records and signatures
# ----------------------------
Sig = Tuple[int, int, int]  # (dom_count, ent_bin, leak_bin)


@dataclass
class Candidate:
    sig: Sig
    entropy_avg: float
    leakage: float
    score: float


def dominant_support(p: np.ndarray, threshold: float) -> np.ndarray:
    """
    Return indices with p >= threshold. If none, return argmax.
    """
    idx = np.where(p >= threshold)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(p))], dtype=int)
    return idx


def make_signature_and_metrics(
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    V_dyn: np.ndarray,
    E_dyn: np.ndarray,
    dom_threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
) -> Tuple[Sig, float, float]:
    """
    Build:
      - dom_count: size of union of dominant basis indices from the two eigenvectors
      - entropy_avg: average Shannon entropy (bits) of the two eigenvectors' basis distributions
      - leakage: average leakage of a basis-state superposition over times list, under exact dynamics
      - signature: (dom_count, ent_bin, leak_bin)
    """
    p_i = np.abs(vec_i) ** 2
    p_j = np.abs(vec_j) ** 2
    ent_i = shannon_entropy_bits(p_i)
    ent_j = shannon_entropy_bits(p_j)
    entropy_avg = 0.5 * (ent_i + ent_j)

    S_i = dominant_support(p_i, dom_threshold)
    S_j = dominant_support(p_j, dom_threshold)
    S = np.unique(np.concatenate([S_i, S_j]))
    dom_count = int(S.size)

    # initial state: equal superposition over S in computational basis
    d = vec_i.shape[0]
    psi0 = np.zeros(d, dtype=complex)
    psi0[S] = 1.0
    psi0 = psi0 / np.linalg.norm(psi0)

    leaks: List[float] = []
    for t in times:
        psi_t = exact_evolve_from_eig(V_dyn, E_dyn, psi0, t)
        p_t = np.abs(psi_t) ** 2
        in_sub = float(np.sum(p_t[S]))
        leaks.append(max(0.0, 1.0 - in_sub))

    leakage = float(np.mean(leaks))

    ent_bin = int(math.floor(entropy_avg / ent_step + 1e-12))
    leak_bin = int(math.floor(leakage / leak_step + 1e-12))
    sig: Sig = (dom_count, ent_bin, leak_bin)
    return sig, entropy_avg, leakage


def compute_scores(cands: List[Candidate]) -> None:
    """
    Score is a monotone "stability/structure" proxy:
      raw = (1 - leakage) / (1 + dom_count) / (1 + entropy_avg)
    Then normalized to [0,1] across this model.
    """
    if not cands:
        return
    raw = np.array([(1.0 - c.leakage) / (1.0 + c.sig[0]) / (1.0 + c.entropy_avg) for c in cands], dtype=float)
    mx = float(np.max(raw)) if raw.size else 0.0
    if mx <= 0:
        for c in cands:
            c.score = 0.0
        return
    for i, c in enumerate(cands):
        c.score = float(raw[i] / mx)


def stable_mask_from_scores(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = scores.size
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.ceil(stable_frac * n)))
    # top-k by score
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def enrichment_table(
    sigs: List[Sig],
    stable_mask: np.ndarray,
    min_overall: int,
    min_stable: int,
    smoothing: float,
) -> Dict[Sig, Tuple[int, int, float]]:
    """
    Return dict: sig -> (overall_count, stable_count, enrichment)
    Enrichment = (stable_count/stable_total) / (overall_count/total)
    with optional Laplace-style smoothing applied to counts.
    """
    total = len(sigs)
    stable_total = int(np.sum(stable_mask))
    overall: Dict[Sig, int] = {}
    stable: Dict[Sig, int] = {}
    for i, s in enumerate(sigs):
        overall[s] = overall.get(s, 0) + 1
        if stable_mask[i]:
            stable[s] = stable.get(s, 0) + 1

    out: Dict[Sig, Tuple[int, int, float]] = {}
    for s, oc in overall.items():
        sc = stable.get(s, 0)
        if oc < min_overall or sc < min_stable:
            continue
        # smoothed proportions
        p_stable = (sc + smoothing) / (stable_total + smoothing * 2.0)
        p_overall = (oc + smoothing) / (total + smoothing * 2.0)
        enr = float(p_stable / p_overall)
        out[s] = (oc, sc, enr)
    return out


def topK_sigs_by_enrichment(table: Dict[Sig, Tuple[int, int, float]], topK: int) -> List[Sig]:
    items = sorted(table.items(), key=lambda kv: kv[1][2], reverse=True)
    return [k for (k, _) in items[:topK]]


def bootstrap_ratio_ci(
    sig: Sig,
    sigs_real: List[Sig],
    sigs_null: List[Sig],
    scores_real: np.ndarray,
    scores_null: np.ndarray,
    stable_frac: float,
    min_overall: int,
    min_stable: int,
    smoothing: float,
    B: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for ratio of enrichment (REAL/NULL) for a given signature.
    Returns (ratio_point, lo, hi).
    """
    # point estimate:
    mR = stable_mask_from_scores(scores_real, stable_frac)
    mN = stable_mask_from_scores(scores_null, stable_frac)
    tabR = enrichment_table(sigs_real, mR, min_overall, min_stable, smoothing)
    tabN = enrichment_table(sigs_null, mN, min_overall, min_stable, smoothing)
    enrR = tabR.get(sig, (0, 0, 0.0))[2]
    enrN = tabN.get(sig, (0, 0, 0.0))[2]
    ratio_point = (enrR + 1e-12) / (enrN + 1e-12)

    # bootstrap
    nR = len(sigs_real)
    nN = len(sigs_null)
    if nR == 0 or nN == 0 or B <= 0:
        return ratio_point, ratio_point, ratio_point

    ratios = []
    idxR = np.arange(nR)
    idxN = np.arange(nN)
    for _ in range(B):
        bR = rng.choice(idxR, size=nR, replace=True)
        bN = rng.choice(idxN, size=nN, replace=True)

        b_sigsR = [sigs_real[i] for i in bR]
        b_sigsN = [sigs_null[i] for i in bN]

        b_scoresR = scores_real[bR]
        b_scoresN = scores_null[bN]

        bmR = stable_mask_from_scores(b_scoresR, stable_frac)
        bmN = stable_mask_from_scores(b_scoresN, stable_frac)

        b_tabR = enrichment_table(b_sigsR, bmR, min_overall, min_stable, smoothing)
        b_tabN = enrichment_table(b_sigsN, bmN, min_overall, min_stable, smoothing)

        b_enrR = b_tabR.get(sig, (0, 0, 0.0))[2]
        b_enrN = b_tabN.get(sig, (0, 0, 0.0))[2]
        ratios.append((b_enrR + 1e-12) / (b_enrN + 1e-12))

    lo, hi = np.quantile(ratios, [0.025, 0.975]).tolist()
    return ratio_point, float(lo), float(hi)


# ----------------------------
# Batch run: REAL vs NULL_HAAR_BASIS (spectrum-matched)
# ----------------------------
@dataclass
class ModelRun:
    name: str
    candidates: List[Candidate]
    sigs: List[Sig]
    scores: np.ndarray
    entropy_bins: np.ndarray
    leak_bins: np.ndarray
    elapsed_s: float


def run_batch(
    batch_seed: int,
    n_qubits: int,
    num_terms: int,
    n_seeds: int,
    eps_neighbor: float,
    dom_threshold: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    coeff_scale: float,
    rng_global: np.random.Generator,
) -> Tuple[ModelRun, ModelRun]:
    """
    For each seed:
      - build REAL Pauli-sum Hamiltonian
      - diagonalize -> (E, V)
      - build NULL_HAAR_BASIS: same E, eigenvectors U ~ Haar
      - detect near-degenerate neighbor pairs on E (same for both)
      - for each pair: compute signature + leakage proxy under exact dynamics
    """
    t0 = time.time()
    d = 2 ** n_qubits
    cache = PauliCache(n_qubits)

    real_cands: List[Candidate] = []
    null_cands: List[Candidate] = []

    # deterministic per-seed: derive independent rng streams
    for s in range(n_seeds):
        seed = batch_seed + s
        rng = np.random.default_rng(seed)

        H, _terms = build_random_hamiltonian(rng, n_qubits, num_terms, cache, coeff_scale=coeff_scale)
        E, V = np.linalg.eigh(H)

        # build Haar eigenbasis for NULL (spectrum matched)
        U = haar_unitary(rng, d)

        # neighbor scan
        for i in range(d - 1):
            if abs(E[i + 1] - E[i]) >= eps_neighbor:
                continue

            # REAL candidate
            sigR, entR, leakR = make_signature_and_metrics(
                V[:, i], V[:, i + 1],
                V_dyn=V, E_dyn=E,
                dom_threshold=dom_threshold,
                ent_step=ent_step,
                leak_step=leak_step,
                times=times,
            )
            real_cands.append(Candidate(sig=sigR, entropy_avg=entR, leakage=leakR, score=0.0))

            # NULL candidate (same spectrum, random eigenbasis)
            sigN, entN, leakN = make_signature_and_metrics(
                U[:, i], U[:, i + 1],
                V_dyn=U, E_dyn=E,
                dom_threshold=dom_threshold,
                ent_step=ent_step,
                leak_step=leak_step,
                times=times,
            )
            null_cands.append(Candidate(sig=sigN, entropy_avg=entN, leakage=leakN, score=0.0))

    # score normalization within each model
    compute_scores(real_cands)
    compute_scores(null_cands)

    def pack(name: str, cands: List[Candidate], elapsed_base: float) -> ModelRun:
        sigs = [c.sig for c in cands]
        scores = np.array([c.score for c in cands], dtype=float) if cands else np.zeros((0,), dtype=float)
        ent_bins = np.array([c.sig[1] for c in cands], dtype=int) if cands else np.zeros((0,), dtype=int)
        leak_bins = np.array([c.sig[2] for c in cands], dtype=int) if cands else np.zeros((0,), dtype=int)
        return ModelRun(name=name, candidates=cands, sigs=sigs, scores=scores,
                        entropy_bins=ent_bins, leak_bins=leak_bins,
                        elapsed_s=elapsed_base)

    elapsed = time.time() - t0
    # split elapsed approximately (we measured once)
    real_run = pack("REAL", real_cands, elapsed_base=elapsed)
    null_run = pack("NULL_HAAR_BASIS", null_cands, elapsed_base=elapsed)
    return real_run, null_run


# ----------------------------
# Reporting helpers
# ----------------------------
def fmt_stats(x: np.ndarray) -> str:
    if x.size == 0:
        return "n=0"
    return f"n={x.size} | mean={np.mean(x):.3f} | median={np.median(x):.3f} | min={np.min(x):.3f} | max={np.max(x):.3f}"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def report_batch_summary(
    batch_id: int,
    real: ModelRun,
    null: ModelRun,
    args: argparse.Namespace,
    rng_ci: np.random.Generator,
) -> Dict[str, object]:
    """
    Produce readable output for one batch and return a dict of key artifacts for cross-batch analysis.
    """
    print("\n" + "=" * 62)
    print(f"Batch {batch_id} | seed_offset={args.seed0 + batch_id * args.batch_stride} | seeds={args.seeds_per_batch}")
    print("-" * 62)

    print(f"Model: REAL           | candidates={len(real.candidates)} | elapsed~{real.elapsed_s:.1f}s")
    print(f"Model: NULL_HAAR_BASIS| candidates={len(null.candidates)} | elapsed~{null.elapsed_s:.1f}s")

    # Distribution-level effect sizes (bins + raw)
    entR = real.entropy_bins.astype(float)
    entN = null.entropy_bins.astype(float)
    leakR = real.leak_bins.astype(float)
    leakN = null.leak_bins.astype(float)
    print("\nDistribution diagnostics (bin-level):")
    print(f"  entropy_bin REAL: {fmt_stats(entR)}")
    print(f"  entropy_bin NULL: {fmt_stats(entN)}")
    print(f"  leak_bin    REAL: {fmt_stats(leakR)}")
    print(f"  leak_bin    NULL: {fmt_stats(leakN)}")

    # Stable set
    mR = stable_mask_from_scores(real.scores, args.stable_frac)
    mN = stable_mask_from_scores(null.scores, args.stable_frac)

    # Enrichment tables with smoothing + min-count filtering
    tabR = enrichment_table(real.sigs, mR, args.min_overall, args.min_stable, args.smoothing)
    tabN = enrichment_table(null.sigs, mN, args.min_overall, args.min_stable, args.smoothing)

    print("\nEligibility after min-count filter:")
    print(f"  REAL eligible families: {len(tabR)}")
    print(f"  NULL eligible families: {len(tabN)}")

    topR = topK_sigs_by_enrichment(tabR, args.topK)
    print(f"\nTop-{args.topK} REAL families by enrichment (filtered, smoothed):")
    print("Format: sig=(dom, ent_bin, leak_bin) | overall | stable | enrichment | enr_NULL | ratio | ratio_CI_95")
    shown = 0
    rows = []
    for sig in topR:
        ocR, scR, enrR = tabR[sig]
        enrN = tabN.get(sig, (0, 0, 0.0))[2]
        ratio, lo, hi = bootstrap_ratio_ci(
            sig,
            sigs_real=real.sigs,
            sigs_null=null.sigs,
            scores_real=real.scores,
            scores_null=null.scores,
            stable_frac=args.stable_frac,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            smoothing=args.smoothing,
            B=args.bootstrap,
            rng=rng_ci,
        )
        rows.append((sig, ocR, scR, enrR, enrN, ratio, lo, hi))
        # concise printing
        print(f"  {sig} | {ocR:5d} | {scR:5d} | {enrR:7.2f}x | {enrN:7.2f}x | {ratio:7.2f}x | [{lo:7.2f}, {hi:7.2f}]")
        shown += 1
        if shown >= args.topK:
            break

    # top family set for overlap evaluation
    top_set = set(topR)

    # store batch artifacts
    return {
        "batch_id": batch_id,
        "seed_offset": args.seed0 + batch_id * args.batch_stride,
        "n_real": len(real.candidates),
        "n_null": len(null.candidates),
        "top_real": top_set,
        "top_real_list": topR,
        "rows": rows,
        "entropy_bin_median_real": float(np.median(entR)) if entR.size else float("nan"),
        "entropy_bin_median_null": float(np.median(entN)) if entN.size else float("nan"),
        "leak_bin_median_real": float(np.median(leakR)) if leakR.size else float("nan"),
        "leak_bin_median_null": float(np.median(leakN)) if leakN.size else float("nan"),
    }


def report_cross_batch_overlap(batch_artifacts: List[Dict[str, object]], topK: int):
    print("\n" + "=" * 62)
    print(f"Cross-batch convergence (Top-{topK} REAL-family overlap)")
    print("-" * 62)
    sets = [a["top_real"] for a in batch_artifacts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            jac = jaccard(sets[i], sets[j])
            inter = len(sets[i] & sets[j])
            print(f"  Batch {i} vs {j}: overlap={inter:2d} | Jaccard={jac:.3f}")

    # multi-way intersection and union
    inter_all = sets[0].copy()
    union_all = set()
    for s in sets:
        inter_all &= s
        union_all |= s

    print(f"\n  Intersection across all batches: {len(inter_all)} families")
    if inter_all:
        inter_list = sorted(list(inter_all))[:min(10, len(inter_all))]
        print(f"  Example intersecting families (up to 10): {inter_list}")

    print(f"  Union across all batches: {len(union_all)} families")
    print("Interpretation guide:")
    print("  - If pairwise overlaps are consistently large (and intersection non-trivial),")
    print("    the family labels are stable against batch resampling.")
    print("  - If overlaps collapse toward ~0, the labeling is not replicable at this resolution.")


def parse_times(s: str) -> List[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="v0_7 convergence test: replicable signature families under matched null.")
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--terms", type=int, default=5)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--batch_stride", type=int, default=100000, help="offset between batches to decorrelate RNG streams")
    ap.add_argument("--eps", type=float, default=0.05, help="neighbor near-degeneracy threshold")
    ap.add_argument("--dom_threshold", type=float, default=0.25)
    ap.add_argument("--ent_step", type=float, default=0.10)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--times", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=25)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--smoothing", type=float, default=0.5, help="Laplace smoothing for proportions")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument("--coeff_scale", type=float, default=1.0)
    ap.add_argument("--output", type=str, default="v0_7_convergence_families_v3_output.txt")
    args = ap.parse_args()

    tee = Tee(args.output)
    sys.stdout = tee

    try:
        times = parse_times(args.times)
        print("=== v0_7: Convergence of Signature Families under Matched Null (v3) ===")
        print(f"Qubits: {args.n_qubits} (d={2**args.n_qubits}) | terms={args.terms}")
        print(f"Batches: {args.batches} x seeds_per_batch={args.seeds_per_batch} | seed0={args.seed0} | stride={args.batch_stride}")
        print(f"Neighbor eps={args.eps}")
        print(f"Dominant threshold={args.dom_threshold} | ent_step={args.ent_step} | leak_step={args.leak_step}")
        print(f"Leakage proxy times={times} (exact spectral evolution)")
        print(f"Evidence settings: stable_frac={args.stable_frac:.3f} | topK={args.topK} | min_overall={args.min_overall} | min_stable={args.min_stable} | smoothing={args.smoothing}")
        print(f"Bootstrap: B={args.bootstrap} | seed={args.bootstrap_seed}")
        print(f"Output: {args.output}")
        print()

        rng_ci = np.random.default_rng(args.bootstrap_seed)
        rng_global = np.random.default_rng(999)

        batch_artifacts: List[Dict[str, object]] = []
        for b in range(args.batches):
            batch_seed = args.seed0 + b * args.batch_stride
            real, null = run_batch(
                batch_seed=batch_seed,
                n_qubits=args.n_qubits,
                num_terms=args.terms,
                n_seeds=args.seeds_per_batch,
                eps_neighbor=args.eps,
                dom_threshold=args.dom_threshold,
                ent_step=args.ent_step,
                leak_step=args.leak_step,
                times=times,
                coeff_scale=args.coeff_scale,
                rng_global=rng_global,
            )
            art = report_batch_summary(b, real, null, args, rng_ci)
            batch_artifacts.append(art)

        report_cross_batch_overlap(batch_artifacts, args.topK)

        print("\n" + "=" * 62)
        print("Bottom-line reading (method-only, no hype):")
        print("-" * 62)
        print("1) If Top-K overlaps remain substantial across batches, the family labels are replicable at this resolution.")
        print("2) If REAL-vs-NULL ratios (with bootstrap CI) are consistently > 1 for the same families,")
        print("   then the stable-set concentration is not explained by the matched null at these settings.")
        print("3) If ratios or overlaps collapse when you slightly change (eps, dom_threshold, ent_step, leak_step),")
        print("   then the labels are likely discretization artifacts; coarsen bins or raise min_overall.")
        print("\n=== End of v0_7_convergence_families_v3 ===")

    finally:
        try:
            tee.flush()
            tee.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
