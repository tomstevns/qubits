#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_6_1_Final_patched.py — Systematic Signature Detection (v0_6.1) with evidence-hygiene output

This is a PATCHED reporting/evidence-hygiene layer built on the same discovery logic as v0_6.1_Final.py:
  - Hamiltonian sampling (Pauli-sum)
  - near-degenerate neighbor scan
  - eigenvector structure extraction (dominant basis support)
  - simple time-evolution leakage (exact spectral evolution)
  - interestingness score (baseline-normalized leakage)

What this patch changes (ONLY aggregation/reporting, not discovery logic):
  A) Coarse-grains fine signature keys into SIGNATURE FAMILIES (reduces fragmentation)
  B) Applies minimum support filtering for claims (reduces one-off artifacts)
  C) Uses top-K replication (K=50/100) instead of brittle top-10 replication

Output:
  - Console AND UTF-8 log file (default: v0_6_1_Final_patched_output.txt)

Run examples:
  python v0_6_1_Final_patched.py --seeds 50000
  python v0_6_1_Final_patched.py --seeds 50000 --topK 100 --min_support 100
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp
except Exception as e:
    raise RuntimeError("Requires qiskit (SparsePauliOp). Install: pip install qiskit") from e


# -----------------------------
# Logging helper (UTF-8)
# -----------------------------
class Tee:
    def __init__(self, filename: str, mode: str = "w", encoding: str = "utf-8"):
        self.filename, self.mode, self.encoding = filename, mode, encoding
        self._file = None
        self._stdout = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._file = open(self.filename, self.mode, encoding=self.encoding, errors="replace")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout = self._stdout
        finally:
            if self._file:
                try:
                    self._file.flush()
                finally:
                    self._file.close()
        return False

    def write(self, obj):
        s = str(obj)
        if self._stdout:
            self._stdout.write(s)
        if self._file:
            self._file.write(s)

    def flush(self):
        if self._stdout:
            self._stdout.flush()
        if self._file:
            self._file.flush()


# -----------------------------
# Core math utilities (unchanged intent)
# -----------------------------
def shannon_entropy_base2(prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(prob, eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def build_random_hamiltonian_sparse(n_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    paulis = ["I", "X", "Y", "Z"]
    labels, coeffs = [], []
    for _ in range(num_terms):
        labels.append("".join(rng.choice(paulis, size=n_qubits)))
        coeffs.append(complex(rng.uniform(-1.0, 1.0)))
    return SparsePauliOp(labels, coeffs=np.asarray(coeffs, dtype=complex))


def diagonalize(H: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray]:
    mat = H.to_matrix()
    evals, evecs = np.linalg.eigh(mat)
    return evals.real, evecs  # columns are eigenvectors


def evolve(evals: np.ndarray, evecs: np.ndarray, t: float, psi0: np.ndarray) -> np.ndarray:
    phases = np.exp(-1j * evals * t)
    a = evecs.conj().T @ psi0
    return evecs @ (phases * a)


def dominant_support(prob: np.ndarray, prob_threshold: float) -> np.ndarray:
    return np.sort(np.where(prob >= prob_threshold)[0])


def signature_key(dom_count: int, ent_i: float, ent_j: float, leakage: float, ent_bin: float, leak_bin: float) -> Tuple[int, int, int, int]:
    """Fine signature key (same idea as v0_6.1_Final)."""
    ei = int(round(ent_i / ent_bin))
    ej = int(round(ent_j / ent_bin))
    if ej < ei:
        ei, ej = ej, ei  # order-invariant
    lk = int(round(leakage / leak_bin))
    return (int(dom_count), ei, ej, lk)


def score_interestingness(leakage_raw: float, dom_count: int, d: int) -> float:
    """Baseline-normalized leakage -> [0,1], higher is better."""
    baseline = 1.0 - (dom_count / float(d))
    if baseline <= 1e-12:
        return 0.0
    ratio = max(0.0, min(1.0, leakage_raw / baseline))
    return 1.0 - ratio


# -----------------------------
# Patch layer: signature families + support filtering
# -----------------------------
SigKey = Tuple[int, int, int, int]
FamilyKey = Tuple[int, int, int, int]


def signature_family_from_fine(
    sig: SigKey,
    ent_bin_fine: float,
    leak_bin_fine: float,
    ent_step_family: float,
    leak_step_family: float,
) -> FamilyKey:
    """
    Map fine signature bins -> coarser family bins.

    sig = (dom_count, ent_i_bin_fine, ent_j_bin_fine, leak_bin_fine_idx)
    Convert the binned values back to approximate physical values using fine bin sizes,
    then re-bin using family steps.
    """
    dom_count, ei_f, ej_f, lk_f = sig

    ent_i_est = float(ei_f) * float(ent_bin_fine)
    ent_j_est = float(ej_f) * float(ent_bin_fine)
    leak_est = float(lk_f) * float(leak_bin_fine)

    ei = int(round(ent_i_est / ent_step_family))
    ej = int(round(ent_j_est / ent_step_family))
    if ej < ei:
        ei, ej = ej, ei

    lk = int(round(leak_est / leak_step_family))
    return (int(dom_count), int(ei), int(ej), int(lk))


def aggregate_counts_by_family(
    fine_counts: Dict[SigKey, int],
    ent_bin_fine: float,
    leak_bin_fine: float,
    ent_step_family: float,
    leak_step_family: float,
) -> Dict[FamilyKey, int]:
    out: Dict[FamilyKey, int] = {}
    for sig, cnt in fine_counts.items():
        fam = signature_family_from_fine(sig, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
        out[fam] = out.get(fam, 0) + int(cnt)
    return out


def rank_families(
    family_overall: Dict[FamilyKey, int],
    family_stable: Dict[FamilyKey, int],
    stable_frac: float,
    min_support: int,
    top_k: int,
) -> List[Tuple[FamilyKey, float, float, int, int]]:
    """
    Rank supported families. Returns:
      (family_key, enrichment, stable_rate, stable_count, overall_count)
    """
    ranked: List[Tuple[FamilyKey, float, float, int, int]] = []
    if stable_frac <= 0:
        return ranked

    for fam, ca in family_overall.items():
        if ca < min_support:
            continue
        cs = family_stable.get(fam, 0)
        stable_rate = cs / ca if ca > 0 else 0.0
        enrichment = (stable_rate / stable_frac) if stable_frac > 0 else float("nan")
        ranked.append((fam, float(enrichment), float(stable_rate), int(cs), int(ca)))

    ranked.sort(key=lambda x: (-x[1], -x[2], -x[4], x[0]))
    return ranked[:top_k]


def bootstrap_family_enrichment_ci(
    candidates: Sequence["Candidate"],
    target_family: FamilyKey,
    top_fraction: float,
    n_boot: int,
    rng_seed: int,
    ent_bin_fine: float,
    leak_bin_fine: float,
    ent_step_family: float,
    leak_step_family: float,
    min_support: int,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the enrichment of a target FAMILY.
    Uses resampling over candidates (keeps discovery logic unchanged).
    """
    rng = np.random.default_rng(rng_seed)
    N = len(candidates)
    if N == 0:
        return (float("nan"), float("nan"), float("nan"))

    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        sample = [candidates[i] for i in idx]
        scores = np.array([c.score for c in sample], dtype=float)
        cutoff = np.quantile(scores, 1.0 - top_fraction)
        stable = [c for c in sample if c.score >= cutoff]
        if not stable:
            continue

        fine_overall: Dict[SigKey, int] = {}
        fine_stable: Dict[SigKey, int] = {}
        for c in sample:
            fine_overall[c.sig] = fine_overall.get(c.sig, 0) + 1
        for c in stable:
            fine_stable[c.sig] = fine_stable.get(c.sig, 0) + 1

        fam_overall = aggregate_counts_by_family(fine_overall, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
        fam_stable = aggregate_counts_by_family(fine_stable, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)

        if fam_overall.get(target_family, 0) < min_support:
            continue

        stable_frac = len(stable) / len(sample)
        ca = fam_overall.get(target_family, 0)
        cs = fam_stable.get(target_family, 0)
        stable_rate = cs / ca if ca > 0 else 0.0
        enrichment = stable_rate / stable_frac if stable_frac > 0 else float("nan")
        if math.isfinite(enrichment):
            vals.append(float(enrichment))

    if len(vals) < max(20, n_boot // 20):
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(vals, dtype=float)
    return (float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


# -----------------------------
# Candidate extraction (core logic)
# -----------------------------
@dataclass(frozen=True)
class Candidate:
    seed: int
    pair: Tuple[int, int]
    dE: float
    dom_count: int
    entropy_i: float
    entropy_j: float
    leakage_raw: float
    score: float
    sig: SigKey  # fine signature key


def candidates_for_seed(
    seed: int,
    n_qubits: int,
    num_terms: int,
    epsilon: float,
    prob_threshold: float,
    ent_bin: float,
    leak_bin: float,
    t: float,
) -> List[Candidate]:
    H = build_random_hamiltonian_sparse(n_qubits, num_terms, seed)
    evals, evecs = diagonalize(H)
    d = 2 ** n_qubits
    out: List[Candidate] = []

    for i in range(len(evals) - 1):
        j = i + 1
        dE = float(abs(evals[j] - evals[i]))
        if dE >= epsilon:
            continue

        psi_i, psi_j = evecs[:, i], evecs[:, j]
        prob_i, prob_j = np.abs(psi_i) ** 2, np.abs(psi_j) ** 2
        ent_i, ent_j = shannon_entropy_base2(prob_i), shannon_entropy_base2(prob_j)

        supp = np.unique(np.concatenate([dominant_support(prob_i, prob_threshold),
                                         dominant_support(prob_j, prob_threshold)]))
        dom_count = int(len(supp))
        if dom_count <= 0 or dom_count >= d:
            continue

        psi0 = np.zeros((d,), dtype=complex)
        psi0[supp] = 1.0
        psi0 /= np.linalg.norm(psi0)

        psi_t = evolve(evals, evecs, t, psi0)
        prob_t = np.abs(psi_t) ** 2
        stay = float(np.sum(prob_t[supp]))
        leakage_raw = max(0.0, min(1.0, 1.0 - stay))

        score = score_interestingness(leakage_raw, dom_count, d)
        sig = signature_key(dom_count, ent_i, ent_j, leakage_raw, ent_bin, leak_bin)

        out.append(Candidate(seed, (i, j), dE, dom_count, float(ent_i), float(ent_j), float(leakage_raw), float(score), sig))

    return out


def run_scan(
    seeds: Sequence[int],
    n_qubits: int,
    num_terms: int,
    epsilon: float,
    prob_threshold: float,
    ent_bin: float,
    leak_bin: float,
    t: float,
) -> List[Candidate]:
    all_c: List[Candidate] = []
    for sd in seeds:
        all_c.extend(candidates_for_seed(sd, n_qubits, num_terms, epsilon, prob_threshold, ent_bin, leak_bin, t))
    return all_c


# -----------------------------
# Reporting utilities
# -----------------------------
def print_basic_stats(cands: Sequence[Candidate]) -> None:
    if not cands:
        print("No candidates retained.")
        return
    scores = np.array([c.score for c in cands], dtype=float)
    leak = np.array([c.leakage_raw for c in cands], dtype=float)
    dom = np.array([c.dom_count for c in cands], dtype=int)

    print(f"Retained candidates: {len(cands)}")
    print(f"Score: mean={scores.mean():.3f}, median={np.median(scores):.3f}, max={scores.max():.3f}")
    print(f"Leakage_raw: mean={leak.mean():.3f}, median={np.median(leak):.3f}, min={leak.min():.3f}")
    print("dom_count distribution:")
    for k in sorted(set(dom.tolist())):
        n = int(np.sum(dom == k))
        print(f"  dom_count={k}: {100.0*n/len(dom):.1f}% (n={n})")


def holdout_overlap_topK(
    rankA: List[Tuple[FamilyKey, float, float, int, int]],
    rankB: List[Tuple[FamilyKey, float, float, int, int]],
    K: int,
) -> Tuple[float, int, int]:
    setA = set([x[0] for x in rankA[:K]])
    setB = set([x[0] for x in rankB[:K]])
    if not setA or not setB:
        return (float("nan"), 0, 0)
    inter = len(setA & setB)
    union = len(setA | setB)
    return (inter / union if union else 0.0, inter, union)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50000)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--num_terms", type=int, default=5)
    ap.add_argument("--t", type=float, default=1.0)

    ap.add_argument("--ent_bin", type=float, default=0.10)
    ap.add_argument("--leak_bin", type=float, default=0.05)

    ap.add_argument("--family_ent_step", type=float, default=0.10)
    ap.add_argument("--family_leak_step", type=float, default=0.05)
    ap.add_argument("--min_support", type=int, default=50)
    ap.add_argument("--topK", type=int, default=50)

    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--output", type=str, default="v0_6_1_Final_patched_output.txt")

    ap.add_argument("--prob_threshold", type=float, default=0.25)
    ap.add_argument("--epsilon", type=float, default=0.05)

    args = ap.parse_args()

    n_qubits = int(args.n_qubits)
    num_terms = int(args.num_terms)
    d = 2 ** n_qubits
    t = float(args.t)

    ent_bin_fine = float(args.ent_bin)
    leak_bin_fine = float(args.leak_bin)

    ent_step_family = float(args.family_ent_step)
    leak_step_family = float(args.family_leak_step)
    min_support = int(args.min_support)
    topK = int(args.topK)

    prob_threshold = float(args.prob_threshold)
    epsilon = float(args.epsilon)

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    half = len(seeds) // 2
    seeds_A, seeds_B = seeds[:half], seeds[half:]

    thresholds = [0.20, 0.25, 0.30]
    epsilons = [0.02, 0.05, 0.10]
    ent_bins = [0.05, 0.10]

    with Tee(args.output):
        print("=== v0_6.1_Final_patched: Evidence Hygiene for Signature Families ===")
        print(f"Qubits: {n_qubits} (d={d}) | terms={num_terms} | seeds={len(seeds)} | t={t}")
        print(f"Nominal discovery: prob_threshold={prob_threshold:.2f}, epsilon={epsilon:.2f}, ent_bin(fine)={ent_bin_fine:.2f}, leak_bin(fine)={leak_bin_fine:.2f}")
        print(f"Patch layer: family_ent_step={ent_step_family:.2f}, family_leak_step={leak_step_family:.2f}, min_support={min_support}, topK={topK}\n")

        sweep_N = min(5000, len(seeds))
        sweep_seeds = seeds[:sweep_N]
        print("=== 1) Robustness Sweep (family-based) ===")
        print(f"Using first {sweep_N} seeds for sweep (speed). Full run used for enrichment + hold-out.\n")

        sweep_top_fams: Dict[Tuple[float, float, float], List[FamilyKey]] = {}

        for th in thresholds:
            for eps in epsilons:
                for eb in ent_bins:
                    c = run_scan(sweep_seeds, n_qubits, num_terms, eps, th, eb, leak_bin_fine, t)
                    if not c:
                        sweep_top_fams[(th, eps, eb)] = []
                        print(f"[Sweep] th={th:.2f}, eps={eps:.2f}, ent_bin={eb:.2f} -> candidates=0")
                        continue

                    scores = np.array([x.score for x in c], dtype=float)
                    cutoff = np.quantile(scores, 0.95)
                    stable = [x for x in c if x.score >= cutoff]
                    stable_frac = len(stable) / len(c) if c else 0.0

                    fine_overall: Dict[SigKey, int] = {}
                    fine_stable: Dict[SigKey, int] = {}
                    for x in c:
                        fine_overall[x.sig] = fine_overall.get(x.sig, 0) + 1
                    for x in stable:
                        fine_stable[x.sig] = fine_stable.get(x.sig, 0) + 1

                    fam_overall = aggregate_counts_by_family(fine_overall, eb, leak_bin_fine, ent_step_family, leak_step_family)
                    fam_stable = aggregate_counts_by_family(fine_stable, eb, leak_bin_fine, ent_step_family, leak_step_family)

                    ranked = rank_families(fam_overall, fam_stable, stable_frac, min_support=max(10, min_support // 5), top_k=10)
                    sweep_top_fams[(th, eps, eb)] = [r[0] for r in ranked]

                    print(f"[Sweep] th={th:.2f}, eps={eps:.2f}, ent_bin={eb:.2f} -> candidates={len(c)} | stable={len(stable)}")
                    for fam, enr, sr, cs, ca in ranked[:3]:
                        print(f"  fam={fam}  enrichment={enr:.2f}x  stable_rate={sr:.3f}  stable={cs}  overall={ca}")
                    print()

        sets = [set(v) for v in sweep_top_fams.values() if v]
        if len(sets) >= 2:
            jac = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    a, b = sets[i], sets[j]
                    inter, union = len(a & b), len(a | b)
                    jac.append(inter / union if union else 0.0)
            print("Robustness summary (family top-10 sets):")
            print(f"  Avg pairwise Jaccard overlap: {float(np.mean(jac)):.3f}\n")
        else:
            print("Robustness summary: insufficient non-empty sweep results to compute overlap.\n")

        print("=== 2) Enrichment Test (family-based, nominal) + Bootstrap CI ===")
        print(f"Nominal run: th={prob_threshold:.2f}, eps={epsilon:.2f}, ent_bin(fine)={ent_bin_fine:.2f}\n")

        c_full = run_scan(seeds, n_qubits, num_terms, epsilon, prob_threshold, ent_bin_fine, leak_bin_fine, t)
        print("Nominal basic stats:")
        print_basic_stats(c_full)
        print()

        if not c_full:
            print("No candidates retained in nominal run. Exiting.\n")
            print("=== End of v0_6.1_Final_patched ===")
            print(f"Log written to: {os.path.abspath(args.output)}")
            return

        fine_overall_full: Dict[SigKey, int] = {}
        for x in c_full:
            fine_overall_full[x.sig] = fine_overall_full.get(x.sig, 0) + 1
        fam_overall_full = aggregate_counts_by_family(fine_overall_full, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)

        for frac in [0.01, 0.05]:
            print(f"Stable set = top {int(frac*100)}% by score")
            scores = np.array([x.score for x in c_full], dtype=float)
            cutoff = np.quantile(scores, 1.0 - frac)
            stable = [x for x in c_full if x.score >= cutoff]
            stable_frac = len(stable) / len(c_full) if c_full else 0.0

            fine_stable: Dict[SigKey, int] = {}
            for x in stable:
                fine_stable[x.sig] = fine_stable.get(x.sig, 0) + 1
            fam_stable = aggregate_counts_by_family(fine_stable, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)

            ranked = rank_families(fam_overall_full, fam_stable, stable_frac, min_support=min_support, top_k=10)
            if not ranked:
                print("  No supported families to rank (try lowering --min_support).\n")
                continue

            for fam, enr, sr, cs, ca in ranked[:5]:
                mean, lo, hi = bootstrap_family_enrichment_ci(
                    c_full, fam, frac, args.bootstrap, 123,
                    ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family, min_support
                )
                ci = "CI unavailable" if math.isnan(mean) else f"bootstrap={mean:.2f}x, 95% CI [{lo:.2f}, {hi:.2f}]"
                print(f"  fam={fam}  point={enr:.2f}x  stable_rate={sr:.3f}  stable={cs}  overall={ca}  |  {ci}")
            print()

        print("=== 3) Hold-out Replication (family-based, nominal) ===")
        print(f"Half A seeds={len(seeds_A)} | Half B seeds={len(seeds_B)} | topK={topK}\n")

        cA = run_scan(seeds_A, n_qubits, num_terms, epsilon, prob_threshold, ent_bin_fine, leak_bin_fine, t)
        cB = run_scan(seeds_B, n_qubits, num_terms, epsilon, prob_threshold, ent_bin_fine, leak_bin_fine, t)

        def rank_from_candidates(cands: List[Candidate], stable_fraction: float) -> List[Tuple[FamilyKey, float, float, int, int]]:
            if not cands:
                return []
            scores = np.array([x.score for x in cands], dtype=float)
            cutoff = np.quantile(scores, 1.0 - stable_fraction)
            stable = [x for x in cands if x.score >= cutoff]
            stable_frac = len(stable) / len(cands) if cands else 0.0

            fine_overall: Dict[SigKey, int] = {}
            fine_stable: Dict[SigKey, int] = {}
            for x in cands:
                fine_overall[x.sig] = fine_overall.get(x.sig, 0) + 1
            for x in stable:
                fine_stable[x.sig] = fine_stable.get(x.sig, 0) + 1

            fam_overall = aggregate_counts_by_family(fine_overall, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
            fam_stable = aggregate_counts_by_family(fine_stable, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
            return rank_families(fam_overall, fam_stable, stable_frac, min_support=min_support, top_k=max(topK, 10))

        rankA = rank_from_candidates(cA, stable_fraction=0.05)
        rankB = rank_from_candidates(cB, stable_fraction=0.05)

        if not rankA or not rankB:
            print("Hold-out: insufficient supported families in one or both halves (try lowering --min_support).\n")
        else:
            jacc, inter, union = holdout_overlap_topK(rankA, rankB, K=topK)
            print(f"Hold-out overlap on supported family keys: Jaccard={jacc:.3f} (inter={inter}, union={union})")
            print("Top overlapping families (up to 20):")
            overlap = list(set([x[0] for x in rankA[:topK]]) & set([x[0] for x in rankB[:topK]]))
            for fam in overlap[:20]:
                print(f"  fam={fam}")
            print()

        print("=== End of v0_6.1_Final_patched ===")
        print(f"Log written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
