#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_6_1_Final_patched_sigV2.py — Systematic Signature Detection (v0_6.1) with evidence-hygiene output (SigV2)

This is the SigV2 (recommended) version of the v0_6.1 evidence layer.

It keeps the same discovery logic as v0_6 / v0_6.1:
  - Random Pauli-sum Hamiltonians (SparsePauliOp)
  - Neighbor near-degenerate detection: |E_{i+1}-E_i| < epsilon
  - Computational-basis structure extraction:
      * dominant basis support size (dom_count)
      * Shannon entropy of amplitude distribution (bits)
  - Lightweight stability proxy:
      * initialize a state supported on the dominant union
      * exact spectral evolution using diagonalization
      * leakage_raw = 1 - mass remaining inside the support
  - Interestingness score:
      * baseline-normalized leakage (so small dom_count is not unfairly penalized)

SigV2 improvements (without changing the physics intent):
  1) Output hygiene: UTF-8 logging + ASCII-only symbols ("->")
  2) Less-fragile evidence checks:
       - signature families (coarser bins)
       - min_support filtering
       - hold-out replication via top-K family overlap
  3) Paste-ready compact results block at the end.

Output:
  v0_6_1_Final_patched_sigV2_output.txt (default; UTF-8)

Examples:
  python v0_6_1_Final_patched_sigV2.py --seeds 50000
  python v0_6_1_Final_patched_sigV2.py --seeds 50000 --topK 100 --min_support 100
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
    """Tee stdout to both console and a UTF-8 file."""

    def __init__(self, filename: str, mode: str = "w", encoding: str = "utf-8"):
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self._file = None
        self._stdout = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._file = open(self.filename, self.mode, encoding=self.encoding, errors="replace", buffering=1)
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout = self._stdout
        finally:
            if self._file:
                try:
                    self._file.flush()
                except Exception:
                    pass
                try:
                    self._file.close()
                except Exception:
                    pass
        return False

    def write(self, obj):
        s = str(obj)
        if self._stdout:
            self._stdout.write(s)
        if self._file:
            self._file.write(s)

    def flush(self):
        if self._stdout:
            try:
                self._stdout.flush()
            except Exception:
                pass
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass


# -----------------------------
# Core math utilities
# -----------------------------
def shannon_entropy_base2(prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(prob.astype(float), eps, 1.0)
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


def score_interestingness(leakage_raw: float, dom_count: int, d: int) -> float:
    """Baseline-normalized leakage; higher is better."""
    baseline = 1.0 - (dom_count / float(d))
    if baseline <= 1e-12:
        return 0.0
    ratio = max(0.0, min(1.0, leakage_raw / baseline))
    return 1.0 - ratio


# -----------------------------
# Signature keys
# -----------------------------
SigKey = Tuple[int, int, int, int]      # fine key
FamilyKey = Tuple[int, int, int, int]   # family key


def signature_key(dom_count: int, ent_i: float, ent_j: float, leakage: float,
                  ent_bin: float, leak_bin: float) -> SigKey:
    ei = int(round(ent_i / ent_bin))
    ej = int(round(ent_j / ent_bin))
    if ej < ei:
        ei, ej = ej, ei
    lk = int(round(leakage / leak_bin))
    return (int(dom_count), int(ei), int(ej), int(lk))


def signature_family_from_fine(sig: SigKey,
                               ent_bin_fine: float, leak_bin_fine: float,
                               ent_step_family: float, leak_step_family: float) -> FamilyKey:
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


def aggregate_counts_by_family(fine_counts: Dict[SigKey, int],
                               ent_bin_fine: float, leak_bin_fine: float,
                               ent_step_family: float, leak_step_family: float) -> Dict[FamilyKey, int]:
    out: Dict[FamilyKey, int] = {}
    for sig, cnt in fine_counts.items():
        fam = signature_family_from_fine(sig, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
        out[fam] = out.get(fam, 0) + int(cnt)
    return out


# -----------------------------
# Candidate extraction
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
    sig: SigKey


def candidates_for_seed(seed: int,
                        n_qubits: int,
                        num_terms: int,
                        epsilon: float,
                        prob_threshold: float,
                        ent_bin: float,
                        leak_bin: float,
                        t: float) -> List[Candidate]:
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
        ent_i = shannon_entropy_base2(prob_i)
        ent_j = shannon_entropy_base2(prob_j)

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

        out.append(Candidate(seed, (i, j), dE, dom_count, float(ent_i), float(ent_j),
                             float(leakage_raw), float(score), sig))
    return out


def run_scan(seeds: Sequence[int],
             n_qubits: int,
             num_terms: int,
             epsilon: float,
             prob_threshold: float,
             ent_bin: float,
             leak_bin: float,
             t: float) -> List[Candidate]:
    all_c: List[Candidate] = []
    for sd in seeds:
        all_c.extend(candidates_for_seed(sd, n_qubits, num_terms, epsilon, prob_threshold, ent_bin, leak_bin, t))
    return all_c


# -----------------------------
# Evidence utilities
# -----------------------------
def rank_families(fam_overall: Dict[FamilyKey, int],
                  fam_stable: Dict[FamilyKey, int],
                  stable_frac: float,
                  min_support: int,
                  top_k: int) -> List[Tuple[FamilyKey, float, float, int, int]]:
    ranked: List[Tuple[FamilyKey, float, float, int, int]] = []
    if stable_frac <= 0:
        return ranked

    for fam, ca in fam_overall.items():
        if ca < min_support:
            continue
        cs = fam_stable.get(fam, 0)
        stable_rate = cs / ca if ca > 0 else 0.0
        enrichment = (stable_rate / stable_frac) if stable_frac > 0 else float("nan")
        ranked.append((fam, float(enrichment), float(stable_rate), int(cs), int(ca)))

    ranked.sort(key=lambda x: (-x[1], -x[2], -x[4], x[0]))
    return ranked[:top_k]


def holdout_overlap_topK(rankA: List[Tuple[FamilyKey, float, float, int, int]],
                         rankB: List[Tuple[FamilyKey, float, float, int, int]],
                         K: int) -> Tuple[float, int, int]:
    setA = set([x[0] for x in rankA[:K]])
    setB = set([x[0] for x in rankB[:K]])
    if not setA or not setB:
        return (float("nan"), 0, 0)
    inter = len(setA & setB)
    union = len(setA | setB)
    return (inter / union if union else 0.0, inter, union)


def bootstrap_family_enrichment_ci(cands: Sequence[Candidate],
                                   target_family: FamilyKey,
                                   top_fraction: float,
                                   n_boot: int,
                                   rng_seed: int,
                                   ent_bin_fine: float,
                                   leak_bin_fine: float,
                                   ent_step_family: float,
                                   leak_step_family: float,
                                   min_support: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(rng_seed)
    N = len(cands)
    if N == 0:
        return (float("nan"), float("nan"), float("nan"))

    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        sample = [cands[int(i)] for i in idx]
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

    if len(vals) < max(30, n_boot // 15):
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(vals, dtype=float)
    return (float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def print_basic_stats(cands: Sequence[Candidate], n_qubits: int) -> None:
    if not cands:
        print("No candidates retained.")
        return

    d = 2 ** n_qubits
    scores = np.array([c.score for c in cands], dtype=float)
    leak = np.array([c.leakage_raw for c in cands], dtype=float)
    dom = np.array([c.dom_count for c in cands], dtype=int)

    print(f"Retained candidates: {len(cands)}")
    print(f"Score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f}")
    print(f"Leakage_raw(mean/median/min)={leak.mean():.3f}/{np.median(leak):.3f}/{leak.min():.3f}")
    print(f"dom_count(min/median/max)={dom.min()}/{int(np.median(dom))}/{dom.max()} (d={d})")
    print("dom_count distribution:")
    for k in sorted(set(dom.tolist())):
        n = int(np.sum(dom == k))
        print(f"  dom_count={k}: {100.0*n/len(dom):.1f}% (n={n})")


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
    ap.add_argument("--output", type=str, default="v0_6_1_Final_patched_sigV2_output.txt")

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

    with Tee(args.output):
        print("=== v0_6.1 SigV2: Evidence Hygiene for Signature Families ===")
        print(f"Qubits: {n_qubits} (d={d}) | terms={num_terms} | seeds={len(seeds)} | t={t}")
        print(f"Nominal: prob_threshold={prob_threshold:.2f}, epsilon={epsilon:.2f}, ent_bin={ent_bin_fine:.2f}, leak_bin={leak_bin_fine:.2f}")
        print(f"Family: ent_step={ent_step_family:.2f}, leak_step={leak_step_family:.2f} | min_support={min_support} | topK={topK}")
        print(f"Output: {os.path.abspath(args.output)}\n")

        print("=== 1) Enrichment (family-based) + Bootstrap CI ===\n")
        c_full = run_scan(seeds, n_qubits, num_terms, epsilon, prob_threshold, ent_bin_fine, leak_bin_fine, t)
        print_basic_stats(c_full, n_qubits)
        print()

        if not c_full:
            print("No candidates retained. Exiting.")
            print("=== End of v0_6.1 SigV2 ===")
            return

        fine_overall: Dict[SigKey, int] = {}
        for x in c_full:
            fine_overall[x.sig] = fine_overall.get(x.sig, 0) + 1
        fam_overall = aggregate_counts_by_family(fine_overall, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)

        compact_lines: List[str] = []

        for frac in [0.01, 0.05]:
            print(f"Stable set = top {int(frac*100)}% by score")
            scores = np.array([x.score for x in c_full], dtype=float)
            cutoff = np.quantile(scores, 1.0 - frac)
            stable = [x for x in c_full if x.score >= cutoff]
            stable_frac = len(stable) / len(c_full)

            fine_stable: Dict[SigKey, int] = {}
            for x in stable:
                fine_stable[x.sig] = fine_stable.get(x.sig, 0) + 1
            fam_stable = aggregate_counts_by_family(fine_stable, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)

            ranked = rank_families(fam_overall, fam_stable, stable_frac, min_support=min_support, top_k=10)
            if not ranked:
                print("  No supported families to rank (try lowering --min_support).\n")
                continue

            for fam, enr, sr, cs, ca in ranked[:5]:
                mean, lo, hi = bootstrap_family_enrichment_ci(
                    c_full, fam, frac, int(args.bootstrap), 123,
                    ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family, min_support
                )
                ci = "bootstrap CI unavailable" if math.isnan(mean) else f"bootstrap={mean:.2f}x, CI95% [{lo:.2f}, {hi:.2f}]"
                print(f"  fam={fam}  point={enr:.2f}x  stable_rate={sr:.3f}  stable={cs}  overall={ca}  |  {ci}")

            for fam, enr, sr, cs, ca in ranked[:3]:
                compact_lines.append(
                    f"stable_set={int(frac*100)}% | fam={fam} | enrichment={enr:.2f}x | stable_rate={sr:.3f} | stable={cs} | overall={ca}"
                )
            print()

        print("=== 2) Hold-out Replication (family-based) ===")
        cA = run_scan(seeds_A, n_qubits, num_terms, epsilon, prob_threshold, ent_bin_fine, leak_bin_fine, t)
        cB = run_scan(seeds_B, n_qubits, num_terms, epsilon, prob_threshold, ent_bin_fine, leak_bin_fine, t)

        def ranks_for_half(cands: List[Candidate], stable_fraction: float) -> List[Tuple[FamilyKey, float, float, int, int]]:
            if not cands:
                return []
            scores = np.array([x.score for x in cands], dtype=float)
            cutoff = np.quantile(scores, 1.0 - stable_fraction)
            stable = [x for x in cands if x.score >= cutoff]
            stable_frac = len(stable) / len(cands)

            fo: Dict[SigKey, int] = {}
            fs: Dict[SigKey, int] = {}
            for x in cands:
                fo[x.sig] = fo.get(x.sig, 0) + 1
            for x in stable:
                fs[x.sig] = fs.get(x.sig, 0) + 1

            fam_o = aggregate_counts_by_family(fo, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
            fam_s = aggregate_counts_by_family(fs, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
            return rank_families(fam_o, fam_s, stable_frac, min_support=min_support, top_k=max(topK, 10))

        rankA = ranks_for_half(cA, stable_fraction=0.05)
        rankB = ranks_for_half(cB, stable_fraction=0.05)

        if not rankA or not rankB:
            print("Hold-out: insufficient supported families in one or both halves (try lowering --min_support).\n")
        else:
            jacc, inter, union = holdout_overlap_topK(rankA, rankB, K=topK)
            if math.isnan(jacc):
                print("Hold-out overlap: n/a (empty topK in one half after filtering)\n")
            else:
                print(f"Hold-out overlap on supported family keys: Jaccard={jacc:.3f} (inter={inter}, union={union})\n")
                compact_lines.append(f"holdout_topK={topK} | Jaccard={jacc:.3f} | inter={inter} | union={union}")

        print("=== 3) Compact Results Block (paste-ready) ===")
        for line in compact_lines:
            print("  " + line)
        print()

        print("=== End of v0_6.1 SigV2 ===")


if __name__ == "__main__":
    main()
