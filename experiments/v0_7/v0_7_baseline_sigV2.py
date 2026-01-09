#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7_baseline_sigV2.py — Baseline / Null-Model Calibration for v0_6–v0_6.1 (SigV2)

Goal
----
Calibrate what "chance structure" looks like under the *same* signature pipeline
used in v0_6 / v0_6.1, using basis-randomization null models.

Models
------
- REAL:
    Use the eigenbasis of each randomly generated Pauli-sum Hamiltonian.
- BASIS_PERMUTE:
    Apply a fixed random permutation of the computational basis (label-invariance check).
- HAAR_BASIS:
    Replace the eigenbasis with a random Haar-like basis while keeping eigenvalues (strong null).

For each near-degenerate neighbor pair (i,i+1), we compute:
  - dom_count: dominant basis union size (threshold on |amp|^2)
  - entropy of each eigenvector (bits)
  - leakage_raw: mass leaving the dominant support under exact spectral evolution
  - score: baseline-normalized leakage

Evidence layer (same style as v0_6.1 SigV2):
  - family-level enrichment in stable set (top fraction by score)
  - bootstrap CI for enrichment
  - hold-out replication (topK family overlap between seed halves)

Output:
  v0_7_baseline_sigV2_output.txt (UTF-8)

Run:
  python v0_7_baseline_sigV2.py --seeds 5000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp
except Exception:
    raise RuntimeError("qiskit is required (SparsePauliOp). Install: pip install qiskit")


# -----------------------------
# Logging helper (UTF-8)
# -----------------------------
class Tee:
    def __init__(self, path: str):
        self.path = path
        self._console = sys.stdout
        self._file = open(path, "w", encoding="utf-8", errors="replace", buffering=1)

    def write(self, obj):
        s = str(obj)
        self._console.write(s)
        self._file.write(s)

    def flush(self):
        try:
            self._console.flush()
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


# -----------------------------
# Utilities
# -----------------------------
def shannon_entropy_bits(p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(psi)
    return psi if nrm == 0 else psi / nrm


def score_interestingness(leakage_raw: float, dom_count: int, d: int) -> float:
    baseline = 1.0 - (dom_count / float(d))
    if baseline <= 1e-12:
        return 0.0
    ratio = max(0.0, min(1.0, leakage_raw / baseline))
    return 1.0 - ratio


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph
    return q


def basis_permutation_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(dim)
    p = np.zeros((dim, dim), dtype=complex)
    for i, j in enumerate(perm):
        p[j, i] = 1.0
    return p


# -----------------------------
# Hamiltonian generator
# -----------------------------
PAULI_CHARS = ["I", "X", "Y", "Z"]


def random_pauli_string(n_qubits: int, rng: np.random.Generator) -> str:
    while True:
        s = "".join(rng.choice(PAULI_CHARS) for _ in range(n_qubits))
        if any(c != "I" for c in s):
            return s


def build_random_hamiltonian_sparse(n_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    labels, coeffs = [], []
    for _ in range(num_terms):
        labels.append(random_pauli_string(n_qubits, rng))
        coeffs.append(complex(rng.uniform(-1.0, 1.0)))
    return SparsePauliOp(labels, coeffs=np.asarray(coeffs, dtype=complex))


def eigensystem(H: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(H.to_matrix(), dtype=complex)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals.real)
    return vals.real[order], vecs[:, order]


def neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(len(evals) - 1):
        if abs(evals[i + 1] - evals[i]) < eps:
            pairs.append((i, i + 1))
    return pairs


def evolve_exact(evals: np.ndarray, evecs: np.ndarray, t: float, psi0: np.ndarray) -> np.ndarray:
    phases = np.exp(-1j * evals * t)
    a = evecs.conj().T @ psi0
    return evecs @ (phases * a)


def dominant_union_mask(v_i: np.ndarray, v_j: np.ndarray, p_thresh: float) -> np.ndarray:
    pi = np.abs(v_i) ** 2
    pj = np.abs(v_j) ** 2
    return (pi >= p_thresh) | (pj >= p_thresh)


# -----------------------------
# Signature keys
# -----------------------------
SigKey = Tuple[int, int, int, int]
FamilyKey = Tuple[int, int, int, int]


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
# Candidate
# -----------------------------
@dataclass(frozen=True)
class Candidate:
    seed: int
    model: str
    pair: Tuple[int, int]
    dE: float
    dom_count: int
    ent_i: float
    ent_j: float
    leakage_raw: float
    score: float
    sig: SigKey


def candidates_for_seed(seed: int,
                        model: str,
                        n_qubits: int,
                        num_terms: int,
                        eps: float,
                        p_dom_thresh: float,
                        ent_bin: float,
                        leak_bin: float,
                        t: float,
                        P: Optional[np.ndarray],
                        rng_basis: np.random.Generator) -> List[Candidate]:
    H = build_random_hamiltonian_sparse(n_qubits=n_qubits, num_terms=num_terms, seed=seed)
    evals, evecs = eigensystem(H)
    d = 2 ** n_qubits

    if model == "REAL":
        V = evecs
    elif model == "BASIS_PERMUTE":
        if P is None:
            raise ValueError("Permutation matrix P is required for BASIS_PERMUTE.")
        V = P @ evecs
    elif model == "HAAR_BASIS":
        U = random_unitary(d, rng_basis)
        V = U @ evecs
    else:
        raise ValueError(f"Unknown model: {model}")

    pairs = neighbor_pairs(evals, eps)
    if not pairs:
        return []

    out: List[Candidate] = []
    for i, j in pairs:
        dE = float(abs(evals[j] - evals[i]))
        phi_i = V[:, i]
        phi_j = V[:, j]

        pi = np.abs(phi_i) ** 2
        pj = np.abs(phi_j) ** 2
        ent_i = shannon_entropy_bits(pi)
        ent_j = shannon_entropy_bits(pj)

        mask = dominant_union_mask(phi_i, phi_j, p_dom_thresh)
        dom_count = int(np.sum(mask))
        if dom_count <= 0 or dom_count >= d:
            continue

        psi0 = np.zeros((d,), dtype=complex)
        psi0[mask] = 1.0
        psi0 = normalize_state(psi0)

        psi_t = evolve_exact(evals, V, t, psi0)
        prob_t = np.abs(psi_t) ** 2
        stay = float(np.sum(prob_t[mask]))
        leakage_raw = max(0.0, min(1.0, 1.0 - stay))

        score = score_interestingness(leakage_raw, dom_count, d)
        sig = signature_key(dom_count, ent_i, ent_j, leakage_raw, ent_bin, leak_bin)

        out.append(Candidate(seed, model, (int(i), int(j)), dE, dom_count, float(ent_i), float(ent_j),
                             float(leakage_raw), float(score), sig))
    return out


def run_model_scan(model: str,
                   seeds: Sequence[int],
                   n_qubits: int,
                   num_terms: int,
                   eps: float,
                   p_dom_thresh: float,
                   ent_bin: float,
                   leak_bin: float,
                   t: float,
                   P: Optional[np.ndarray],
                   rng_basis: np.random.Generator) -> List[Candidate]:
    all_c: List[Candidate] = []
    for sd in seeds:
        all_c.extend(candidates_for_seed(sd, model, n_qubits, num_terms, eps, p_dom_thresh, ent_bin, leak_bin, t, P, rng_basis))
    return all_c


# -----------------------------
# Evidence layer
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
        enrichment = stable_rate / stable_frac if stable_frac > 0 else float("nan")
        ranked.append((fam, float(enrichment), float(stable_rate), int(cs), int(ca)))
    ranked.sort(key=lambda x: (-x[1], -x[2], -x[4], x[0]))
    return ranked[:top_k]


def bootstrap_family_enrichment_ci(cands: Sequence[Candidate],
                                   target_family: FamilyKey,
                                   stable_frac: float,
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
        cutoff = np.quantile(scores, 1.0 - stable_frac)
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

        stable_frac_eff = len(stable) / len(sample)
        ca = fam_overall.get(target_family, 0)
        cs = fam_stable.get(target_family, 0)
        stable_rate = cs / ca if ca > 0 else 0.0
        enrichment = stable_rate / stable_frac_eff if stable_frac_eff > 0 else float("nan")
        if math.isfinite(enrichment):
            vals.append(float(enrichment))

    if len(vals) < max(30, n_boot // 15):
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(vals, dtype=float)
    return (float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


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


def basic_stats(cands: Sequence[Candidate]) -> str:
    if not cands:
        return "No candidates."
    scores = np.array([c.score for c in cands], dtype=float)
    leak = np.array([c.leakage_raw for c in cands], dtype=float)
    dom = np.array([c.dom_count for c in cands], dtype=int)
    return (
        f"candidates={len(cands)} | score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f} | "
        f"leakage(mean/median/min)={leak.mean():.3f}/{np.median(leak):.3f}/{leak.min():.3f} | "
        f"dom_count(median)={int(np.median(dom))}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--num_terms", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--p_dom_thresh", type=float, default=0.05)
    ap.add_argument("--t", type=float, default=1.0)

    ap.add_argument("--stable_frac", type=float, default=0.05)
    ap.add_argument("--min_support", type=int, default=20)
    ap.add_argument("--topK", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=400)

    ap.add_argument("--ent_bin", type=float, default=0.25)
    ap.add_argument("--leak_bin", type=float, default=0.01)
    ap.add_argument("--family_ent_step", type=float, default=0.25)
    ap.add_argument("--family_leak_step", type=float, default=0.01)

    ap.add_argument("--output", type=str, default="v0_7_baseline_sigV2_output.txt")
    args = ap.parse_args()

    n_qubits = int(args.n_qubits)
    d = 2 ** n_qubits
    seeds = list(range(int(args.seed_offset), int(args.seed_offset) + int(args.seeds)))
    half = len(seeds) // 2
    seeds_A, seeds_B = seeds[:half], seeds[half:]

    ent_bin_fine = float(args.ent_bin)
    leak_bin_fine = float(args.leak_bin)
    ent_step_family = float(args.family_ent_step)
    leak_step_family = float(args.family_leak_step)

    rng_perm = np.random.default_rng(202512)
    P = basis_permutation_matrix(d, rng_perm)
    rng_basis = np.random.default_rng(9001)

    models = ["REAL", "BASIS_PERMUTE", "HAAR_BASIS"]

    tee = Tee(args.output)
    sys.stdout = tee
    try:
        print("=== v0_7 SigV2: Baseline / Null-Model Calibration ===")
        print(f"Qubits: {n_qubits} (d={d}) | terms={int(args.num_terms)} | seeds={len(seeds)}")
        print(f"Near-degenerate neighbor eps={float(args.eps):.3f}")
        print(f"Dominant threshold={float(args.p_dom_thresh):.3f} | ent_bin={ent_bin_fine:.3f} | leak_bin={leak_bin_fine:.3f}")
        print(f"Family binning: ent_step={ent_step_family:.3f} | leak_step={leak_step_family:.3f}")
        print(f"Evidence: stable_frac={float(args.stable_frac):.3f} | min_support={int(args.min_support)} | topK={int(args.topK)} | bootstrap={int(args.bootstrap)}")
        print(f"Output: {os.path.abspath(args.output)}\n")

        t0 = time.time()

        for model in models:
            print("----------------------------------------------")
            print(f"Model: {model}")
            t_model = time.time()

            cands = run_model_scan(
                model=model,
                seeds=seeds,
                n_qubits=n_qubits,
                num_terms=int(args.num_terms),
                eps=float(args.eps),
                p_dom_thresh=float(args.p_dom_thresh),
                ent_bin=ent_bin_fine,
                leak_bin=leak_bin_fine,
                t=float(args.t),
                P=P if model == "BASIS_PERMUTE" else None,
                rng_basis=rng_basis,
            )

            print(basic_stats(cands))
            if not cands:
                print("No candidates. Consider increasing --eps or --seeds.\n")
                continue

            scores = np.array([c.score for c in cands], dtype=float)
            cutoff = np.quantile(scores, 1.0 - float(args.stable_frac))
            stable = [c for c in cands if c.score >= cutoff]
            stable_frac_eff = len(stable) / len(cands)

            fine_overall: Dict[SigKey, int] = {}
            fine_stable: Dict[SigKey, int] = {}
            for c in cands:
                fine_overall[c.sig] = fine_overall.get(c.sig, 0) + 1
            for c in stable:
                fine_stable[c.sig] = fine_stable.get(c.sig, 0) + 1

            fam_overall = aggregate_counts_by_family(fine_overall, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
            fam_stable = aggregate_counts_by_family(fine_stable, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)

            ranked = rank_families(
                fam_overall, fam_stable, stable_frac_eff,
                min_support=int(args.min_support),
                top_k=max(10, int(args.topK))
            )

            print(f"Stable set size: {len(stable)} (effective stable_frac={stable_frac_eff:.3f})")

            if not ranked:
                print("Top enriched signature families: (none passed min_support)")
            else:
                print("Top enriched signature families (family_key -> EF, stable_rate, counts, bootstrap CI):")
                for fam, ef, sr, cs, ca in ranked[:int(args.topK)]:
                    mean, lo, hi = bootstrap_family_enrichment_ci(
                        cands, fam, float(args.stable_frac), int(args.bootstrap), 1234,
                        ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family, int(args.min_support)
                    )
                    ci_txt = "CI: n/a" if (math.isnan(mean)) else f"bootstrap={mean:.2f}x, CI95% [{lo:.2f}, {hi:.2f}]"
                    print(f"  fam={fam}  EF={ef:.2f}x  stable_rate={sr:.3f}  stable={cs}  overall={ca}  |  {ci_txt}")

            # Hold-out replication (family-based)
            seedsA_set = set(seeds_A)
            seedsB_set = set(seeds_B)
            cA = [c for c in cands if c.seed in seedsA_set]
            cB = [c for c in cands if c.seed in seedsB_set]

            def ranks_for_half(c_half: List[Candidate]) -> List[Tuple[FamilyKey, float, float, int, int]]:
                if not c_half:
                    return []
                sc = np.array([c.score for c in c_half], dtype=float)
                cut = np.quantile(sc, 1.0 - float(args.stable_frac))
                st = [c for c in c_half if c.score >= cut]
                st_frac = len(st) / len(c_half) if c_half else 0.0

                fo: Dict[SigKey, int] = {}
                fs: Dict[SigKey, int] = {}
                for c in c_half:
                    fo[c.sig] = fo.get(c.sig, 0) + 1
                for c in st:
                    fs[c.sig] = fs.get(c.sig, 0) + 1
                fam_o = aggregate_counts_by_family(fo, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
                fam_s = aggregate_counts_by_family(fs, ent_bin_fine, leak_bin_fine, ent_step_family, leak_step_family)
                return rank_families(fam_o, fam_s, st_frac, min_support=int(args.min_support), top_k=max(10, int(args.topK)))

            rankA = ranks_for_half(cA)
            rankB = ranks_for_half(cB)
            jacc, inter, union = holdout_overlap_topK(rankA, rankB, K=int(args.topK))
            if math.isnan(jacc):
                print("Hold-out replication: n/a (empty topK after filtering in one half)")
            else:
                print(f"Hold-out replication (family keys): Jaccard={jacc:.3f} (inter={inter}, union={union})")

            dt = time.time() - t_model
            print(f"Model runtime: {dt:.1f} s\n")

        dt_total = time.time() - t0
        print("----------------------------------------------")
        print(f"Total runtime: {dt_total:.1f} s\n")

        print("Interpretation guide (SigV2):")
        print("  - REAL should show stronger and more consistent enrichment than HAAR_BASIS if non-chance structure exists.")
        print("  - BASIS_PERMUTE is a label-invariance check; large differences vs REAL suggest basis-sensitive definitions.")
        print("  - If REAL and HAAR_BASIS look similar, the pipeline is likely measuring chance structure.")
        print("\n=== End of v0_7 SigV2 ===")

    finally:
        sys.stdout = tee._console
        tee.close()


if __name__ == "__main__":
    main()
