"""v0_7 (SigV2) — Baseline / Null-Model Calibration (PARAMETER-MATCHED)

Purpose
-------
This script is a *fair* baseline calibration companion to the v0_6.1 pipeline.
It runs the *same* candidate-extraction and scoring logic under three models:

  1) REAL          : eigenvectors from the Hamiltonian diagonalization
  2) BASIS_PERMUTE : a fixed computational-basis permutation applied to REAL eigenvectors
  3) HAAR_BASIS    : a Haar-random basis rotation applied to REAL eigenvectors

The goal is to characterize what “chance structure” looks like under identical
parameters, so that non-chance claims can be made conservatively.

Key fixes vs the failing local version
-------------------------------------
- No missing functions (NameError removed)
- UTF-8 safe logging (no Unicode arrows; robust Windows encoding)
- Console output is mirrored to an output .txt file

Run
---
python v0_7_baseline_sigV2_matched_FRESH.py --seeds 5000 --topK 50

Notes
-----
This is intentionally “non-AI”: it is baseline calibration only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# -----------------------------
# Logging (console + file)
# -----------------------------
class Tee:
    """Mirror stdout to a UTF-8 text file.

    Avoids Windows cp1252 failures by forcing UTF-8 and by not emitting
    non-ASCII characters from this script.
    """

    def __init__(self, filename: str):
        self._console = sys.stdout
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self._file = open(filename, "w", encoding="utf-8", errors="replace")

    def write(self, s: str) -> int:
        # stdout expects str
        if not isinstance(s, str):
            s = str(s)
        self._console.write(s)
        return self._file.write(s)

    def flush(self) -> None:
        try:
            self._console.flush()
        except Exception:
            pass
        try:
            if not self._file.closed:
                self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
        except Exception:
            pass


# -----------------------------
# Types
# -----------------------------
SigKey = Tuple[int, int, int, int]  # (dom_count, ent_i_bin, ent_j_bin, leak_bin)
FamilyKey = Tuple[int, int, int]    # (dom_count, ent_family, leak_family)


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


# -----------------------------
# Core math utilities
# -----------------------------
def shannon_entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def dominant_union_mask(phi_i: np.ndarray, phi_j: np.ndarray, p_dom_thresh: float) -> np.ndarray:
    """Boolean mask over computational basis where either state has prob >= threshold."""
    pi = np.abs(phi_i) ** 2
    pj = np.abs(phi_j) ** 2
    return (pi >= p_dom_thresh) | (pj >= p_dom_thresh)


def leakage_proxy(phi_i: np.ndarray, phi_j: np.ndarray, t: float) -> float:
    """A lightweight leakage proxy.

    We deliberately keep this simple and fast: define the candidate subspace
    S = span{phi_i, phi_j}. A stable candidate should largely remain inside S
    under its own Hamiltonian evolution; here we proxy stability by measuring
    how concentrated the candidate pair is in the computational basis.

    For baseline calibration, the *relative* comparison across models is
    the point; the proxy must be consistent, not perfect.
    """
    # Proxy: 1 - overlap of the two states' dominant support (encourages structured, low-support states)
    # In practice: if both states are concentrated on a small shared support, leakage_proxy is low.
    pi = np.abs(phi_i) ** 2
    pj = np.abs(phi_j) ** 2
    # Bhattacharyya overlap
    bc = float(np.sum(np.sqrt(pi * pj)))
    # Map to [0,1] with numerical safety
    bc = max(0.0, min(1.0, bc))
    return 1.0 - bc


def score_candidate(dom_count: int, d: int, ent_i: float, ent_j: float, leakage: float) -> float:
    """Score in [0,1], larger is 'more stable/interesting'.

    This follows the project’s intent: prefer smaller support (low dom_count)
    and lower leakage. Entropy is included indirectly via the signature,
    but not used aggressively in the score to avoid overfitting.
    """
    # Support term: dom_count=1 is best, dom_count=d is worst.
    support_term = 1.0 - ((dom_count - 1) / max(1, d - 1))
    leakage_term = 1.0 - max(0.0, min(1.0, leakage))
    # Conservative mixing
    return float(max(0.0, min(1.0, 0.5 * support_term + 0.5 * leakage_term)))


def bin_int(x: float, step: float) -> int:
    if step <= 0:
        raise ValueError("bin step must be > 0")
    return int(math.floor(x / step + 1e-12))


# -----------------------------
# Hamiltonian generation + eigensystem
# -----------------------------
def build_random_hamiltonian_sparse(n_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    paulis = []
    coeffs = []
    alphabet = ["I", "X", "Y", "Z"]

    for _ in range(num_terms):
        s = "".join(rng.choice(alphabet, size=n_qubits))
        paulis.append(s)
        coeffs.append(float(rng.uniform(-1.0, 1.0)))

    return SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=complex))


def eigensystem(H: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray]:
    # Dense exact diagonalization (n=3 is fine)
    mat = H.to_matrix()  # complex ndarray
    evals, evecs = np.linalg.eigh(mat)
    return evals.real.astype(float), evecs.astype(complex)


# -----------------------------
# Baseline models
# -----------------------------
def basis_permutation_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(d)
    P = np.zeros((d, d), dtype=complex)
    for i, j in enumerate(perm):
        P[i, j] = 1.0
    return P


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR decomposition."""
    X = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2)
    Q, R = np.linalg.qr(X)
    # Fix phases
    diag = np.diag(R)
    phases = diag / np.abs(diag)
    Q = Q * phases
    return Q


# -----------------------------
# Candidate discovery pipeline
# -----------------------------
def neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    """Only check adjacent eigenvalues (fast) in sorted order."""
    idx = np.argsort(evals)
    e = evals[idx]
    out = []
    for k in range(len(e) - 1):
        if abs(e[k + 1] - e[k]) < eps:
            out.append((int(idx[k]), int(idx[k + 1])))
    return out


def candidate_signature(dom_count: int,
                        ent_i: float,
                        ent_j: float,
                        leakage: float,
                        ent_bin: float,
                        leak_bin: float) -> SigKey:
    return (
        int(dom_count),
        bin_int(ent_i, ent_bin),
        bin_int(ent_j, ent_bin),
        bin_int(leakage, leak_bin),
    )


def aggregate_counts_by_family(fine_counts: Dict[SigKey, int],
                               ent_bin: float,
                               leak_bin: float,
                               ent_step: float,
                               leak_step: float) -> Dict[FamilyKey, int]:
    """Coarsen fine signature keys into family keys."""
    fam: Dict[FamilyKey, int] = {}
    for (dom, ei, ej, lb), cnt in fine_counts.items():
        ent_mean = 0.5 * ((ei + 0.5) * ent_bin + (ej + 0.5) * ent_bin)
        leak_val = (lb + 0.5) * leak_bin
        fam_key: FamilyKey = (
            int(dom),
            bin_int(ent_mean, ent_step),
            bin_int(leak_val, leak_step),
        )
        fam[fam_key] = fam.get(fam_key, 0) + int(cnt)
    return fam


def rank_families(fam_overall: Dict[FamilyKey, int],
                  fam_stable: Dict[FamilyKey, int],
                  stable_frac_eff: float,
                  min_support: int,
                  top_k: int) -> List[Tuple[FamilyKey, float, float, int, int]]:
    """Rank families by enrichment factor."""
    ranked: List[Tuple[FamilyKey, float, float, int, int]] = []
    total_overall = sum(fam_overall.values())
    total_stable = sum(fam_stable.values())
    if total_overall == 0 or total_stable == 0:
        return []

    for fam, overall_cnt in fam_overall.items():
        stable_cnt = fam_stable.get(fam, 0)
        if stable_cnt < min_support:
            continue
        # rates
        p_overall = overall_cnt / total_overall
        p_stable = stable_cnt / total_stable
        ef = (p_stable / p_overall) if p_overall > 0 else float("nan")
        # stable_rate relative to expected stable fraction
        # (how much of this family lands in stable set)
        stable_rate = stable_cnt / overall_cnt if overall_cnt > 0 else 0.0
        ranked.append((fam, float(ef), float(stable_rate), int(stable_cnt), int(overall_cnt)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def holdout_overlap_topK(rankA: List[Tuple[FamilyKey, float, float, int, int]],
                         rankB: List[Tuple[FamilyKey, float, float, int, int]],
                         K: int) -> Tuple[float, int, int]:
    setA = {x[0] for x in rankA[:K]}
    setB = {x[0] for x in rankB[:K]}
    if not setA or not setB:
        return float("nan"), 0, 0
    inter = len(setA & setB)
    union = len(setA | setB)
    return inter / union if union else float("nan"), inter, union


def bootstrap_ci_enrichment(cands: List[Candidate],
                            fam: FamilyKey,
                            stable_frac: float,
                            B: int,
                            rng_seed: int,
                            ent_bin: float,
                            leak_bin: float,
                            ent_step: float,
                            leak_step: float,
                            min_support: int) -> Tuple[float, float, float]:
    """Bootstrap CI for family enrichment factor."""
    if not cands:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(rng_seed)
    n = len(cands)
    efs: List[float] = []
    for _ in range(B):
        sample = [cands[int(i)] for i in rng.integers(0, n, size=n)]
        scores = np.array([c.score for c in sample], dtype=float)
        cutoff = np.quantile(scores, 1.0 - stable_frac)
        stable = [c for c in sample if c.score >= cutoff]
        if not stable:
            continue

        # fine counts
        fine_o: Dict[SigKey, int] = {}
        fine_s: Dict[SigKey, int] = {}
        for c in sample:
            fine_o[c.sig] = fine_o.get(c.sig, 0) + 1
        for c in stable:
            fine_s[c.sig] = fine_s.get(c.sig, 0) + 1

        fam_o = aggregate_counts_by_family(fine_o, ent_bin, leak_bin, ent_step, leak_step)
        fam_s = aggregate_counts_by_family(fine_s, ent_bin, leak_bin, ent_step, leak_step)
        ranked = rank_families(fam_o, fam_s, len(stable) / len(sample), min_support=min_support, top_k=1_000_000)
        # pull EF for target fam
        ef_val = None
        for f, ef, *_ in ranked:
            if f == fam:
                ef_val = ef
                break
        if ef_val is not None and math.isfinite(ef_val):
            efs.append(float(ef_val))

    if not efs:
        return float("nan"), float("nan"), float("nan")
    efs = sorted(efs)
    mean = float(np.mean(efs))
    lo = float(np.quantile(efs, 0.025))
    hi = float(np.quantile(efs, 0.975))
    return mean, lo, hi


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
            raise ValueError("Permutation matrix P is required for BASIS_PERMUTE")
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
        # Skip degenerate trivialities: full-support or empty-support
        if dom_count <= 0 or dom_count >= d:
            continue

        leak = float(leakage_proxy(phi_i, phi_j, t))
        score = float(score_candidate(dom_count, d, ent_i, ent_j, leak))
        sig = candidate_signature(dom_count, ent_i, ent_j, leak, ent_bin, leak_bin)
        out.append(Candidate(seed=seed, model=model, pair=(int(i), int(j)), dE=dE,
                             dom_count=dom_count, ent_i=ent_i, ent_j=ent_j,
                             leakage_raw=leak, score=score, sig=sig))
    return out


def run_model_scan(model: str,
                   seeds: Iterable[int],
                   n_qubits: int,
                   num_terms: int,
                   eps: float,
                   p_dom_thresh: float,
                   ent_bin: float,
                   leak_bin: float,
                   t: float,
                   P: Optional[np.ndarray],
                   rng_basis: np.random.Generator,
                   progress_every: int = 500) -> List[Candidate]:
    out: List[Candidate] = []
    for idx, seed in enumerate(seeds, start=1):
        out.extend(
            candidates_for_seed(
                seed=seed,
                model=model,
                n_qubits=n_qubits,
                num_terms=num_terms,
                eps=eps,
                p_dom_thresh=p_dom_thresh,
                ent_bin=ent_bin,
                leak_bin=leak_bin,
                t=t,
                P=P,
                rng_basis=rng_basis,
            )
        )
        if progress_every and (idx % progress_every == 0):
            print(f"  progress: {idx} / {len(list(seeds))} seeds processed")
    return out


def basic_stats(cands: List[Candidate]) -> str:
    if not cands:
        return "candidates=0"
    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leakage_raw for c in cands], dtype=float)
    doms = np.array([c.dom_count for c in cands], dtype=int)
    return (
        f"candidates={len(cands)} | "
        f"score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f} | "
        f"leakage(mean/median/min)={leaks.mean():.3f}/{np.median(leaks):.3f}/{leaks.min():.3f} | "
        f"dom_count(median)={int(np.median(doms))}"
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--num_terms", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--seed_offset", type=int, default=0)

    # Parameter-matched defaults (v0_6.1-style)
    ap.add_argument("--eps", type=float, default=0.050)
    ap.add_argument("--p_dom_thresh", type=float, default=0.250)
    ap.add_argument("--ent_bin", type=float, default=0.100)
    ap.add_argument("--leak_bin", type=float, default=0.050)
    ap.add_argument("--family_ent_step", type=float, default=0.100)
    ap.add_argument("--family_leak_step", type=float, default=0.050)
    ap.add_argument("--t", type=float, default=1.0)

    # Evidence layer
    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--min_support", type=int, default=10)
    ap.add_argument("--topK", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=200)

    ap.add_argument("--output", type=str, default="v0_7_baseline_sigV2_matched_FRESH1_output.txt")
    args = ap.parse_args()

    n_qubits = int(args.n_qubits)
    d = 2 ** n_qubits
    seeds = list(range(int(args.seed_offset), int(args.seed_offset) + int(args.seeds)))
    half = len(seeds) // 2
    seeds_A, seeds_B = seeds[:half], seeds[half:]

    rng_perm = np.random.default_rng(202512)
    P = basis_permutation_matrix(d, rng_perm)
    rng_basis = np.random.default_rng(9001)

    models = ["REAL", "BASIS_PERMUTE", "HAAR_BASIS"]

    tee = Tee(args.output)
    sys.stdout = tee
    try:
        print("=== v0_7 SigV2: Baseline / Null-Model Calibration (MATCHED) ===")
        print(f"Qubits: {n_qubits} (d={d}) | terms={int(args.num_terms)} | seeds={len(seeds)}")
        print(f"Near-degenerate neighbor eps={float(args.eps):.3f}")
        print(f"Dominant threshold={float(args.p_dom_thresh):.3f} | ent_bin={float(args.ent_bin):.3f} | leak_bin={float(args.leak_bin):.3f}")
        print(f"Family binning: ent_step={float(args.family_ent_step):.3f} | leak_step={float(args.family_leak_step):.3f}")
        print(f"Evidence: stable_frac={float(args.stable_frac):.3f} | min_support={int(args.min_support)} | topK={int(args.topK)} | bootstrap={int(args.bootstrap)}")
        print(f"Output: {os.path.abspath(args.output)}\n")

        t0 = time.time()

        for model in models:
            print("----------------------------------------------")
            print(f"Model: {model}")
            t_model = time.time()

            cands = []
            # Use explicit loop here to avoid recomputing list(seeds) for progress
            for idx, seed in enumerate(seeds, start=1):
                cands.extend(
                    candidates_for_seed(
                        seed=seed,
                        model=model,
                        n_qubits=n_qubits,
                        num_terms=int(args.num_terms),
                        eps=float(args.eps),
                        p_dom_thresh=float(args.p_dom_thresh),
                        ent_bin=float(args.ent_bin),
                        leak_bin=float(args.leak_bin),
                        t=float(args.t),
                        P=P if model == "BASIS_PERMUTE" else None,
                        rng_basis=rng_basis,
                    )
                )
                if idx % 500 == 0:
                    print(f"  progress: {idx}/{len(seeds)} seeds")

            print(basic_stats(cands))
            if not cands:
                print("No candidates. Consider increasing --eps or --seeds.\n")
                continue

            scores = np.array([c.score for c in cands], dtype=float)
            cutoff = np.quantile(scores, 1.0 - float(args.stable_frac))
            stable = [c for c in cands if c.score >= cutoff]
            stable_frac_eff = len(stable) / len(cands)
            print(f"Stable set size: {len(stable)} (effective stable_frac={stable_frac_eff:.3f})")

            fine_overall: Dict[SigKey, int] = {}
            fine_stable: Dict[SigKey, int] = {}
            for c in cands:
                fine_overall[c.sig] = fine_overall.get(c.sig, 0) + 1
            for c in stable:
                fine_stable[c.sig] = fine_stable.get(c.sig, 0) + 1

            fam_overall = aggregate_counts_by_family(
                fine_overall,
                float(args.ent_bin),
                float(args.leak_bin),
                float(args.family_ent_step),
                float(args.family_leak_step),
            )
            fam_stable = aggregate_counts_by_family(
                fine_stable,
                float(args.ent_bin),
                float(args.leak_bin),
                float(args.family_ent_step),
                float(args.family_leak_step),
            )

            ranked = rank_families(
                fam_overall,
                fam_stable,
                stable_frac_eff,
                min_support=int(args.min_support),
                top_k=max(10, int(args.topK)),
            )

            if not ranked:
                print("Top enriched signature families: (none passed min_support)")
            else:
                print("Top enriched signature families (family_key -> EF, stable_rate, counts, bootstrap CI):")
                for fam, ef, sr, cs, ca in ranked[: int(args.topK)]:
                    mean, lo, hi = bootstrap_ci_enrichment(
                        cands,
                        fam,
                        float(args.stable_frac),
                        int(args.bootstrap),
                        1234,
                        float(args.ent_bin),
                        float(args.leak_bin),
                        float(args.family_ent_step),
                        float(args.family_leak_step),
                        int(args.min_support),
                    )
                    ci_txt = "CI: n/a" if not math.isfinite(mean) else f"bootstrap={mean:.2f}x, CI95% [{lo:.2f}, {hi:.2f}]"
                    print(f"  fam={fam}  EF={ef:.2f}x  stable_rate={sr:.3f}  stable={cs}  overall={ca}  |  {ci_txt}")

            # Hold-out replication
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
                fam_o = aggregate_counts_by_family(fo, float(args.ent_bin), float(args.leak_bin), float(args.family_ent_step), float(args.family_leak_step))
                fam_s = aggregate_counts_by_family(fs, float(args.ent_bin), float(args.leak_bin), float(args.family_ent_step), float(args.family_leak_step))
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
        print("  - If REAL shows stronger enrichment and better hold-out overlap than HAAR_BASIS, that supports non-chance structure.")
        print("  - BASIS_PERMUTE is a label-invariance sanity check; large deviations vs REAL suggest basis-sensitive definitions.")
        print("  - If REAL and HAAR_BASIS look similar, treat observed signatures as likely chance structure under this pipeline.")
        print("\n=== End of v0_7 SigV2 ===")

    finally:
        sys.stdout = tee._console
        tee.close()


if __name__ == "__main__":
    main()
