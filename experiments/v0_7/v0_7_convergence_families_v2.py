
"""
v0_7_convergence_families_v2.py

Goal
-----
A "clean" convergence + calibration runner that addresses the observed issue:
signature families acting like non-replicable labels.

Implements (without changing the core scientific logic):
1) Coarser / merged family keys (fewer bins; fewer degrees of freedom)
2) Support-first ranking (stable_count + overall_count) then enrichment
3) Cross-model enrichment with smoothing + minimum-count filters (readable ratios)
4) Distribution-level effect sizes (REAL vs NULL) with bootstrap uncertainty
5) 3 independent batches with different seed offsets + Top-K overlap metrics

Models
------
REAL: eigenvectors from diagonalizing H
NULL_HAAR_BASIS: same spectrum, but eigenbasis replaced with a random Haar unitary basis

Dependencies
------------
- numpy
- qiskit (for SparsePauliOp / Pauli)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp, Pauli
except Exception as e:
    raise RuntimeError(
        "This script requires qiskit. Install e.g. `pip install qiskit`.\n"
        f"Import error: {e}"
    )


# -----------------------------
# I/O utilities
# -----------------------------
class Tee:
    """Write to stdout + a UTF-8 text file (avoids cp1252 issues on Windows)."""

    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout

    def write(self, obj):
        self._stdout.write(obj)
        self.file.write(obj)

    def flush(self):
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# -----------------------------
# Core math helpers
# -----------------------------
def shannon_entropy_base2(p: np.ndarray) -> float:
    """Shannon entropy in bits, with p assumed to sum to ~1."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary via QR decomposition of a complex Ginibre matrix."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph
    return q


def build_random_hamiltonian_sparse(n_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    """Random Pauli-sum Hamiltonian: H = sum_k c_k P_k."""
    rng = np.random.default_rng(seed)
    paulis = []
    coeffs = []
    for _ in range(num_terms):
        s = "".join(rng.choice(["I", "X", "Y", "Z"], size=n_qubits))
        paulis.append(Pauli(s))
        coeffs.append(float(rng.uniform(-1.0, 1.0)))
    return SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=np.complex128))


def eigen_decompose(H: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray]:
    """Return (evals_sorted, evecs_sorted_as_columns)."""
    mat = H.to_matrix()
    evals, evecs = np.linalg.eigh(mat)
    idx = np.argsort(evals.real)
    evals = evals[idx].real
    evecs = evecs[:, idx]
    return evals, evecs


def find_near_degenerate_neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    """Neighbor-only scan over sorted eigenvalues. Returns list of (i, i+1, deltaE)."""
    pairs = []
    for i in range(len(evals) - 1):
        dE = float(abs(evals[i + 1] - evals[i]))
        if dE < eps:
            pairs.append((i, i + 1, dE))
    return pairs


def support_from_evecs(evecs: np.ndarray, i: int, j: int, dom_threshold: float) -> List[int]:
    """
    Support basis indices for eigenvectors i and j: union of basis states
    with probability >= dom_threshold, fallback to top-2 if empty.
    """
    vi = evecs[:, i]
    vj = evecs[:, j]
    pi = np.abs(vi) ** 2
    pj = np.abs(vj) ** 2

    idx = set(np.where(pi >= dom_threshold)[0].tolist() + np.where(pj >= dom_threshold)[0].tolist())
    if len(idx) == 0:
        comb = pi + pj
        top = np.argsort(comb)[::-1][:2]
        idx = set(top.tolist())
    return sorted(idx)


def leakage_proxy_from_support(
    evals: np.ndarray,
    evecs: np.ndarray,
    support: List[int],
    times: List[float],
) -> float:
    """
    Leakage proxy:
      - Prepare |psi0> = uniform superposition over computational-basis support states
      - Evolve under H via spectral decomposition (exact, cheap in d=8):
            U(t) = V diag(exp(-i E t)) V^\dagger
      - Leakage(t) = 1 - sum_{k in support} |psi_t[k]|^2
      - Return mean leakage over provided times
    """
    dim = evecs.shape[0]
    psi0 = np.zeros(dim, dtype=np.complex128)
    amp = 1.0 / np.sqrt(len(support))
    for k in support:
        psi0[k] = amp

    Vh_psi0 = np.conjugate(evecs).T @ psi0
    leakages = []
    for t in times:
        phases = np.exp(-1j * evals * t)
        psi_t = evecs @ (phases * Vh_psi0)
        p = np.abs(psi_t) ** 2
        stay = float(np.sum(p[support]))
        leakages.append(max(0.0, 1.0 - stay))
    return float(np.mean(leakages))


def candidate_features(
    evals: np.ndarray,
    evecs: np.ndarray,
    i: int,
    j: int,
    dom_threshold: float,
    times: List[float],
) -> Tuple[int, float, float]:
    """
    Return (dom_count, ent_med, leakage)
      dom_count: size of support
      ent_med: median entropy of eigenvectors i,j in computational basis
      leakage: proxy leakage from support under time evolution
    """
    vi = evecs[:, i]
    vj = evecs[:, j]
    pi = np.abs(vi) ** 2
    pj = np.abs(vj) ** 2
    ent_i = shannon_entropy_base2(pi)
    ent_j = shannon_entropy_base2(pj)
    ent_med = float(np.median([ent_i, ent_j]))

    support = support_from_evecs(evecs, i, j, dom_threshold)
    dom_count = int(len(support))
    leakage = leakage_proxy_from_support(evals, evecs, support, times)
    return dom_count, ent_med, leakage


def family_key_coarse(dom_count: int, ent_med: float, leakage: float, ent_step: float, leak_step: float) -> Tuple[int, int, int]:
    """Coarse, merged signature family key: (dom_count_capped, ent_bin, leak_bin)."""
    dom_cap = int(min(dom_count, 8))
    ent_bin = int(np.floor(ent_med / ent_step + 1e-9))
    leak_bin = int(np.floor(leakage / leak_step + 1e-9))
    return (dom_cap, ent_bin, leak_bin)


def score_interestingness(dom_count: int, leakage: float) -> float:
    """Simple score used to define a stable set (low leakage, small support)."""
    stability = max(0.0, 1.0 - leakage)
    size_penalty = 1.0 / (1.0 + dom_count)
    return float(stability * size_penalty)


def enrichment_table(
    keys: List[Tuple[int, int, int]],
    scores: np.ndarray,
    stable_frac: float,
    topK: int,
    alpha: float,
    min_overall: int,
    min_stable: int,
) -> List[Tuple[Tuple[int, int, int], int, int, float]]:
    """
    Within-model enrichment:
      enr = P(sig | stable) / P(sig | overall)
    with Laplace smoothing (alpha) to keep values finite.

    Ranking (support-first):
      stable_count desc, overall_count desc, enrichment desc
    """
    n = len(keys)
    if n == 0:
        return []

    k_stable = max(1, int(np.floor(stable_frac * n)))
    idx_sorted = np.argsort(scores)[::-1]
    stable_idx = set(idx_sorted[:k_stable].tolist())

    overall: Dict[Tuple[int, int, int], int] = {}
    stable: Dict[Tuple[int, int, int], int] = {}
    for t, sig in enumerate(keys):
        overall[sig] = overall.get(sig, 0) + 1
        if t in stable_idx:
            stable[sig] = stable.get(sig, 0) + 1

    total_overall = n
    total_stable = k_stable

    rows = []
    for sig, ov in overall.items():
        st = stable.get(sig, 0)
        if ov < min_overall or st < min_stable:
            continue

        p_st = (st + alpha) / (total_stable + alpha)
        p_ov = (ov + alpha) / (total_overall + alpha)
        enr = float(p_st / p_ov)
        rows.append((sig, ov, st, enr))

    rows.sort(key=lambda r: (r[2], r[1], r[3]), reverse=True)
    return rows[:topK]


def overlap_jaccard(a: List[Tuple[int, int, int]], b: List[Tuple[int, int, int]]) -> Tuple[int, float]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter, (0.0 if union == 0 else inter / union)


# -----------------------------
# Batch runner
# -----------------------------
@dataclass
class BatchResult:
    top_fams_real: List[Tuple[int, int, int]]
    top_fams_null: List[Tuple[int, int, int]]
    median_entbin_real: float
    median_entbin_null: float
    median_entbin_diff: float
    ci_entbin_diff: Tuple[float, float]


def run_batch(batch_id: int, seed_offset: int, args: argparse.Namespace) -> BatchResult:
    t0 = time.time()

    keys_real: List[Tuple[int, int, int]] = []
    keys_null: List[Tuple[int, int, int]] = []
    scores_real: List[float] = []
    scores_null: List[float] = []
    entbin_real: List[int] = []
    entbin_null: List[int] = []

    for s in range(args.seeds):
        seed = int(seed_offset + s)
        H = build_random_hamiltonian_sparse(args.qubits, args.terms, seed)
        evals, V_real = eigen_decompose(H)
        pairs = find_near_degenerate_neighbor_pairs(evals, args.eps)

        rng_null = np.random.default_rng(seed + 10_000_000 + 97 * batch_id)
        V_null = haar_unitary(2 ** args.qubits, rng_null)

        for (i, j, _dE) in pairs:
            dom_r, ent_med_r, leak_r = candidate_features(evals, V_real, i, j, args.dom_threshold, args.times)
            dom_n, ent_med_n, leak_n = candidate_features(evals, V_null, i, j, args.dom_threshold, args.times)

            sig_r = family_key_coarse(dom_r, ent_med_r, leak_r, args.ent_step, args.leak_step)
            sig_n = family_key_coarse(dom_n, ent_med_n, leak_n, args.ent_step, args.leak_step)

            keys_real.append(sig_r)
            keys_null.append(sig_n)

            scores_real.append(score_interestingness(dom_r, leak_r))
            scores_null.append(score_interestingness(dom_n, leak_n))

            entbin_real.append(sig_r[1])
            entbin_null.append(sig_n[1])

    scores_real_arr = np.array(scores_real, dtype=float)
    scores_null_arr = np.array(scores_null, dtype=float)
    entbin_real_arr = np.array(entbin_real, dtype=float)
    entbin_null_arr = np.array(entbin_null, dtype=float)

    top_real_rows = enrichment_table(
        keys_real, scores_real_arr, args.stable_frac, args.topK, args.alpha, args.min_overall, args.min_stable
    )
    top_null_rows = enrichment_table(
        keys_null, scores_null_arr, args.stable_frac, args.topK, args.alpha, args.min_overall, args.min_stable
    )

    top_fams_real = [r[0] for r in top_real_rows]
    top_fams_null = [r[0] for r in top_null_rows]

    med_r = float(np.median(entbin_real_arr)) if entbin_real_arr.size else float("nan")
    med_n = float(np.median(entbin_null_arr)) if entbin_null_arr.size else float("nan")
    diff = float(med_r - med_n)

    # Bootstrap CI for diff in medians (paired by index)
    B = args.bootstrap
    if entbin_real_arr.size and entbin_null_arr.size:
        n = min(entbin_real_arr.size, entbin_null_arr.size)
        er = entbin_real_arr[:n]
        en = entbin_null_arr[:n]
        diffs = []
        rng = np.random.default_rng(args.bootstrap_seed + 1000 * batch_id)
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            diffs.append(float(np.median(er[idx]) - np.median(en[idx])))
        diffs.sort()
        ci = (diffs[int(0.025 * B)], diffs[int(0.975 * B)])
    else:
        ci = (float("nan"), float("nan"))

    elapsed = time.time() - t0

    print(f"\n--- Batch {batch_id+1}/{args.batches} ---")
    print(f"seed_offset={seed_offset} | candidates_real={len(keys_real)} | candidates_null={len(keys_null)} | elapsed={elapsed:.2f}s")
    print(f"Median(ent_bin): REAL={med_r:.3f} | NULL={med_n:.3f} | diff={diff:.3f} | bootstrap CI95={ci}")

    def _print_top(label: str, rows: List[Tuple[Tuple[int,int,int], int, int, float]]):
        print(f"\nTop families ({label}) [rank: stable_count, overall_count, enrichment]")
        if not rows:
            print("  (no families passed min-count filters)")
            return
        print("  Format: sig=(dom, ent_bin, leak_bin) | overall | stable | enrichment")
        for sig, ov, st, enr in rows[: min(len(rows), 10)]:
            print(f"  {sig} | {ov:5d} | {st:5d} | {enr:8.2f}x")
        if len(rows) > 10:
            print(f"  ... ({len(rows)} total returned; showing 10)")

    _print_top("REAL", top_real_rows)
    _print_top("NULL", top_null_rows)

    return BatchResult(
        top_fams_real=top_fams_real,
        top_fams_null=top_fams_null,
        median_entbin_real=med_r,
        median_entbin_null=med_n,
        median_entbin_diff=diff,
        ci_entbin_diff=ci,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--terms", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)

    ap.add_argument("--eps", type=float, default=0.050)
    ap.add_argument("--dom-threshold", type=float, default=0.250)

    # Coarser bins to avoid over-fragmented labels
    ap.add_argument("--ent-step", type=float, default=0.200)
    ap.add_argument("--leak-step", type=float, default=0.100)

    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])

    ap.add_argument("--stable-frac", type=float, default=0.01)
    ap.add_argument("--topK", type=int, default=25)

    # Smoothing + filters
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min-overall", type=int, default=25)
    ap.add_argument("--min-stable", type=int, default=3)

    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap-seed", type=int, default=7)

    ap.add_argument("--out", type=str, default="v0_7_convergence_families_v2_output.txt")
    args = ap.parse_args()

    tee = Tee(args.out)
    sys.stdout = tee

    try:
        print("=== v0_7 Convergence (Families v2) ===")
        print(f"Qubits: {args.qubits} (d={2**args.qubits}) | terms={args.terms} | seeds/batch={args.seeds} | batches={args.batches}")
        print(f"Neighbor eps={args.eps}")
        print(f"dom_threshold={args.dom_threshold} | ent_step={args.ent_step} | leak_step={args.leak_step}")
        print(f"times={args.times} | stable_frac={args.stable_frac} | topK={args.topK}")
        print(f"smoothing alpha={args.alpha} | min_overall={args.min_overall} | min_stable={args.min_stable}")
        print(f"bootstrap={args.bootstrap} | bootstrap_seed={args.bootstrap_seed}")
        print(f"Output file: {os.path.abspath(args.out)}")

        batch_results: List[BatchResult] = []
        offsets = [args.seed0 + k * args.seeds for k in range(args.batches)]
        for b in range(args.batches):
            batch_results.append(run_batch(b, offsets[b], args))

        print("\n=== Convergence diagnostics (Top-K overlap) ===")
        for i in range(args.batches):
            for j in range(i + 1, args.batches):
                inter_r, jac_r = overlap_jaccard(batch_results[i].top_fams_real, batch_results[j].top_fams_real)
                inter_n, jac_n = overlap_jaccard(batch_results[i].top_fams_null, batch_results[j].top_fams_null)
                print(f"REAL overlap Top-{args.topK}: batch{i+1} vs batch{j+1}: inter={inter_r:2d} | Jaccard={jac_r:.3f}")
                print(f"NULL overlap Top-{args.topK}: batch{i+1} vs batch{j+1}: inter={inter_n:2d} | Jaccard={jac_n:.3f}")

        diffs = np.array([r.median_entbin_diff for r in batch_results], dtype=float)
        print("\n=== Distribution-level stability (entropy-bin median difference) ===")
        print(f"diffs across batches: {diffs.tolist()}")
        print(f"mean(diff)={float(np.mean(diffs)):.3f} | std(diff)={float(np.std(diffs)):.3f}")

        print("\n=== Reading guide ===")
        print("1) If REAL Top-K overlaps are non-zero and reasonably stable, family labels are becoming replicable.")
        print("2) The entropy-bin median difference (REAL - NULL) is a robust distribution-level signal; if stable across batches,")
        print("   it supports convergence even if some families remain noisy.")
        print("3) If Top-K overlap is still ~0, increase coarsening (ent_step/leak_step), raise min_overall, and/or increase support-first ranking.")

        print("\n=== End of v0_7 convergence v2 ===")

    finally:
        try:
            sys.stdout = tee._stdout
        except Exception:
            pass
        tee.close()


if __name__ == "__main__":
    main()
