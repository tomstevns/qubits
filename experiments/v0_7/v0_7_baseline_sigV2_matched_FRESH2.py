# v0_7_baseline_sigV2_matched_FRESH2.py
# Baseline / null-model calibration matched to v0_6.1 SigV2
# Focus: "fair" baseline for enrichment/replication claims, avoiding HAAR_BASIS artifacts.
#
# Key change vs earlier baseline attempts:
#   - REAL: compute candidate signatures + stability score from the same pipeline as v0_6.1-style extraction
#   - NULL (PERMUTED-SCORE): permutation test that breaks the association between (signature family) and (stability score)
#       while preserving the full marginal distributions of signatures and scores.
#   - Optional diagnostic null: HAAR-BASIS (expected to look diffuse in computational basis; not used for inference).
#
# Output: ASCII-safe and tee'd to a UTF-8 text file.

from __future__ import annotations
import argparse
import math
import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from qiskit.quantum_info import SparsePauliOp

class Tee:
    """Print to both stdout and a UTF-8 text file (Windows-safe)."""
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout

    def write(self, s: str):
        if not isinstance(s, str):
            s = str(s)
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

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default

def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))

def bin_int(x: float, step: float) -> int:
    if step <= 0:
        return 0
    return int(math.floor(x / step + 1e-12))

def jaccard(a: List[Tuple[int,int,int,int]], b: List[Tuple[int,int,int,int]]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

PAULIS = ["I", "X", "Y", "Z"]

def random_pauli_string(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(PAULIS) for _ in range(n))

def build_random_hamiltonian(n_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    rng = random.Random(seed)
    paulis = []
    coeffs = []
    for _ in range(num_terms):
        p = random_pauli_string(n_qubits, rng)
        if p == "I"*n_qubits and rng.random() < 0.95:
            idx = rng.randrange(n_qubits)
            p = p[:idx] + rng.choice(["X","Y","Z"]) + p[idx+1:]
        paulis.append(p)
        coeffs.append(rng.uniform(-1.0, 1.0))
    return SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=np.complex128))

def diagonalize(H: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = H.to_matrix()
    evals, evecs = np.linalg.eigh(M)
    return evals.real, evecs, M

def neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int,int,float]]:
    idx = np.argsort(evals)
    pairs = []
    for a, b in zip(idx[:-1], idx[1:]):
        de = abs(float(evals[a] - evals[b]))
        if de < eps:
            pairs.append((int(a), int(b), float(de)))
    return pairs

def dominant_basis(vec: np.ndarray, p_dom: float) -> Tuple[List[int], np.ndarray]:
    probs = np.abs(vec)**2
    order = np.argsort(-probs)
    dom = [int(i) for i in order if probs[i] >= p_dom]
    return dom, probs

def leakage_proxy_from_basis(Hmat: np.ndarray, basis_indices: List[int], t_list: List[float]) -> float:
    if len(basis_indices) == 0:
        return 1.0
    dim = Hmat.shape[0]
    m = len(basis_indices)
    psi0 = np.zeros(dim, dtype=np.complex128)
    amp = 1.0 / math.sqrt(m)
    for i in basis_indices:
        psi0[i] = amp

    evals, evecs = np.linalg.eigh(Hmat)
    leaks = []
    for t in t_list:
        phases = np.exp(-1j * evals * t)
        U = (evecs * phases) @ evecs.conj().T
        psit = U @ psi0
        probs = np.abs(psit)**2
        inside = float(np.sum(probs[basis_indices]))
        leaks.append(max(0.0, min(1.0, 1.0 - inside)))
    return float(np.mean(leaks))

@dataclass
class Candidate:
    seed: int
    i: int
    j: int
    de: float
    dom_count: int
    ent_i: float
    ent_j: float
    leakage: float
    score: float
    sig: Tuple[int,int,int,int]

def compute_score(dom_count: int, ent_mean: float, leakage: float) -> float:
    stability = 1.0 - leakage
    sparsity = 1.0 / max(1.0, float(dom_count))
    simplicity = 1.0 / (1.0 + ent_mean)
    s = stability * sparsity * simplicity
    return float(max(0.0, min(1.0, s)))

def make_signature(dom_count: int, ent_i: float, ent_j: float, leakage: float,
                   ent_step: float, leak_step: float) -> Tuple[int,int,int,int]:
    return (int(dom_count), bin_int(ent_i, ent_step), bin_int(ent_j, ent_step), bin_int(leakage, leak_step))

def extract_candidates_for_seed(seed: int, args) -> List[Candidate]:
    H = build_random_hamiltonian(args.qubits, args.num_terms, seed)
    evals, evecs, Hmat = diagonalize(H)
    pairs = neighbor_pairs(evals, args.eps)
    if not pairs:
        return []

    cands = []
    for (i, j, de) in pairs:
        vi = evecs[:, i]
        vj = evecs[:, j]
        dom_i, pi = dominant_basis(vi, args.p_dom)
        dom_j, pj = dominant_basis(vj, args.p_dom)

        dom_union = sorted(set(dom_i) | set(dom_j))
        dom_count = len(dom_union)

        ent_i = shannon_entropy(pi)
        ent_j = shannon_entropy(pj)
        ent_mean = 0.5*(ent_i + ent_j)

        leakage = leakage_proxy_from_basis(Hmat, dom_union, args.t_list)
        score = compute_score(dom_count, ent_mean, leakage)
        sig = make_signature(dom_count, ent_i, ent_j, leakage, args.family_ent_step, args.family_leak_step)

        cands.append(Candidate(seed, i, j, de, dom_count, ent_i, ent_j, leakage, score, sig))
    return cands

def stable_set_indices(scores: np.ndarray, frac: float) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, int(math.ceil(frac * n)))
    idx = np.argsort(-scores)
    return idx[:k]

def enrichment_for_signatures(sigs: List[Tuple[int,int,int,int]], stable_mask: np.ndarray) -> Dict[Tuple[int,int,int,int], float]:
    N = len(sigs)
    if N == 0:
        return {}
    stable_idx = np.where(stable_mask)[0]
    Ns = len(stable_idx)
    if Ns == 0:
        return {}

    overall = {}
    stable = {}
    for k, s in enumerate(sigs):
        overall[s] = overall.get(s, 0) + 1
        if stable_mask[k]:
            stable[s] = stable.get(s, 0) + 1

    enr = {}
    for s, c_all in overall.items():
        c_st = stable.get(s, 0)
        p_all = c_all / N
        p_st = c_st / Ns
        enr[s] = safe_div(p_st, p_all, default=0.0)
    return enr

def counts_support(sigs: List[Tuple[int,int,int,int]]) -> Dict[Tuple[int,int,int,int], int]:
    d = {}
    for s in sigs:
        d[s] = d.get(s, 0) + 1
    return d

def top_families(enr: Dict[Tuple[int,int,int,int], float], support: Dict[Tuple[int,int,int,int], int], topK: int, min_support: int) -> List[Tuple[int,int,int,int]]:
    items = []
    for sig, ef in enr.items():
        if support.get(sig, 0) >= min_support:
            items.append((ef, sig))
    items.sort(reverse=True, key=lambda x: x[0])
    return [sig for _, sig in items[:topK]]

def bootstrap_ci_for_top(top: List[Tuple[int,int,int,int]], sigs: List[Tuple[int,int,int,int]], scores: np.ndarray,
                        frac: float, n_boot: int, seed: int) -> Dict[Tuple[int,int,int,int], Tuple[float,float,float]]:
    N = len(sigs)
    if N == 0:
        return {s: (0.0, 0.0, 0.0) for s in top}
    sigs_arr = np.array(sigs, dtype=object)
    rng = np.random.default_rng(seed)
    boot = {s: [] for s in top}
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        b_sigs = sigs_arr[idx].tolist()
        b_scores = scores[idx]
        st_idx = stable_set_indices(b_scores, frac)
        mask = np.zeros(N, dtype=bool)
        mask[st_idx] = True
        enr = enrichment_for_signatures(b_sigs, mask)
        for s in top:
            boot[s].append(float(enr.get(s, 0.0)))
    out = {}
    for s, vals in boot.items():
        vals = np.array(vals, dtype=float)
        out[s] = (float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
    return out

def report_real(cands: List[Candidate], args):
    sigs = [c.sig for c in cands]
    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leakage for c in cands], dtype=float)
    doms = np.array([c.dom_count for c in cands], dtype=int)

    print("\n=== REAL: Signature enrichment (stable-set vs overall) ===")
    if len(cands) == 0:
        print("No candidates.")
        return

    print(f"candidates={len(cands)} | score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f} | leakage(mean/median/min)={leaks.mean():.3f}/{np.median(leaks):.3f}/{leaks.min():.3f} | dom_count(median)={int(np.median(doms))}")
    support = counts_support(sigs)

    for frac in args.stable_fracs:
        st_idx = stable_set_indices(scores, frac)
        mask = np.zeros(len(scores), dtype=bool)
        mask[st_idx] = True
        enr = enrichment_for_signatures(sigs, mask)
        top = top_families(enr, support, args.topK, args.min_support)
        ci = bootstrap_ci_for_top(top, sigs, scores, frac, args.bootstrap, args.bootstrap_seed)

        print(f"\nStable fraction={frac:.3f} | stable_n={len(st_idx)} | topK={len(top)}")
        print("sig=(dom, ent_i_bin, ent_j_bin, leak_bin) | support | EF (boot_mean) [CI95%]")
        for sig in top:
            ef = float(enr.get(sig, 0.0))
            m, lo, hi = ci.get(sig, (ef, ef, ef))
            print(f"{sig} | n={support.get(sig,0)} | EF={ef:.2f} (boot_mean={m:.2f}) [{lo:.2f}, {hi:.2f}]")

def report_holdout(cands: List[Candidate], args):
    print("\n=== REAL: Hold-out replication (seed halves) ===")
    if len(cands) == 0:
        print("No candidates.")
        return
    seeds = sorted(set(c.seed for c in cands))
    if len(seeds) < 2:
        print("Not enough seeds.")
        return
    mid = len(seeds)//2
    sA, sB = set(seeds[:mid]), set(seeds[mid:])
    A = [c for c in cands if c.seed in sA]
    B = [c for c in cands if c.seed in sB]

    def top_for(cs: List[Candidate], frac: float) -> List[Tuple[int,int,int,int]]:
        sigs = [c.sig for c in cs]
        if not sigs:
            return []
        scores = np.array([c.score for c in cs], dtype=float)
        st = stable_set_indices(scores, frac)
        mask = np.zeros(len(scores), dtype=bool)
        mask[st] = True
        enr = enrichment_for_signatures(sigs, mask)
        sup = counts_support(sigs)
        return top_families(enr, sup, args.topK, args.min_support)

    for frac in args.stable_fracs:
        topA = top_for(A, frac)
        topB = top_for(B, frac)
        print(f"stable_frac={frac:.3f} | topA={len(topA)} topB={len(topB)} | Jaccard={jaccard(topA, topB):.3f}")

def report_permutation_null(cands: List[Candidate], args):
    print("\n=== NULL: Permuted-score enrichment baseline ===")
    if len(cands) == 0:
        print("No candidates.")
        return

    sigs = [c.sig for c in cands]
    scores = np.array([c.score for c in cands], dtype=float)
    support = counts_support(sigs)

    rng = np.random.default_rng(args.null_seed)
    perm_scores = scores.copy()
    rng.shuffle(perm_scores)

    print("Null mode: PERMUTED-SCORE (break signature<->stability association; preserve marginals)")

    for frac in args.stable_fracs:
        st_idx = stable_set_indices(perm_scores, frac)
        mask = np.zeros(len(scores), dtype=bool)
        mask[st_idx] = True
        enr = enrichment_for_signatures(sigs, mask)
        top = top_families(enr, support, args.topK, args.min_support)

        # Bootstrap CI under null: resample and reshuffle per replicate
        boot = {sig: [] for sig in top}
        N = len(sigs)
        brng = np.random.default_rng(args.bootstrap_seed + 1)
        for _ in range(args.bootstrap):
            idx = brng.integers(0, N, size=N)
            b_sigs = [sigs[i] for i in idx]
            b_scores = scores[idx].copy()
            brng.shuffle(b_scores)
            b_st = stable_set_indices(b_scores, frac)
            b_mask = np.zeros(N, dtype=bool)
            b_mask[b_st] = True
            b_enr = enrichment_for_signatures(b_sigs, b_mask)
            for sig in top:
                boot[sig].append(float(b_enr.get(sig, 0.0)))

        print(f"\nStable fraction={frac:.3f} | stable_n={len(st_idx)} | topK={len(top)}")
        print("sig | support | EF_null [CI95%]")
        for sig in top:
            vals = np.array(boot[sig], dtype=float)
            lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
            ef = float(enr.get(sig, 0.0))
            print(f"{sig} | n={support.get(sig,0)} | EF_null={ef:.2f} [{lo:.2f}, {hi:.2f}]")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--num_terms", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--p_dom", type=float, default=0.25)
    ap.add_argument("--family_ent_step", type=float, default=0.10)
    ap.add_argument("--family_leak_step", type=float, default=0.05)
    ap.add_argument("--t_list", type=str, default="0.5,1.0")
    ap.add_argument("--stable_fracs", type=str, default="0.01,0.05")
    ap.add_argument("--topK", type=int, default=50)
    ap.add_argument("--min_support", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument("--null_seed", type=int, default=999)
    ap.add_argument("--out", type=str, default="v0_7_baseline_sigV2_matched_FRESH2_output.txt")
    return ap.parse_args()

def main():
    args = parse_args()
    args.t_list = [float(x.strip()) for x in args.t_list.split(",") if x.strip()]
    args.stable_fracs = [float(x.strip()) for x in args.stable_fracs.split(",") if x.strip()]

    tee = Tee(args.out)
    sys.stdout = tee

    print("=== v0_7 SigV2: Baseline / Null-Model Calibration (Matched FRESH2) ===")
    print(f"Qubits: {args.qubits} (d={2**args.qubits}) | terms={args.num_terms} | seeds={args.seeds}")
    print(f"Near-degenerate neighbor eps={args.eps:.3f}")
    print(f"Dominant threshold={args.p_dom:.3f} | family bins: ent_step={args.family_ent_step:.3f} | leak_step={args.family_leak_step:.3f}")
    print(f"Evidence: stable_fracs={args.stable_fracs} | topK={args.topK} | bootstrap={args.bootstrap} | min_support={args.min_support}")
    print(f"Output: {os.path.abspath(args.out)}")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    t0 = time.time()
    cands: List[Candidate] = []
    for s in seeds:
        cands.extend(extract_candidates_for_seed(s, args))
    dt = time.time() - t0
    print(f"\nModel: REAL | candidates={len(cands)} | elapsed={dt:.1f}s")

    report_real(cands, args)
    report_holdout(cands, args)
    report_permutation_null(cands, args)

    print("\n=== End of v0_7 SigV2 Matched Baseline (FRESH2) ===")
    tee.flush()
    tee.close()

if __name__ == "__main__":
    main()
