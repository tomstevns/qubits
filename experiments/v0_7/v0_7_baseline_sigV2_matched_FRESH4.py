#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_7 SigV2 (Matched + Fair Baseline)
-----------------------------------
Calibrates a "chance structure" baseline for the v0_6.1 signature pipeline
using a spectrum-matched null model (Haar-random eigenbasis with the same eigenvalues).

Outputs (REAL and NULL):
  - Top signature families by enrichment (stable-set vs overall)
  - Bootstrap CIs for enrichment factors (top signatures)
  - Hold-out replication score (seed split)
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp  # type: ignore
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False


class Tee:
    """Mirror stdout to a UTF-8 text file (Windows-safe)."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._f = open(filepath, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout
        self._closed = False

    def write(self, s):
        if self._closed:
            return
        if not isinstance(s, str):
            s = str(s)
        self._stdout.write(s)
        self._f.write(s)

    def flush(self):
        if self._closed:
            return
        self._stdout.flush()
        self._f.flush()

    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        except Exception:
            pass
        try:
            self._f.close()
        finally:
            self._closed = True


def safe_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph.conj()
    return q


PAULIS = ["I", "X", "Y", "Z"]


def build_random_hamiltonian_matrix(n_qubits: int, num_terms: int, rng: np.random.Generator) -> np.ndarray:
    if HAS_QISKIT:
        labels, coeffs = [], []
        for _ in range(num_terms):
            lab = "".join(rng.choice(PAULIS, size=n_qubits))
            c = float(rng.uniform(-1.0, 1.0))
            labels.append(lab)
            coeffs.append(c)
        op = SparsePauliOp(labels, coeffs=np.array(coeffs, dtype=complex))
        return np.asarray(op.to_matrix(), dtype=complex)

    # Manual Pauli matrices fallback
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pm = {"I": I, "X": X, "Y": Y, "Z": Z}

    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for _ in range(num_terms):
        lab = "".join(rng.choice(PAULIS, size=n_qubits))
        c = float(rng.uniform(-1.0, 1.0))
        term = np.array([[1]], dtype=complex)
        for ch in lab:
            term = np.kron(term, pm[ch])
        H = H + c * term
    return H


def build_unitary_from_eigendecomp(evals: np.ndarray, evecs: np.ndarray, t: float) -> np.ndarray:
    phase = np.exp(-1j * evals * t)
    return (evecs * phase) @ evecs.conj().T


def leakage_proxy(evals: np.ndarray, evecs: np.ndarray, n_qubits: int, a: int, b: int, times: List[float]) -> float:
    dim = 2 ** n_qubits
    psi0 = np.zeros(dim, dtype=complex)
    psi0[a] = 1.0 / math.sqrt(2.0)
    psi0[b] = 1.0 / math.sqrt(2.0)

    leaks = []
    for t in times:
        U = build_unitary_from_eigendecomp(evals, evecs, t)
        psit = U @ psi0
        pa = float(np.abs(psit[a]) ** 2)
        pb = float(np.abs(psit[b]) ** 2)
        leaks.append(max(0.0, 1.0 - (pa + pb)))
    return float(np.mean(leaks)) if leaks else 0.0


def make_sig_key(dom_count: int, ent_i: float, ent_j: float, leak: float, ent_step: float, leak_step: float) -> Tuple[int, int, int, int]:
    bi = int(round(ent_i / ent_step))
    bj = int(round(ent_j / ent_step))
    bl = int(round(leak / leak_step))
    return (int(dom_count), bi, bj, bl)


def score_candidate(dom_count: int, ent_mean: float, leak: float) -> float:
    compact = 2.0 / max(2, dom_count)
    ent_factor = 1.0 / (1.0 + ent_mean)
    stab = 1.0 - leak
    s = compact * ent_factor * stab
    return float(max(0.0, min(1.0, s)))


@dataclass(frozen=True)
class Candidate:
    seed: int
    i: int
    j: int
    deltaE: float
    dom_count: int
    ent_i: float
    ent_j: float
    leak: float
    score: float
    sig: Tuple[int, int, int, int]


def extract_candidates(seed: int, args, model: str, rng: np.random.Generator) -> List[Candidate]:
    H = build_random_hamiltonian_matrix(args.qubits, args.num_terms, rng)
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    if model == "NULL_HAAR_BASIS":
        evecs_use = haar_unitary(2 ** args.qubits, rng)
    else:
        evecs_use = evecs

    cands: List[Candidate] = []
    for i in range(len(evals) - 1):
        dE = float(abs(evals[i + 1] - evals[i]))
        if dE >= args.eps:
            continue

        vi = evecs_use[:, i]
        vj = evecs_use[:, i + 1]

        pi = np.abs(vi) ** 2
        pj = np.abs(vj) ** 2

        ent_i = safe_entropy(pi)
        ent_j = safe_entropy(pj)

        dom_union = set(np.where(pi >= args.dom_thr)[0].tolist()) | set(np.where(pj >= args.dom_thr)[0].tolist())
        if len(dom_union) < 2:
            top2 = np.argsort(-pi)[:2].tolist()
            dom_union = set(top2)

        dom_count = int(len(dom_union))

        # Use the two most dominant basis states from vi as leakage probes
        top2 = np.argsort(-pi)[:2].tolist()
        a, b = int(top2[0]), int(top2[1])
        leak = leakage_proxy(evals, evecs_use, args.qubits, a, b, args.times)

        ent_mean = 0.5 * (ent_i + ent_j)
        score = score_candidate(dom_count, ent_mean, leak)
        sig = make_sig_key(dom_count, ent_i, ent_j, leak, args.ent_step, args.leak_step)

        cands.append(Candidate(seed=seed, i=i, j=i + 1, deltaE=dE, dom_count=dom_count,
                               ent_i=float(ent_i), ent_j=float(ent_j), leak=float(leak),
                               score=float(score), sig=sig))
    return cands


def stable_mask(scores: np.ndarray, stable_frac: float) -> np.ndarray:
    n = scores.size
    k = max(1, int(math.ceil(stable_frac * n)))
    thr = np.partition(scores, n - k)[n - k]
    return scores >= thr


def enrichment_rows(sigs: List[Tuple[int, int, int, int]], mask: np.ndarray):
    overall: Dict[Tuple[int,int,int,int], int] = {}
    stable: Dict[Tuple[int,int,int,int], int] = {}
    for s, m in zip(sigs, mask):
        overall[s] = overall.get(s, 0) + 1
        if m:
            stable[s] = stable.get(s, 0) + 1

    n_all = len(sigs)
    n_st = int(np.sum(mask))
    rows = []
    for s, cnt_all in overall.items():
        cnt_st = stable.get(s, 0)
        p_all = cnt_all / max(1, n_all)
        p_st = cnt_st / max(1, n_st)
        enr = (p_st / p_all) if p_all > 0 else 0.0
        rows.append((s, cnt_all, cnt_st, enr))
    rows.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return rows, n_all, n_st


def bootstrap_ci(top_sigs: List[Tuple[int,int,int,int]], sigs: List[Tuple[int,int,int,int]], scores: np.ndarray,
                 stable_frac: float, B: int, rng: np.random.Generator):
    n = len(sigs)
    if n == 0:
        return {s: (0.0, 0.0, 0.0) for s in top_sigs}
    sig_arr = np.array(sigs, dtype=object)

    store = {s: [] for s in top_sigs}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        b_scores = scores[idx]
        b_sigs = sig_arr[idx].tolist()
        m = stable_mask(b_scores, stable_frac)
        rows, _, _ = enrichment_rows(b_sigs, m)
        enr_map = {r[0]: r[3] for r in rows}
        for s in top_sigs:
            store[s].append(float(enr_map.get(s, 0.0)))

    out = {}
    for s, vals in store.items():
        v = np.array(vals, dtype=float)
        out[s] = (float(np.mean(v)), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    return out


def holdout_replication(seeds: List[int], args) -> Tuple[float, float]:
    half = len(seeds) // 2
    A = seeds[:half]
    B = seeds[half:]

    rngA = np.random.default_rng(args.holdout_seed)
    rngB = np.random.default_rng(args.holdout_seed + 1)

    def run(seed_list: List[int], rng_master: np.random.Generator) -> List[Candidate]:
        all_c = []
        for s in seed_list:
            rng = np.random.default_rng(int(rng_master.integers(0, 2**63-1)))
            all_c.extend(extract_candidates(s, args, "REAL", rng))
        return all_c

    cA = run(A, rngA)
    cB = run(B, rngB)
    if len(cA) < 10 or len(cB) < 10:
        return 0.0, 0.0

    sigA = [c.sig for c in cA]
    scrA = np.array([c.score for c in cA], dtype=float)
    mA = stable_mask(scrA, args.stable_frac)
    rowsA, _, _ = enrichment_rows(sigA, mA)
    topA = [r[0] for r in rowsA[:args.topK]]

    sigB = [c.sig for c in cB]
    scrB = np.array([c.score for c in cB], dtype=float)
    mB = stable_mask(scrB, args.stable_frac)
    rowsB, _, _ = enrichment_rows(sigB, mB)
    enrB = {r[0]: r[3] for r in rowsB}

    mean_enr = float(np.mean([enrB.get(s, 0.0) for s in topA])) if topA else 0.0

    topB = [r[0] for r in rowsB[:args.topK]]
    setA, setB = set(topA), set(topB)
    jac = float(len(setA & setB) / max(1, len(setA | setB)))
    return mean_enr, jac


def print_header(args, out_path: str):
    print("=== v0_7 SigV2: Baseline / Null-Model Calibration (Matched) ===")
    print(f"Qubits: {args.qubits} (d={2**args.qubits}) | terms={args.num_terms} | seeds={args.seeds}")
    print(f"Near-degenerate neighbor eps={args.eps:.3f}")
    print(f"Dominant threshold={args.dom_thr:.3f} | ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
    print(f"Evidence: stable_frac={args.stable_frac} | topK={args.topK} | bootstrap={args.bootstrap}")
    print(f"Models: REAL + NULL_HAAR_BASIS (spectrum-matched)")
    print(f"Output: {out_path}")
    print("")


def report(model: str, cands: List[Candidate], args, rng_ci: np.random.Generator):
    print("----------------------------------------------")
    print(f"Model: {model}")

    if not cands:
        print("No near-degenerate neighbor pairs found under current eps.")
        return

    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak for c in cands], dtype=float)
    doms = np.array([c.dom_count for c in cands], dtype=int)
    sigs = [c.sig for c in cands]

    print(f"candidates={len(cands)} | score(mean/median/max)={scores.mean():.3f}/{np.median(scores):.3f}/{scores.max():.3f} | "
          f"leakage(mean/median/min)={leaks.mean():.3f}/{np.median(leaks):.3f}/{leaks.min():.3f} | dom_count(median)={int(np.median(doms))}")

    m = stable_mask(scores, args.stable_frac)
    rows, _, _ = enrichment_rows(sigs, m)

    print("")
    print("Top signature families (stable-set vs overall):")
    print("Format: sig=(dom, ent_bin_i, ent_bin_j, leak_bin) | overall | stable | enrichment")
    top = rows[:args.topK]
    for (s, cnt_all, cnt_st, enr) in top[:min(15, len(top))]:
        print(f"  {s} | {cnt_all:5d} | {cnt_st:5d} | {enr:6.2f}x")
    if len(top) > 15:
        print(f"  ... ({len(top)} total in topK={args.topK}, truncated display)")

    print("")
    if args.bootstrap > 0 and top:
        sig_for_ci = [r[0] for r in top[:10]]
        ci = bootstrap_ci(sig_for_ci, sigs, scores, args.stable_frac, args.bootstrap, rng_ci)
        print("Bootstrap CI for enrichment (top signatures):")
        for s in sig_for_ci:
            mean, lo, hi = ci[s]
            print(f"  {s} | mean={mean:6.2f}x | 95% CI [{lo:6.2f}x, {hi:6.2f}x]")
    else:
        print("Bootstrap disabled or no signatures for CI.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5000)
    p.add_argument("--qubits", type=int, default=3)
    p.add_argument("--num_terms", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--dom_thr", type=float, default=0.25)
    p.add_argument("--ent_step", type=float, default=0.10)
    p.add_argument("--leak_step", type=float, default=0.05)
    p.add_argument("--stable_frac", type=float, default=0.01)
    p.add_argument("--topK", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--master_seed", type=int, default=12345)
    p.add_argument("--bootstrap_seed", type=int, default=54321)
    p.add_argument("--holdout_seed", type=int, default=7777)
    p.add_argument("--out", type=str, default="v0_7_baseline_sigV2_matched_FRESH4_output.txt")
    p.add_argument("--times", type=float, nargs="*", default=[0.5, 1.0, 1.5])
    return p.parse_args()


def main():
    args = parse_args()
    out_path = os.path.abspath(args.out)

    tee = Tee(out_path)
    sys.stdout = tee
    try:
        print_header(args, out_path)

        t0 = time.time()
        rng_master = np.random.default_rng(args.master_seed)

        seeds = list(range(args.seeds))
        results = {"REAL": [], "NULL_HAAR_BASIS": []}

        for model in results.keys():
            start = time.time()
            all_c = []
            for s in seeds:
                seed_stream = int(rng_master.integers(0, 2**63 - 1))
                rng = np.random.default_rng(seed_stream + (0 if model == "REAL" else 1))
                all_c.extend(extract_candidates(s, args, model, rng))
            results[model] = all_c
            print(f"Model: {model} | candidates={len(all_c)} | elapsed={time.time()-start:.1f}s")

        print("")
        rng_ci = np.random.default_rng(args.bootstrap_seed)
        for model, cands in results.items():
            report(model, cands, args, rng_ci)

        print("")
        print("----------------------------------------------")
        print("Hold-out replication (REAL):")
        mean_enr, jac = holdout_replication(seeds, args)
        print(f"  Mean enrichment of topK(A) signatures evaluated in B: {mean_enr:.2f}x")
        print(f"  Jaccard overlap of topK(A) vs topK(B): {jac:.3f}")

        print("")
        print("=== End of v0_7 ===")
        print(f"Total elapsed: {time.time() - t0:.1f}s")
    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
