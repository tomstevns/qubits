"""
v0_6_1_interestingness_fixed.py — FIX for heapq tie-breaking

Your crash happens because heapq compares tuple elements. If scores tie and the next
element is a dict, Python cannot compare dict<dict.

Fix: push (score, tie_id, info_dict), where tie_id is an ever-increasing int.

This preserves the scientific logic; it only fixes heap ordering mechanics.
"""

import sys
import heapq
import numpy as np
from numpy.linalg import eigh
from collections import Counter, defaultdict
from itertools import product

OUTPUT_FILE = "v0_6_1_output_interestingness.txt"


class Tee:
    """
    Mirror stdout both to console and to a UTF-8 text file.
    Handles Unicode safely and avoids flush-on-closed issues.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except (ValueError, UnicodeEncodeError):
                # ValueError can happen if a stream is already closed.
                # UnicodeEncodeError can happen on some Windows consoles.
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass


# -------------------------
# Configuration
# -------------------------
N_QUBITS = 3
N_SEEDS = 60               # change to 500 / 50000
NUM_TERMS = 5
NEAR_DEGENERACY_EPS = 1e-2

DOMINANT_THRESHOLD = 0.15

TIME_STEPS = 20
DT = 0.2
TOTAL_TIME = TIME_STEPS * DT

LEAKAGE_BINS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]

TOP_SIGNATURES = 25
TOP_CANDIDATES = 20
MIN_COUNT_FOR_SCORE_RANK = 10

np.set_printoptions(precision=6, suppress=True)
FULL_DIM = 2 ** N_QUBITS

# -------------------------
# Pauli matrices
# -------------------------
PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


def kron_n(pauli_string: str) -> np.ndarray:
    m = PAULI_MATRICES[pauli_string[0]]
    for p in pauli_string[1:]:
        m = np.kron(m, PAULI_MATRICES[p])
    return m


PAULI_STRINGS = ["".join(s) for s in product("IXYZ", repeat=N_QUBITS)]
PAULI_TENSORS = {s: kron_n(s) for s in PAULI_STRINGS}


def random_hamiltonian(seed: int, num_terms: int = NUM_TERMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = np.zeros((FULL_DIM, FULL_DIM), dtype=complex)
    for _ in range(num_terms):
        pauli = "".join(rng.choice(list("IXYZ"), size=N_QUBITS))
        coeff = rng.uniform(-1.0, 1.0)
        H += coeff * PAULI_TENSORS[pauli]
    return H


def dominant_basis_states(vec: np.ndarray):
    probs = np.abs(vec) ** 2
    dom = [(i, probs[i]) for i in range(len(probs)) if probs[i] > DOMINANT_THRESHOLD]
    dom.sort(key=lambda x: -x[1])
    return dom


def amplitude_entropy(vec: np.ndarray) -> float:
    probs = np.abs(vec) ** 2
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log2(probs)))


def evolve_from_eigendecomp(evals: np.ndarray, evecs: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """
    Exact unitary evolution using spectral decomposition:
      psi(t) = V diag(exp(-i E t)) V^† psi(0)
    """
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * evals * t)
    return evecs @ (phases * coeffs)


def leakage_raw(psi: np.ndarray, subspace_indices: np.ndarray) -> float:
    inside = float(np.sum(np.abs(psi[subspace_indices]) ** 2))
    return float(max(0.0, 1.0 - inside))


def leakage_bin_index(L: float, edges=LEAKAGE_BINS) -> int:
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if a <= L < b:
            return k
    return len(edges) - 2


def leakage_bin_label(bin_idx: int, edges=LEAKAGE_BINS) -> str:
    a = edges[bin_idx]
    b = edges[bin_idx + 1]
    return f"{a:g}-{b:g}"


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def baseline_leakage_for_k(k: int, d: int) -> float:
    """
    For a random state, expected in-subspace probability ~ k/d,
    hence expected leakage ~ 1 - k/d.
    """
    return max(1e-12, 1.0 - (k / float(d)))


def adjusted_leakage(L_raw: float, k: int, d: int) -> float:
    return L_raw / baseline_leakage_for_k(k, d)


def interestingness_score(L_raw: float, k: int, d: int) -> float:
    """
    Score in [0,1], higher is better.
      score = 1 - clamp( L_raw / (1 - k/d) )
    """
    return 1.0 - clamp01(adjusted_leakage(L_raw, k, d))


def near_degenerate_pairs_sorted(evals: np.ndarray, eps: float):
    """
    Neighbor scan in sorted spectrum: yields all (i,j) such that evals[j]-evals[i] < eps.
    evals assumed sorted by eigh.
    """
    n = len(evals)
    for i in range(n):
        j = i + 1
        while j < n and (evals[j] - evals[i]) < eps:
            yield i, j
            j += 1


def summarize(values):
    if not values:
        return {}
    vals = sorted(values)

    def pct(p):
        k = int(round((p / 100.0) * (len(vals) - 1)))
        return vals[k]

    return {
        "n": len(vals),
        "min": float(vals[0]),
        "p10": float(pct(10)),
        "p50": float(pct(50)),
        "p90": float(pct(90)),
        "max": float(vals[-1]),
        "mean": float(sum(vals) / len(vals)),
    }


def main():
    signature_counter = Counter()
    sig_sum_score = defaultdict(float)
    sig_sum_leak = defaultdict(float)
    sig_min_leak = defaultdict(lambda: float("inf"))

    total_pairs_found = 0
    total_candidates_considered = 0
    trivial_skipped = 0
    retained_candidates = 0

    leakage_values_all = []
    score_values_all = []

    # FIX: heap entries are (score, tie_id, info_dict) so ties never compare dicts.
    top_heap = []
    tie_id = 0

    print("=== v0_6.1: Systematic Signature Detection + Interestingness Scoring (fixed heap) ===")
    print(f"Qubits: {N_QUBITS}")
    print(f"Seeds scanned: {N_SEEDS}")
    print(f"Pauli terms per H: {NUM_TERMS}")
    print(f"Near-degeneracy epsilon: {NEAR_DEGENERACY_EPS}")
    print(f"Dominant threshold: {DOMINANT_THRESHOLD}")
    print(f"Dynamics: total_time={TOTAL_TIME} (TIME_STEPS={TIME_STEPS}, DT={DT})")
    print(f"Leakage bins: {LEAKAGE_BINS}")
    print("Signature key: (dom_count, entropy_i_centi, entropy_j_centi, leakage_bin_idx)")
    print("Scoring: score = 1 - clamp( L_raw / (1 - k/d) )")
    print("")

    for seed in range(N_SEEDS):
        H = random_hamiltonian(seed)
        evals, evecs = eigh(H)  # evals sorted

        for i, j in near_degenerate_pairs_sorted(evals, NEAR_DEGENERACY_EPS):
            total_pairs_found += 1

            psi_i = evecs[:, i]
            psi_j = evecs[:, j]

            dom_i = dominant_basis_states(psi_i)
            dom_j = dominant_basis_states(psi_j)
            if not dom_i or not dom_j:
                continue

            total_candidates_considered += 1

            # Build candidate support set from dominant indices of both eigenvectors
            dominant_indices = sorted(set([idx for idx, _ in dom_i] + [idx for idx, _ in dom_j]))
            dom_count = int(len(dominant_indices))

            if dom_count == FULL_DIM:
                trivial_skipped += 1
                continue

            # Uniform superposition over dominant basis indices
            psi0 = np.zeros(FULL_DIM, dtype=complex)
            psi0[dominant_indices] = 1.0
            psi0 /= np.linalg.norm(psi0)

            # Evolve and compute leakage
            psi_t = evolve_from_eigendecomp(evals, evecs, psi0, TOTAL_TIME)
            L_raw = leakage_raw(psi_t, np.array(dominant_indices, dtype=int))
            L_bin = leakage_bin_index(L_raw)

            retained_candidates += 1
            leakage_values_all.append(L_raw)

            score = interestingness_score(L_raw, dom_count, FULL_DIM)
            score_values_all.append(score)

            entropy_i = amplitude_entropy(psi_i)
            entropy_j = amplitude_entropy(psi_j)

            sig_key = (dom_count, int(round(entropy_i * 100)), int(round(entropy_j * 100)), L_bin)

            signature_counter[sig_key] += 1
            sig_sum_score[sig_key] += score
            sig_sum_leak[sig_key] += L_raw
            if L_raw < sig_min_leak[sig_key]:
                sig_min_leak[sig_key] = L_raw

            # Top-K candidates (fixed heap tie-break)
            if TOP_CANDIDATES > 0:
                info = {
                    "seed": seed,
                    "pair": (i, j),
                    "deltaE": float(abs(evals[i] - evals[j])),
                    "dom_count": dom_count,
                    "entropy_i": float(entropy_i),
                    "entropy_j": float(entropy_j),
                    "L_raw": float(L_raw),
                    "L_adj": float(adjusted_leakage(L_raw, dom_count, FULL_DIM)),
                    "score": float(score),
                    "bin": leakage_bin_label(L_bin),
                }

                tie_id += 1
                entry = (score, tie_id, info)

                if len(top_heap) < TOP_CANDIDATES:
                    heapq.heappush(top_heap, entry)
                else:
                    if score > top_heap[0][0]:
                        heapq.heapreplace(top_heap, entry)

    def mean_score(sig_key):
        return sig_sum_score[sig_key] / float(signature_counter[sig_key])

    def mean_leak(sig_key):
        return sig_sum_leak[sig_key] / float(signature_counter[sig_key])

    print("=== Accounting ===")
    print(f"Total near-degenerate pairs found: {total_pairs_found}")
    print(f"Total candidates considered: {total_candidates_considered}")
    print(f"Trivial full-space candidates skipped (dom_count == {FULL_DIM}): {trivial_skipped}")
    print(f"Candidates retained: {retained_candidates}")
    print("")

    print("=== Leakage (raw) summary across retained candidates ===")
    print(summarize(leakage_values_all) if leakage_values_all else "None")
    print("")

    print("=== Interestingness score summary across retained candidates ===")
    print(summarize(score_values_all) if score_values_all else "None")
    print("")

    print(f"=== Signature Summary by Frequency (Top {TOP_SIGNATURES}) ===")
    print("Display: (dom_count, entropy_i_bits, entropy_j_bits, leakage_bin) -> occurrences")
    for (dom_count, e_i_c, e_j_c, L_bin), count in signature_counter.most_common(TOP_SIGNATURES):
        print(f"({dom_count}, {e_i_c/100.0:.2f}, {e_j_c/100.0:.2f}, '{leakage_bin_label(L_bin)}') -> {count}")
    print("")

    eligible = [k for k, c in signature_counter.items() if c >= MIN_COUNT_FOR_SCORE_RANK]
    eligible.sort(key=lambda k: mean_score(k), reverse=True)

    print(f"=== Signature Summary by Mean Interestingness (count >= {MIN_COUNT_FOR_SCORE_RANK}) ===")
    print("Display: (dom_count, entropy_i_bits, entropy_j_bits, leakage_bin) -> count, mean_score, mean_leak, min_leak")
    for sig_key in eligible[:TOP_SIGNATURES]:
        c = signature_counter[sig_key]
        dom_count, e_i_c, e_j_c, L_bin = sig_key
        print(
            f"({dom_count}, {e_i_c/100.0:.2f}, {e_j_c/100.0:.2f}, '{leakage_bin_label(L_bin)}')"
            f" -> count={c}, mean_score={mean_score(sig_key):.3f}, mean_leak={mean_leak(sig_key):.3f}, min_leak={sig_min_leak[sig_key]:.3f}"
        )
    print("")

    compact = [k for k in eligible if k[0] <= 4]
    compact.sort(key=lambda k: mean_score(k), reverse=True)

    print(f"=== Compact Subspaces Only (dom_count <= 4, count >= {MIN_COUNT_FOR_SCORE_RANK}) ===")
    for sig_key in compact[:TOP_SIGNATURES]:
        c = signature_counter[sig_key]
        dom_count, e_i_c, e_j_c, L_bin = sig_key
        print(
            f"({dom_count}, {e_i_c/100.0:.2f}, {e_j_c/100.0:.2f}, '{leakage_bin_label(L_bin)}')"
            f" -> count={c}, mean_score={mean_score(sig_key):.3f}, mean_leak={mean_leak(sig_key):.3f}, min_leak={sig_min_leak[sig_key]:.3f}"
        )
    print("")

    if top_heap:
        print(f"=== Top {min(TOP_CANDIDATES, len(top_heap))} Individual Candidates by Interestingness ===")
        best = sorted(top_heap, key=lambda x: x[0], reverse=True)  # (score, tie_id, info)
        for rank, (s, _, info) in enumerate(best, start=1):
            print(
                f"[{rank}] score={info['score']:.3f}  L_raw={info['L_raw']:.3f}  L_adj={info['L_adj']:.3f}  "
                f"bin={info['bin']}  dom={info['dom_count']}  seed={info['seed']}  pair={info['pair']}  "
                f"dE={info['deltaE']:.6f}  entropies=({info['entropy_i']:.3f},{info['entropy_j']:.3f})"
            )
        print("")

    print("=== End of v0_6.1 ===")


if __name__ == "__main__":
    _ORIG_STDOUT = sys.stdout
    _out_f = None
    try:
        _out_f = open(OUTPUT_FILE, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, _out_f)
        main()
    finally:
        sys.stdout = _ORIG_STDOUT
        if _out_f is not None:
            try:
                _out_f.close()
            except Exception:
                pass
