"""
v0_6_sensitivity_fast2.py — Systematic Signature Detection (final clean / faster)
-------------------------------------------------------------------------------

Implements two targeted optimizations WITHOUT changing the scientific logic:

(1) Neighbor-scan for near-degenerate detection
    - eigenvalues from eigh() are sorted
    - for each i, only scan forward j until (E[j]-E[i]) >= eps
    - finds exactly the same pairs satisfying |Ei-Ej| < eps, but avoids O(d^2) scanning

(2) Integer signature keys
    - entropy stored as integer "centi-bits": int(round(entropy * 100))
    - leakage stored as integer bin index
    - faster counting and stable keys for large runs (50k+)

Other properties kept consistent with your v0_6 sensitivity approach:
- 3 qubits, random Pauli-sum Hamiltonians
- dominant basis extraction threshold
- exact one-shot evolution via eigendecomposition for total time t
- leakage computed as probability outside the dominant-index set
- ASCII printing and UTF-8 output file to avoid Windows cp1252 issues

Output:
- Console is mirrored to v0_6_output.txt (UTF-8)
"""

import sys
import numpy as np
from numpy.linalg import eigh
from collections import Counter, defaultdict
from itertools import product

# =========================
# Output redirection
# =========================

OUTPUT_FILE = "v0_6_output_snsivity_fast2.txt"

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except (ValueError, UnicodeEncodeError):
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass

_ORIG_STDOUT = sys.stdout
_out_f = open(OUTPUT_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _out_f)

# =========================
# Configuration
# =========================

N_QUBITS = 3
N_SEEDS = 50000            # set 60 / 500 / 50000
NUM_TERMS = 5
NEAR_DEGENERACY_EPS = 1e-2

DOMINANT_THRESHOLD = 0.15  # same concept as before

# Dynamics sensitivity knobs
TIME_STEPS = 20
DT = 0.2
TOTAL_TIME = TIME_STEPS * DT

# Leakage bins (resolution)
LEAKAGE_BINS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]

np.set_printoptions(precision=6, suppress=True)

FULL_DIM = 2 ** N_QUBITS

# =========================
# Pauli definitions + precompute all tensor products
# =========================

PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]])
}

def kron_n(pauli_string: str) -> np.ndarray:
    m = PAULI_MATRICES[pauli_string[0]]
    for p in pauli_string[1:]:
        m = np.kron(m, PAULI_MATRICES[p])
    return m

PAULI_STRINGS = ["".join(s) for s in product("IXYZ", repeat=N_QUBITS)]
PAULI_TENSORS = {s: kron_n(s) for s in PAULI_STRINGS}

# =========================
# Hamiltonian construction
# =========================

def random_hamiltonian(seed: int, num_terms: int = NUM_TERMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = np.zeros((FULL_DIM, FULL_DIM), dtype=complex)
    for _ in range(num_terms):
        pauli = "".join(rng.choice(list("IXYZ"), size=N_QUBITS))
        coeff = rng.uniform(-1.0, 1.0)
        H += coeff * PAULI_TENSORS[pauli]
    return H

# =========================
# Structural analysis
# =========================

def dominant_basis_states(vec: np.ndarray):
    probs = np.abs(vec) ** 2
    dom = [(i, probs[i]) for i in range(len(probs)) if probs[i] > DOMINANT_THRESHOLD]
    dom.sort(key=lambda x: -x[1])
    return dom

def amplitude_entropy(vec: np.ndarray) -> float:
    probs = np.abs(vec) ** 2
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log2(probs)))

# =========================
# Evolution (one-shot)
# =========================

def evolve_from_eigendecomp(evals: np.ndarray, evecs: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * evals * t)
    return evecs @ (phases * coeffs)

def leakage_raw(psi: np.ndarray, subspace_indices: np.ndarray) -> float:
    inside = float(np.sum(np.abs(psi[subspace_indices]) ** 2))
    return float(max(0.0, 1.0 - inside))

def leakage_bin_index(L: float, edges=LEAKAGE_BINS) -> int:
    # returns bin index 0..len(edges)-2
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if a <= L < b:
            return k
    return len(edges) - 2

def leakage_bin_label(bin_idx: int, edges=LEAKAGE_BINS) -> str:
    a = edges[bin_idx]
    b = edges[bin_idx + 1]
    return f"{a:g}-{b:g}"

# =========================
# Near-degenerate pair enumeration (neighbor-scan)
# =========================

def near_degenerate_pairs_sorted(evals: np.ndarray, eps: float):
    """
    Yield all (i, j) with i < j and |E[i] - E[j]| < eps.
    Assumes evals is sorted ascending (true for numpy.linalg.eigh).
    """
    n = len(evals)
    for i in range(n):
        j = i + 1
        while j < n and (evals[j] - evals[i]) < eps:
            yield i, j
            j += 1

# =========================
# Main experiment
# =========================

signature_counter = Counter()

total_pairs_found = 0
total_candidates_considered = 0
trivial_skipped = 0
retained_candidates = 0

leakage_values_all = []
leakage_values_by_domcount = defaultdict(list)

print("=== v0_6_sensitivity_fast2: Systematic Signature Detection ===")
print(f"Qubits: {N_QUBITS}")
print(f"Seeds scanned: {N_SEEDS}")
print(f"Pauli terms per H: {NUM_TERMS}")
print(f"Near-degeneracy epsilon: {NEAR_DEGENERACY_EPS}")
print(f"Dominant threshold: {DOMINANT_THRESHOLD}")
print(f"Dynamics: total_time={TOTAL_TIME} (TIME_STEPS={TIME_STEPS}, DT={DT})")
print(f"Leakage bins: {LEAKAGE_BINS}")
print("Signature key uses: (dom_count, entropy_i_centi, entropy_j_centi, leakage_bin_idx)")
print("")

for seed in range(N_SEEDS):
    H = random_hamiltonian(seed)
    evals, evecs = eigh(H)  # sorted evals

    for i, j in near_degenerate_pairs_sorted(evals, NEAR_DEGENERACY_EPS):
        total_pairs_found += 1

        psi_i = evecs[:, i]
        psi_j = evecs[:, j]

        dom_i = dominant_basis_states(psi_i)
        dom_j = dominant_basis_states(psi_j)
        if not dom_i or not dom_j:
            continue

        total_candidates_considered += 1

        entropy_i = amplitude_entropy(psi_i)
        entropy_j = amplitude_entropy(psi_j)

        # dominant index union
        dominant_indices = sorted(set([idx for idx, _ in dom_i] + [idx for idx, _ in dom_j]))
        dom_count = int(len(dominant_indices))

        # Filter trivial full-space candidates
        if dom_count == FULL_DIM:
            trivial_skipped += 1
            continue

        # psi0 = uniform superposition on dominant indices
        psi0 = np.zeros(FULL_DIM, dtype=complex)
        psi0[dominant_indices] = 1.0
        psi0 /= np.linalg.norm(psi0)

        psi_t = evolve_from_eigendecomp(evals, evecs, psi0, TOTAL_TIME)
        L_raw = leakage_raw(psi_t, np.array(dominant_indices, dtype=int))
        L_bin = leakage_bin_index(L_raw)

        retained_candidates += 1
        leakage_values_all.append(L_raw)
        leakage_values_by_domcount[dom_count].append(L_raw)

        # Integer key: centi-bits for entropy
        e_i_centi = int(round(entropy_i * 100))
        e_j_centi = int(round(entropy_j * 100))

        signature_key = (dom_count, e_i_centi, e_j_centi, L_bin)
        signature_counter[signature_key] += 1

# =========================
# Summaries
# =========================

def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    vals = sorted(values)

    def pct(p):
        k = int(round((p / 100) * (len(vals) - 1)))
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

print("=== Accounting ===")
print(f"Total near-degenerate pairs found: {total_pairs_found}")
print(f"Total candidates considered (non-empty dominant sets): {total_candidates_considered}")
print(f"Trivial full-space candidates skipped (dominant_basis_count == {FULL_DIM}): {trivial_skipped}")
print(f"Candidates retained for signature counting: {retained_candidates}")
print("")

print("=== Leakage (raw) summary across retained candidates ===")
overall = summarize(leakage_values_all)
print(overall if overall else "No retained candidates; nothing to summarize.")
print("")

print("=== Leakage (raw) summary by dominant_basis_count ===")
for dom_count in sorted(leakage_values_by_domcount.keys()):
    print(f"dominant_basis_count={dom_count} -> {summarize(leakage_values_by_domcount[dom_count])}")
print("")

print("=== Signature Summary (Top 25) ===")
print("Display format: (dom_count, entropy_i_bits, entropy_j_bits, leakage_bin_label) -> occurrences")
print("Internal key: (dom_count, entropy_i_centi, entropy_j_centi, leakage_bin_idx)")
print("")
for (dom_count, e_i_centi, e_j_centi, L_bin), count in signature_counter.most_common(25):
    e_i_bits = e_i_centi / 100.0
    e_j_bits = e_j_centi / 100.0
    label = leakage_bin_label(L_bin)
    print(f"({dom_count}, {e_i_bits:.2f}, {e_j_bits:.2f}, '{label}') -> occurrences: {count}")

print("")
print("=== End of v0_6_sensitivity_fast2 ===")

# Restore stdout and close output file
sys.stdout = _ORIG_STDOUT
_out_f.close()
