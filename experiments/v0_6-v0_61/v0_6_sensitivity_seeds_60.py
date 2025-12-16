"""
v0_6_sensitivity_seeds_500.py — Systematic Signature Detection (Sensitivity / Measurement Resolution)
-----------------------------------------------------------------------------------------

What this version adds (compared to v0_6):
1) Higher leakage measurement resolution:
   - L_raw stored at full precision (float)
   - L_bin stored as a categorical bin (e.g., "0-0.01", "0.01-0.05", ...)
   - Signature keys use L_bin (NOT rounding L_raw to 2 decimals)

2) Better accounting / logging:
   - total near-degenerate pairs found
   - total candidate pairs considered
   - trivial full-space candidates skipped (dominant_basis_count == 2**N_QUBITS)
   - candidates retained for signature counting

3) Cleaner output types:
   - entropies are plain Python floats
   - printing uses ASCII ("->") for Windows compatibility

Local execution only. English output only.
"""

import numpy as np
from collections import Counter, defaultdict
from numpy.linalg import eigh
import sys

# =========================
# Output redirection
# =========================

OUTPUT_FILE = "v0_6_output_sensivity_seeds_60.txt"

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

ORIGINAL_STDOUT = sys.stdout
file_out = open(OUTPUT_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, file_out)

# =========================
# Configuration
# =========================

N_QUBITS = 3
N_SEEDS = 60          # You can set 60 / 500 / 50000 here
NUM_TERMS = 5          # Pauli terms in the random Hamiltonian
NEAR_DEGENERACY_EPS = 1e-2

# Dominant basis selection (kept identical to v0_6 baseline)
DOMINANT_THRESHOLD = 0.15

# Dynamics sensitivity knobs (increase to expose leakage)
TIME_STEPS = 20        # was 5 in baseline
DT = 0.2               # total time = TIME_STEPS * DT

# Leakage binning (measurement resolution)
LEAKAGE_BINS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
# Labels are derived automatically from edges above.

np.set_printoptions(precision=6, suppress=True)

FULL_DIM = 2 ** N_QUBITS
BASIS_LABELS = [format(i, f"0{N_QUBITS}b") for i in range(FULL_DIM)]

# =========================
# Pauli definitions
# =========================

PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]])
}

def kron_n(pauli_string: str) -> np.ndarray:
    mat = PAULI_MATRICES[pauli_string[0]]
    for p in pauli_string[1:]:
        mat = np.kron(mat, PAULI_MATRICES[p])
    return mat

# =========================
# Hamiltonian construction
# =========================

def random_hamiltonian(seed: int, num_terms: int = NUM_TERMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    terms = []
    for _ in range(num_terms):
        pauli = "".join(rng.choice(list("IXYZ"), size=N_QUBITS))
        coeff = rng.uniform(-1.0, 1.0)
        terms.append((coeff, pauli))

    H = np.zeros((FULL_DIM, FULL_DIM), dtype=complex)
    for coeff, pauli in terms:
        H += coeff * kron_n(pauli)
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
# Time evolution (exact diagonalization unitary)
# =========================

def evolve_exact(H: np.ndarray, psi: np.ndarray, dt: float, steps: int) -> np.ndarray:
    w, v = eigh(H)
    U_dt = v @ np.diag(np.exp(-1j * w * dt)) @ v.conj().T
    psi_t = psi
    for _ in range(steps):
        psi_t = U_dt @ psi_t
    return psi_t

def leakage_raw(psi: np.ndarray, subspace_indices: list[int]) -> float:
    inside = float(np.sum(np.abs(psi[subspace_indices]) ** 2))
    return float(max(0.0, 1.0 - inside))

def leakage_bin(L: float, edges=LEAKAGE_BINS) -> str:
    for a, b in zip(edges[:-1], edges[1:]):
        if a <= L < b:
            return f"{a:g}-{b:g}"
    return f"{edges[-2]:g}-{edges[-1]:g}"

# =========================
# Main experiment
# =========================

signature_counter = Counter()

# Extra stats
total_pairs_found = 0
total_candidates_considered = 0
trivial_skipped = 0
retained_candidates = 0

# Leakage tracking (raw)
leakage_values_all = []
leakage_values_by_domcount = defaultdict(list)

print("=== v0_6_sensitivity: Systematic Signature Detection ===")
print(f"Qubits: {N_QUBITS}")
print(f"Seeds scanned: {N_SEEDS}")
print(f"Pauli terms per H: {NUM_TERMS}")
print(f"Near-degeneracy epsilon: {NEAR_DEGENERACY_EPS}")
print(f"Dominant threshold: {DOMINANT_THRESHOLD}")
print(f"Dynamics: TIME_STEPS={TIME_STEPS}, DT={DT}, total_time={TIME_STEPS*DT}")
print(f"Leakage bins: {LEAKAGE_BINS}")
print("")

for seed in range(N_SEEDS):
    H = random_hamiltonian(seed)
    evals, evecs = eigh(H)

    for i in range(len(evals)):
        for j in range(i + 1, len(evals)):
            if abs(evals[i] - evals[j]) < NEAR_DEGENERACY_EPS:
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

                dominant_indices = list(
                    set([idx for idx, _ in dom_i] + [idx for idx, _ in dom_j])
                )
                dom_count = int(len(dominant_indices))

                # Filter trivial full-space candidates
                if dom_count == FULL_DIM:
                    trivial_skipped += 1
                    continue

                # Prepare initial state as uniform superposition over dominant indices
                psi0 = np.zeros(FULL_DIM, dtype=complex)
                for idx in dominant_indices:
                    psi0[idx] = 1.0
                psi0 /= np.linalg.norm(psi0)

                # Evolve and measure leakage
                psi_t = evolve_exact(H, psi0, DT, TIME_STEPS)
                L_raw = leakage_raw(psi_t, dominant_indices)
                L_bin = leakage_bin(L_raw)

                retained_candidates += 1

                leakage_values_all.append(L_raw)
                leakage_values_by_domcount[dom_count].append(L_raw)

                # Signature key uses leakage BIN for resolution stability
                signature = (
                    dom_count,
                    round(float(entropy_i), 2),
                    round(float(entropy_j), 2),
                    L_bin
                )
                signature_counter[signature] += 1

# =========================
# Results summary
# =========================

def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    vals = sorted(values)
    def pct(p):
        k = int(round((p/100) * (len(vals)-1)))
        return vals[k]
    return {
        "n": len(vals),
        "min": float(vals[0]),
        "p10": float(pct(10)),
        "p50": float(pct(50)),
        "p90": float(pct(90)),
        "max": float(vals[-1]),
        "mean": float(sum(vals)/len(vals)),
    }

print("=== Accounting ===")
print(f"Total near-degenerate pairs found: {total_pairs_found}")
print(f"Total candidates considered (non-empty dominant sets): {total_candidates_considered}")
print(f"Trivial full-space candidates skipped (dominant_basis_count == {FULL_DIM}): {trivial_skipped}")
print(f"Candidates retained for signature counting: {retained_candidates}")
print("")

print("=== Leakage (raw) summary across retained candidates ===")
overall = summarize(leakage_values_all)
if overall:
    print(overall)
else:
    print("No retained candidates; nothing to summarize.")
print("")

print("=== Leakage (raw) summary by dominant_basis_count ===")
for dom_count in sorted(leakage_values_by_domcount.keys()):
    stats = summarize(leakage_values_by_domcount[dom_count])
    print(f"dominant_basis_count={dom_count} -> {stats}")
print("")

print("=== Signature Summary (Top 25) ===")
print("Format: (dominant_basis_count, entropy_i, entropy_j, leakage_bin) -> occurrences")
print("")
for sig, count in signature_counter.most_common(25):
    print(f"{sig}  ->  occurrences: {count}")

print("")
print("=== End of v0_6_sensitivity ===")

# Restore stdout before closing the file.
sys.stdout = ORIGINAL_STDOUT
file_out.close()
