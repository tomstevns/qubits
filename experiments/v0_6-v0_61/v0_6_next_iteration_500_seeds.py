"""
v0_6.py — Systematic Signature Detection
---------------------------------------

Purpose:
Identify repeatable structural signatures associated with
physics-native, dynamically stable subspaces.

Local execution only.
English output only.
"""

import numpy as np
from collections import Counter
from numpy.linalg import eigh
import sys

# =========================
# Output redirection
# =========================

OUTPUT_FILE = "v0_6_output_next_iteration_500_seeds.txt"

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except (ValueError, UnicodeEncodeError):
                # Ignore writes/flushes to closed files or unsupported encodings.
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
N_SEEDS = 500
NEAR_DEGENERACY_EPS = 1e-2
DOMINANT_THRESHOLD = 0.15
TIME_STEPS = 5
DT = 0.2

np.set_printoptions(precision=4, suppress=True)

FULL_DIM = 2 ** N_QUBITS

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

def random_hamiltonian(seed: int, num_terms: int = 5) -> np.ndarray:
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
# Time evolution
# =========================

def trotter_step(H: np.ndarray, psi: np.ndarray, dt: float) -> np.ndarray:
    w, v = eigh(H)
    U = v @ np.diag(np.exp(-1j * w * dt)) @ v.conj().T
    return U @ psi

def leakage(psi: np.ndarray, subspace_indices: list[int]) -> float:
    return float(1.0 - np.sum(np.abs(psi[subspace_indices]) ** 2))

# =========================
# Main experiment
# =========================

signature_counter = Counter()
trivial_skipped = 0

print("=== v0_6: Systematic Signature Detection ===")
print(f"Qubits: {N_QUBITS}")
print(f"Seeds scanned: {N_SEEDS}")
print("")

for seed in range(N_SEEDS):
    H = random_hamiltonian(seed)
    evals, evecs = eigh(H)

    for i in range(len(evals)):
        for j in range(i + 1, len(evals)):
            if abs(evals[i] - evals[j]) < NEAR_DEGENERACY_EPS:
                psi_i = evecs[:, i]
                psi_j = evecs[:, j]

                dom_i = dominant_basis_states(psi_i)
                dom_j = dominant_basis_states(psi_j)

                if not dom_i or not dom_j:
                    continue

                entropy_i = amplitude_entropy(psi_i)
                entropy_j = amplitude_entropy(psi_j)

                dominant_indices = list(
                    set([idx for idx, _ in dom_i] + [idx for idx, _ in dom_j])
                )

                # Recommendation A: filter trivial "subspaces" that expand to the full Hilbert space.
                if len(dominant_indices) == FULL_DIM:
                    trivial_skipped += 1
                    continue

                psi0 = np.zeros(FULL_DIM, dtype=complex)
                for idx in dominant_indices:
                    psi0[idx] = 1.0
                psi0 /= np.linalg.norm(psi0)

                psi_t = psi0.copy()
                for _ in range(TIME_STEPS):
                    psi_t = trotter_step(H, psi_t, DT)

                L = leakage(psi_t, dominant_indices)

                # Recommendation: cast to plain Python floats for clean printing/signature keys.
                signature = (
                    int(len(dominant_indices)),
                    round(float(entropy_i), 2),
                    round(float(entropy_j), 2),
                    round(float(L), 2)
                )

                signature_counter[signature] += 1

# =========================
# Results summary
# =========================

print("=== Signature Summary ===")
print("Format: (dominant_basis_count, entropy_i, entropy_j, leakage)")
print("")
print(f"Trivial full-space cases skipped (dominant_basis_count == {FULL_DIM}): {trivial_skipped}")
print("")

for sig, count in signature_counter.most_common(15):
    print(f"{sig}  ->  occurrences: {count}")

print("")
print("=== End of v0_6 ===")

# Restore stdout before closing the file to avoid flush-on-closed-file issues.
sys.stdout = ORIGINAL_STDOUT
file_out.close()
