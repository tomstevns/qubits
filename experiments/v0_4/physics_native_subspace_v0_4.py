import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from datetime import datetime
import sys

# ==========================================================
# Helper: redirect console output to a text file
# ==========================================================
output_file = "v0_4_output.txt"
sys.stdout = open(output_file, "w")

print("Physics-Native Subspace Discovery – Prototype v0.4")
print("Run timestamp:", datetime.now())
print("----------------------------------------------------\n")


# ==========================================================
# 1. Build a random Hamiltonian (general version)
# ==========================================================
def build_random_hamiltonian_sparse(n_qubits=3, num_terms=6, seed=None):
    """
    Builds a random Hamiltonian using Pauli strings.
    Uses only I, X, Z to keep the landscape interpretable.
    """
    rng = np.random.default_rng(seed)
    pauli_letters = ["I", "X", "Z"]
    terms = []

    for _ in range(num_terms):
        label = "".join(rng.choice(pauli_letters) for _ in range(n_qubits))

        # skip pure identity
        if set(label) == {"I"}:
            continue

        coeff = rng.uniform(-1.0, 1.0)
        terms.append((label, coeff))

    if not terms:
        # fallback Hamiltonian if everything filtered out
        terms = [("Z" + "I" * (n_qubits - 1), 1.0)]

    labels = [t[0] for t in terms]
    coeffs = [t[1] for t in terms]

    H = SparsePauliOp.from_list(list(zip(labels, coeffs)))
    return H


# ==========================================================
# 2. Find eigenvalues and near-degenerate pairs
# ==========================================================
def find_near_degenerate_pairs(hamiltonian_op, tolerance=0.05):
    """
    Finds eigenvalue pairs |Ei - Ej| < tolerance.
    Returns eigenvalues, eigenvectors, and detected pairs.
    """
    H_mat = hamiltonian_op.to_matrix()
    evals, evecs = np.linalg.eigh(H_mat)

    pairs = []
    for i in range(len(evals)):
        for j in range(i + 1, len(evals)):
            if abs(evals[i] - evals[j]) < tolerance:
                pairs.append((i, j, evals[i], evals[j]))

    return evals, evecs, pairs


# ==========================================================
# 3. Display eigenvectors in computational basis
# ==========================================================
def print_statevector(evec, n_qubits=3, max_terms=6):
    """
    Prints the dominant computational basis states.
    """
    sv = Statevector(evec).data
    idx_sorted = np.argsort(-np.abs(sv))

    print("  Dominant basis states:")
    shown = 0
    for idx in idx_sorted:
        amp = sv[idx]
        if np.abs(amp) < 1e-3:
            continue

        bitstring = format(idx, f"0{n_qubits}b")
        print(f"    |{bitstring}> : amplitude {amp:.3f}")
        shown += 1
        if shown >= max_terms:
            break


# ==========================================================
# 4. MAIN EXECUTION LOOP
# ==========================================================
if __name__ == "__main__":
    n_qubits = 3
    num_terms = 6
    tolerance = 0.05
    max_seed = 50

    print(f"Scanning {max_seed} random Hamiltonians for near-degenerate eigenpairs...")
    found_any = False

    for seed in range(max_seed):
        H = build_random_hamiltonian_sparse(
            n_qubits=n_qubits,
            num_terms=num_terms,
            seed=seed
        )

        evals, evecs, pairs = find_near_degenerate_pairs(H, tolerance=tolerance)

        if pairs:
            found_any = True
            print("\n=======================================")
            print(f"Seed {seed} produced near-degenerate pairs")
            print("Hamiltonian:")
            print(H)
            print("Eigenvalues:")
            for i, ev in enumerate(evals):
                print(f"  {i}: {ev:.4f}")

            print("\nCandidate near-degenerate pairs:")
            for (i, j, ei, ej) in pairs:
                print(f"  Pair ({i}, {j}) with energies {ei:.4f}, {ej:.4f}")
                print("   State", i)
                print_statevector(evecs[:, i], n_qubits=n_qubits)
                print("   State", j)
                print_statevector(evecs[:, j], n_qubits=n_qubits)
                print()

    if not found_any:
        print("\nNo near-degenerate pairs found. Try increasing tolerance.")

    print("\n--- End of v0.4 experiment ---")

sys.stdout.close()
