"""
v0_5.py  –  Backend-ready prototype for physics-native subspace discovery

Goal
----
1. Reuse the v0.4 idea: generate a 3-qubit Pauli Hamiltonian and find a
   near-degenerate eigenpair.
2. Take ONE of those eigenstates and:
   - prepare it as a quantum circuit,
   - run it either on a local simulator or on an IBM backend
     (Heron or similar),
   - compare measured probabilities with the ideal amplitudes.

You can switch between LOCAL and IBM execution with the BACKEND_MODE flag
near the bottom of this file.
"""

import sys
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

# Optional imports (local and IBM)
try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    QiskitRuntimeService = None


# --------------------------------------------------------------------
# Helper: Tee stdout to both console and file
# --------------------------------------------------------------------

class Tee:
    """Simple 'tee' stream: skriver til flere streams på én gang."""
    def __init__(self, *streams):
        self.streams = streams
        # Brug encoding fra første stream, hvis den findes
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()



# --------------------------------------------------------------------
# 1. Hamiltonian construction
# --------------------------------------------------------------------

def build_random_hamiltonian_sparse(
    n_qubits: int = 3,
    num_terms: int = 5,
    seed: int = 31
) -> SparsePauliOp:
    """
    Build a random n-qubit Hamiltonian as a SparsePauliOp:

        H = sum_j c_j P_j     ,  P_j in {I,X,Y,Z}^n

    The seed makes the instance reproducible.
    """
    rng = np.random.default_rng(seed)
    paulis_single = ["I", "X", "Y", "Z"]

    pauli_strings = []
    coeffs = []

    for _ in range(num_terms):
        s = "".join(rng.choice(paulis_single) for _ in range(n_qubits))
        # Avoid the trivial all-identity term
        if set(s) == {"I"}:
            continue
        pauli_strings.append(s)
        coeffs.append(float(rng.uniform(-1.0, 1.0)))

    H = SparsePauliOp(pauli_strings, coeffs=coeffs)
    return H


# --------------------------------------------------------------------
# 2. Diagonalisation and near-degenerate pair detection
# --------------------------------------------------------------------

def find_near_degenerate_pair(
    H: SparsePauliOp,
    epsilon: float = 0.05
):
    """
    Diagonalise H and search for the closest pair of distinct eigenvalues.

    Returns:
        (E_vals, E_vecs, pair_indices)
        where pair_indices is (i,j) or None if no pair is closer than epsilon.
    """
    H_mat = H.to_matrix()
    E_vals, E_vecs = np.linalg.eigh(H_mat)

    n = len(E_vals)
    best_delta = None
    best_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(E_vals[i] - E_vals[j])
            if delta == 0:
                # exact degeneracy – take immediately
                best_delta = 0.0
                best_pair = (i, j)
                break
            if (best_delta is None or delta < best_delta) and delta < epsilon:
                best_delta = delta
                best_pair = (i, j)
        if best_pair is not None and best_delta == 0.0:
            break

    return E_vals, E_vecs, best_pair


def print_eigenpair_info(E_vals, E_vecs, pair):
    """
    Pretty-print eigenvalues and dominant components of the selected pair.
    """
    i, j = pair
    print("\n=== Selected near-degenerate pair ===")
    print(f"indices: {i}, {j}")
    print(f"energies: {E_vals[i]: .6f}, {E_vals[j]: .6f}")
    print("--------------------------------------")

    for label, idx in [("State A", i), ("State B", j)]:
        vec = E_vecs[:, idx]
        print(f"{label} (index {idx}):")
        # Sort basis states by amplitude magnitude
        mags = np.abs(vec)
        order = np.argsort(mags)[::-1]
        for k in order[:8]:
            amp = vec[k]
            bitstring = format(k, f"0{int(np.log2(len(vec)))}b")
            print(f"  |{bitstring}> : {amp.real:+.3f}{amp.imag:+.3f}j   "
                  f"(p = {mags[k]**2:.3f})")
        print()


# --------------------------------------------------------------------
# 3. Circuit construction for one eigenstate
# --------------------------------------------------------------------

def build_state_preparation_circuit(eigvec: np.ndarray) -> QuantumCircuit:
    """
    Build a QuantumCircuit that prepares the given eigenvector.

    We use the generic 'initialize' method. This is not optimised, but
    it is simple and backend-agnostic.
    """
    n_qubits = int(np.log2(len(eigvec)))
    qc = QuantumCircuit(n_qubits)
    qc.initialize(eigvec, list(range(n_qubits)))
    qc.measure_all()
    return qc


# --------------------------------------------------------------------
# 4. Execution backends
# --------------------------------------------------------------------

def run_local(qc: QuantumCircuit, shots: int = 4096):
    """
    Run the circuit on a local AerSimulator (if available).
    """
    if AerSimulator is None:
        raise ImportError("qiskit-aer is not installed in this environment.")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    job = sim.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def run_ibm(
    qc: QuantumCircuit,
    backend_name: str = "ibm_fe_z",
    shots: int = 4096
):
    """
    Run the circuit on an IBM backend via QiskitRuntimeService.

    NOTE:
      - You must have 'qiskit-ibm-runtime' installed.
      - You must have previously saved your IBM Quantum account, e.g.:

            from qiskit_ibm_runtime import QiskitRuntimeService
            QiskitRuntimeService.save_account(channel='cloud',
                                              token='YOUR_API_TOKEN')

    """
    if QiskitRuntimeService is None:
        raise ImportError(
            "qiskit-ibm-runtime is not available. "
            "Install it with: pip install qiskit-ibm-runtime"
        )

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=shots)
    print(f"Submitted job to backend {backend_name}, job ID = {job.job_id()}")
    result = job.result()
    counts = result.get_counts()
    return counts


# --------------------------------------------------------------------
# 5. Main experiment driver
# --------------------------------------------------------------------

def main():
    # -----------------------
    # Configuration section
    # -----------------------
    BACKEND_MODE = "local"   # "local" or "ibm"
    IBM_BACKEND_NAME = "ibm_fe_z"
    SEED = 31
    N_QUBITS = 3
    NUM_TERMS = 5
    EPSILON = 0.05
    SHOTS = 4096

    print("=== v0.5 – Backend-ready prototype ===")
    print(f"Backend mode        : {BACKEND_MODE}")
    print(f"Random seed         : {SEED}")
    print(f"Number of qubits    : {N_QUBITS}")
    print(f"Number of Pauli terms in H : {NUM_TERMS}")
    print(f"Near-degeneracy eps : {EPSILON}\n")

    # 1) Build Hamiltonian
    H = build_random_hamiltonian_sparse(
        n_qubits=N_QUBITS,
        num_terms=NUM_TERMS,
        seed=SEED
    )
    print("Hamiltonian (SparsePauliOp):")
    print(H)
    print()

    # 2) Diagonalise and find near-degenerate pair
    E_vals, E_vecs, pair = find_near_degenerate_pair(H, epsilon=EPSILON)

    print("Eigenvalues:")
    for idx, E in enumerate(E_vals):
        print(f"  {idx}: {E: .6f}")
    print()

    if pair is None:
        print("No near-degenerate pair found with the chosen epsilon.")
        return

    print_eigenpair_info(E_vals, E_vecs, pair)

    # Choose the first state in the pair as the target eigenstate
    target_index = pair[0]
    target_vec = E_vecs[:, target_index]

    # 3) Build preparation circuit
    qc = build_state_preparation_circuit(target_vec)
    print("State-preparation circuit:")
    print(qc)
    print()

    # 4) Run on selected backend
    if BACKEND_MODE == "local":
        counts = run_local(qc, shots=SHOTS)
    elif BACKEND_MODE == "ibm":
        counts = run_ibm(qc, backend_name=IBM_BACKEND_NAME, shots=SHOTS)
    else:
        raise ValueError("BACKEND_MODE must be 'local' or 'ibm'.")

    # 5) Compare with ideal probabilities
    print("\n=== Measurement statistics ===")
    print("Backend counts:")
    for bitstring, c in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  {bitstring}: {c}")

    print("\nIdeal probabilities from eigenvector:")
    probs = np.abs(target_vec) ** 2
    for k, p in enumerate(probs):
        if p < 1e-4:
            continue
        b = format(k, f"0{N_QUBITS}b")
        print(f"  |{b}> : {p:.4f}")

    print("\nv0.5 run finished.")


if __name__ == "__main__":
    # Alt output sendes både til konsol og fil
    output_filename = "v0_5_output.txt"   # evt. ændr sti/navn efter smag
    with open(output_filename, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        original_stdout = sys.stdout
        sys.stdout = tee
        try:
            main()
        finally:
            sys.stdout = original_stdout
