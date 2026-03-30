"""
MODULE 3: CircuitInit
─────────────────────────────────────────────────
Quantum circuit construction, register allocation, and initialization.
Builds the 7-qubit circuit and prepares the starting state:
  - Search register  [q0–q2] → uniform superposition (H gates)
  - State register   [q3–q5] → |012⟩ (identity permutation, starts as |000⟩)
  - Ancilla qubit    [q6]    → |−⟩ = (|0⟩−|1⟩)/√2 for phase kickback

Qubit layout:
  ┌─────────────────────────────────────────────────┐
  │  q0  q1  q2   │   q3  q4  q5   │   q6           │
  │  Search reg   │   State reg    │   Ancilla      │
  │  (braid word) │   (strands)    │   (kickback)   │
  └─────────────────────────────────────────────────┘
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ── Register Sizes ─────────────────────────────────────────────────────────

N_SEARCH  = 3    # qubits for the braid word guess
N_STATE   = 3    # qubits for the strand state register
N_ANCILLA = 1    # ancilla for phase kickback
N_CBITS   = 3    # classical bits (measure search register only)

# Qubit indices (absolute, in the flat circuit)
IDX_SEARCH  = list(range(0, 3))          # [0, 1, 2]
IDX_STATE   = list(range(3, 6))          # [3, 4, 5]
IDX_ANCILLA = 6                          # 6


# ── Circuit Factory ────────────────────────────────────────────────────────

def build_registers():
    """
    Create named Qiskit registers for clarity in circuit diagrams.

    Returns:
        (qr_search, qr_state, qr_ancilla, cr)
    """
    qr_search  = QuantumRegister(N_SEARCH,  name="search")
    qr_state   = QuantumRegister(N_STATE,   name="state")
    qr_ancilla = QuantumRegister(N_ANCILLA, name="anc")
    cr         = ClassicalRegister(N_CBITS, name="out")
    return qr_search, qr_state, qr_ancilla, cr


def initialize_circuit() -> tuple:
    """
    Build the base 7-qubit circuit with named registers.

    Returns:
        (circuit, qr_search, qr_state, qr_ancilla, cr)
        where circuit is an empty QuantumCircuit ready for gates.
    """
    qr_search, qr_state, qr_ancilla, cr = build_registers()
    circuit = QuantumCircuit(qr_search, qr_state, qr_ancilla, cr)
    return circuit, qr_search, qr_state, qr_ancilla, cr


def initialize_superposition(circuit: QuantumCircuit,
                              qr_search: QuantumRegister) -> None:
    """
    Put the search register in uniform superposition over all 8 braid words.

    Applies H to each search qubit:
      |000⟩ → (1/√8) Σ_{b∈{0,1}³} |b⟩

    Args:
        circuit:   the QuantumCircuit to modify in-place
        qr_search: the 3-qubit search register
    """
    circuit.h(qr_search)
    circuit.barrier(label="superposition")


def initialize_ancilla(circuit: QuantumCircuit,
                        qr_ancilla: QuantumRegister) -> None:
    """
    Prepare the ancilla in |−⟩ = (|0⟩ − |1⟩)/√2 for phase kickback.

    |0⟩ → X → |1⟩ → H → |−⟩

    Args:
        circuit:    the QuantumCircuit to modify in-place
        qr_ancilla: the 1-qubit ancilla register
    """
    circuit.x(qr_ancilla[0])
    circuit.h(qr_ancilla[0])
    circuit.barrier(label="ancilla|−⟩")


def initialize_full(circuit: QuantumCircuit,
                     qr_search: QuantumRegister,
                     qr_state:  QuantumRegister,
                     qr_ancilla: QuantumRegister) -> None:
    """
    Full initialization: superposition + ancilla.
    State register needs no explicit init (starts in |000⟩ = identity perm).

    Args:
        circuit, qr_search, qr_state, qr_ancilla: from initialize_circuit()
    """
    initialize_superposition(circuit, qr_search)
    initialize_ancilla(circuit, qr_ancilla)


# ── Convenience Builder ────────────────────────────────────────────────────

def build_initialized_circuit() -> tuple:
    """
    One-call helper: build circuit and apply full initialization.

    Returns:
        (circuit, qr_search, qr_state, qr_ancilla, cr)
        Circuit is initialized and ready for oracle + diffusion.
    """
    circuit, qr_search, qr_state, qr_ancilla, cr = initialize_circuit()
    initialize_full(circuit, qr_search, qr_state, qr_ancilla)
    return circuit, qr_search, qr_state, qr_ancilla, cr


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    import numpy as np

    print("\n🔧  CircuitInit — Self Test\n")

    # Test 1: circuit builds without errors
    circuit, qr_search, qr_state, qr_ancilla, cr = build_initialized_circuit()
    print(f"✓ Circuit created: {circuit.num_qubits} qubits, {circuit.num_clbits} cbits")
    print(f"  Search : {[circuit.find_bit(q).index for q in qr_search]}")
    print(f"  State  : {[circuit.find_bit(q).index for q in qr_state]}")
    print(f"  Ancilla: {circuit.find_bit(qr_ancilla[0]).index}")

    # Test 2: statevector check
    sv = Statevector(circuit)
    probs = sv.probabilities()

    # Search register should be flat (8 terms × 3 state qubits × 2 ancilla = 128 amps)
    # |search⟩|000⟩|−⟩  — each of the 8 search states has prob 1/8 on ancilla=1 side
    n_nonzero = np.count_nonzero(np.abs(sv.data) > 1e-10)
    print(f"\n✓ Statevector has {n_nonzero} non-zero amplitudes (expected 16)")
    print(f"  (8 search states × 2 ancilla states from |−⟩)")

    # Test 3: search register marginal is uniform
    # Sum over state+ancilla dimensions for each search state
    sv_reshaped = sv.data.reshape(2, 2, 2, 2, 2, 2, 2)
    # axes: q0 q1 q2 | q3 q4 q5 | q6  (Qiskit little-endian: q0 is rightmost)
    # Marginal probability on search qubits
    search_probs = np.abs(sv_reshaped) ** 2
    search_marginal = search_probs.sum(axis=(3, 4, 5, 6))   # sum over state+ancilla
    expected = 1 / 8
    max_dev = np.max(np.abs(search_marginal - expected))
    print(f"✓ Search marginal uniform: max deviation = {max_dev:.2e} (expected ~0)")
    assert max_dev < 1e-10, "Search register not uniform!"

    # Test 4: draw circuit
    print("\n📊  Circuit diagram:")
    print(circuit.draw(output="text", fold=80))

    print("\n✅  All CircuitInit tests passed!\n")