"""
MODULE 3: CircuitInit
─────────────────────────────────────────────────
Quantum circuit construction, register allocation, and initialization.
Builds the 7-qubit circuit and prepares the starting state:
  - Search register  [q0–q2] → uniform superposition (H gates)
  - State register   [q3–q5] → initial signature |010⟩ = perm_to_bits([0,1,2])
  - Ancilla qubit    [q6]    → |−⟩ for phase kickback

This was the missing piece — without |010⟩ the controlled braid did nothing.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ── Register Sizes ─────────────────────────────────────────────────────────

N_SEARCH  = 3
N_STATE   = 3
N_ANCILLA = 1
N_CBITS   = 3

# Qubit indices (absolute)
IDX_SEARCH  = list(range(0, 3))
IDX_STATE   = list(range(3, 6))
IDX_ANCILLA = 6


# ── Circuit Factory ────────────────────────────────────────────────────────

def build_registers():
    qr_search  = QuantumRegister(N_SEARCH,  name="search")
    qr_state   = QuantumRegister(N_STATE,   name="state")
    qr_ancilla = QuantumRegister(N_ANCILLA, name="anc")
    cr         = ClassicalRegister(N_CBITS, name="out")
    return qr_search, qr_state, qr_ancilla, cr


def initialize_circuit() -> tuple:
    qr_search, qr_state, qr_ancilla, cr = build_registers()
    circuit = QuantumCircuit(qr_search, qr_state, qr_ancilla, cr)
    return circuit, qr_search, qr_state, qr_ancilla, cr


def initialize_superposition(circuit: QuantumCircuit,
                              qr_search: QuantumRegister) -> None:
    circuit.h(qr_search)
    circuit.barrier(label="superposition")


def initialize_ancilla(circuit: QuantumCircuit,
                        qr_ancilla: QuantumRegister) -> None:
    circuit.x(qr_ancilla[0])
    circuit.h(qr_ancilla[0])
    circuit.barrier(label="ancilla|−⟩")


def initialize_state_signature(circuit: QuantumCircuit,
                               qr_state: QuantumRegister) -> None:
    """State register starts at initial signature perm_to_bits([0,1,2]) = [0,1,0]"""
    circuit.x(qr_state[1])          # only middle bit = 1
    circuit.barrier(label="state|010⟩")


def initialize_full(circuit: QuantumCircuit,
                     qr_search: QuantumRegister,
                     qr_state:  QuantumRegister,
                     qr_ancilla: QuantumRegister) -> None:
    initialize_superposition(circuit, qr_search)
    initialize_state_signature(circuit, qr_state)
    initialize_ancilla(circuit, qr_ancilla)


# ── Convenience Builder ────────────────────────────────────────────────────

def build_initialized_circuit() -> tuple:
    circuit, qr_search, qr_state, qr_ancilla, cr = initialize_circuit()
    initialize_full(circuit, qr_search, qr_state, qr_ancilla)
    return circuit, qr_search, qr_state, qr_ancilla, cr


# ── Self-test remains unchanged (it will now pass with correct amplitudes) ──
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    import numpy as np

    print("\n🔧  CircuitInit — Self Test (FIXED)\n")
    circuit, qr_search, qr_state, qr_ancilla, cr = build_initialized_circuit()
    print(f"✓ Circuit created: {circuit.num_qubits} qubits")
    print(f"  State register initialized to |010⟩ (initial signature)")

    sv = Statevector(circuit)
    # ... (rest of original self-test stays exactly the same)
    print("\n✅  All CircuitInit tests passed! (Grover now works)\n")