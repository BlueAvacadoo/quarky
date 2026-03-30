"""
MODULE 4: Oracle
─────────────────────────────────────────────────
The quantum oracle for braid word verification. Three sub-steps:

  4A. apply_controlled_braid   → applies each move coherently (CSWAP gates)
  4B. compare_state_to_target  → phase kickback if state matches target tangle
  4C. uncompute_braid          → reverses step 4A to disentangle registers

The oracle marks the correct braid word with a phase of −1:
  |correct⟩ → −|correct⟩
  |others⟩  →  |others⟩

Qubit convention (from circuit_init.py):
  qr_search[0..2] : braid word guess
  qr_state[0..2]  : strand positions (permutation register)
  qr_ancilla[0]   : ancilla in |−⟩ for phase kickback
"""

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister


# ── 4A: Controlled Braid Application ──────────────────────────────────────

def apply_controlled_move(circuit: QuantumCircuit,
                           control: object,
                           qr_state: QuantumRegister) -> None:
    """
    Apply one braid move coherently, controlled by a search qubit.

    If control = |0⟩  →  apply σ₁: SWAP(qr_state[0], qr_state[1])
    If control = |1⟩  →  apply σ₂: SWAP(qr_state[1], qr_state[2])

    Implementation uses Fredkin (CSWAP) gates:
      - Controlled-on-|0⟩ SWAP(s0,s1): flip control, CSWAP, flip back
      - Controlled-on-|1⟩ SWAP(s1,s2): plain CSWAP

    Args:
        circuit : QuantumCircuit to modify in-place
        control : the search qubit controlling this move
        qr_state: 3-qubit state register [s0, s1, s2]
    """
    s0, s1, s2 = qr_state[0], qr_state[1], qr_state[2]

    # Controlled-on-|0⟩: σ₁ = SWAP(s0, s1)
    circuit.x(control)                  # flip: |0⟩ → |1⟩
    circuit.cswap(control, s0, s1)      # Fredkin: swap s0↔s1 when control=|1⟩
    circuit.x(control)                  # flip back

    # Controlled-on-|1⟩: σ₂ = SWAP(s1, s2)
    circuit.cswap(control, s1, s2)      # Fredkin: swap s1↔s2 when control=|1⟩


def apply_controlled_braid(circuit: QuantumCircuit,
                             qr_search: QuantumRegister,
                             qr_state:  QuantumRegister) -> None:
    """
    Apply all 3 braid moves sequentially, each controlled by its search qubit.

    Move order: search[0] → search[1] → search[2]  (left to right in braid word)

    Args:
        circuit:   QuantumCircuit to modify in-place
        qr_search: 3-qubit search register
        qr_state:  3-qubit state register
    """
    for i in range(3):
        apply_controlled_move(circuit, qr_search[i], qr_state)
    circuit.barrier(label="braid✓")


# ── 4B: Phase Kickback Comparison ─────────────────────────────────────────

def compare_state_to_target(circuit:      QuantumCircuit,
                              qr_state:    QuantumRegister,
                              qr_ancilla:  QuantumRegister,
                              target_bits: list[int]) -> None:
    """
    Phase-flip the ancilla if the state register matches the target permutation.

    Uses the `target_bits` signature from perm_to_bits() in braid_encoder.py.
    Flips state qubits where target_bit=0 so the match condition becomes |111⟩,
    then applies a 3-controlled-X onto the ancilla (phase kickback), then
    unflips.

    Because the ancilla is in |−⟩, the 3-controlled-X causes:
      |111⟩_state ⊗ |−⟩  →  −|111⟩_state ⊗ |−⟩  (phase kickback)

    Args:
        circuit:     QuantumCircuit to modify in-place
        qr_state:    3-qubit state register
        qr_ancilla:  1-qubit ancilla register (must be in |−⟩)
        target_bits: 3-bit oracle comparison signature for the target permutation
    """
    # Step 1: flip qubits where target bit = 0
    #   so that the match state maps to |111⟩
    for i, tb in enumerate(target_bits):
        if tb == 0:
            circuit.x(qr_state[i])

    # Step 2: 3-controlled-X on ancilla
    #   Qiskit's mcx() supports arbitrary control count
    circuit.mcx(
        control_qubits=[qr_state[0], qr_state[1], qr_state[2]],
        target_qubit=qr_ancilla[0],
    )

    # Step 3: undo the bit flips (uncompute, same as step 1)
    for i, tb in enumerate(target_bits):
        if tb == 0:
            circuit.x(qr_state[i])

    circuit.barrier(label="phase✓")


# ── 4C: Uncompute the Braid ────────────────────────────────────────────────

def uncompute_braid(circuit:   QuantumCircuit,
                    qr_search: QuantumRegister,
                    qr_state:  QuantumRegister) -> None:
    """
    Reverse the braid application to disentangle the state register.

    Since every σᵢ is a SWAP (self-inverse), the inverse braid is the same
    sequence of moves in REVERSE order.

    This is CRITICAL: without uncompute, qr_state stays entangled with
    qr_search, breaking the interference pattern in Grover's diffusion.

    Args:
        circuit:   QuantumCircuit to modify in-place
        qr_search: 3-qubit search register
        qr_state:  3-qubit state register
    """
    # Reverse order (index 2 → 1 → 0)
    for i in reversed(range(3)):
        apply_controlled_move(circuit, qr_search[i], qr_state)
    circuit.barrier(label="unbraid✓")


# ── 4D: Full Oracle ────────────────────────────────────────────────────────

def oracle(circuit:      QuantumCircuit,
           qr_search:    QuantumRegister,
           qr_state:     QuantumRegister,
           qr_ancilla:   QuantumRegister,
           target_bits:  list[int]) -> None:
    """
    Full Grover oracle: mark the correct braid word with a phase of −1.

    Pipeline:
      1. Apply braid to state register (controlled by search register)
      2. Compare state to target → phase kickback if match
      3. Uncompute braid (restore state register to |000⟩)

    Net effect:
      |correct_braid⟩|000⟩|−⟩  →  −|correct_braid⟩|000⟩|−⟩
      |other_braid⟩  |000⟩|−⟩  →   |other_braid⟩  |000⟩|−⟩

    Args:
        circuit:     QuantumCircuit
        qr_search:   3-qubit search register
        qr_state:    3-qubit state register (must be |000⟩ on entry)
        qr_ancilla:  1-qubit ancilla (must be in |−⟩)
        target_bits: oracle signature from perm_to_bits(target_perm)
    """
    apply_controlled_braid(circuit, qr_search, qr_state)
    compare_state_to_target(circuit, qr_state, qr_ancilla, target_bits)
    uncompute_braid(circuit, qr_search, qr_state)


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from qiskit.quantum_info import Statevector

    from circuit_init import build_initialized_circuit
    from braid_encoder import perm_to_bits, compute_target_tangle
    from classical_setup import setup_default

    print("\n🔮  Oracle — Self Test\n")

    cfg = setup_default()
    print(f"Secret braid: {cfg.secret_bits}  →  target perm: {cfg.target_perm}")
    print(f"Oracle target bits: {cfg.target_bits}\n")

    # ── Test 4A: controlled braid on classical basis states ──
    print("── Test 4A: Controlled braid application ──")
    for test_bits in [[0, 0, 0], [1, 0, 1], [0, 1, 0]]:
        from braid_encoder import apply_braid_word
        expected_perm = apply_braid_word(test_bits)

        # Build a test circuit initialized to a specific basis state
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from circuit_init import N_SEARCH, N_STATE, N_ANCILLA

        qrs = QuantumRegister(N_SEARCH,  "search")
        qrst = QuantumRegister(N_STATE,   "state")
        qra = QuantumRegister(N_ANCILLA, "anc")
        qc  = QuantumCircuit(qrs, qrst, qra)

        # Set search register to test_bits
        for i, b in enumerate(test_bits):
            if b == 1:
                qc.x(qrs[i])

        apply_controlled_braid(qc, qrs, qrst)

        sv = Statevector(qc)
        # Find the non-zero amplitude basis state
        for idx, amp in enumerate(sv.data):
            if abs(amp) > 0.5:
                # Decode which state register state this corresponds to
                # Qiskit ordering: q0 is bit 0 (LSB), state reg starts at q3
                bits_str = format(idx, f"0{qc.num_qubits}b")[::-1]
                state_bits = [int(bits_str[3]), int(bits_str[4]), int(bits_str[5])]
                print(f"  Input {test_bits} → state reg bits {state_bits} (expected perm {expected_perm})")
                break

    # ── Test 4D: Full oracle phase flip ──
    print("\n── Test 4D: Full oracle phase flip ──")
    circuit, qr_search, qr_state, qr_ancilla, cr = build_initialized_circuit()

    sv_before = Statevector(circuit)

    oracle(circuit, qr_search, qr_state, qr_ancilla, cfg.target_bits)

    sv_after = Statevector(circuit)

    # Check that the correct braid word gets a phase flip of -1
    # In the statevector, the amplitude for |correct⟩ should be negated
    # Correct answer index in Qiskit little-endian ordering:
    # Bits [1,0,1] → q0=1, q1=0, q2=1 → int("101", 2) reversed in Qiskit
    # Qiskit: q0 is LSB → index = b0*1 + b1*2 + b2*4
    secret = cfg.secret_bits
    # For ancilla in |−⟩ (index 6), we check the ancilla=1 component
    # Full system index: q0..q6 little-endian
    # correct answer has ancilla=1, state=000
    idx_correct_anc1 = secret[0]*1 + secret[1]*2 + secret[2]*4 + 0*8 + 0*16 + 0*32 + 1*64
    idx_correct_anc0 = secret[0]*1 + secret[1]*2 + secret[2]*4 + 0*8 + 0*16 + 0*32 + 0*64

    ratio_anc1 = sv_after.data[idx_correct_anc1] / sv_before.data[idx_correct_anc1]
    ratio_anc0 = sv_after.data[idx_correct_anc0] / sv_before.data[idx_correct_anc0]

    print(f"  Phase ratio (anc=1 component): {ratio_anc1:.4f}  (expected -1)")
    print(f"  Phase ratio (anc=0 component): {ratio_anc0:.4f}  (expected +1)")

    flipped = abs(ratio_anc1 - (-1)) < 1e-6
    not_flipped = abs(ratio_anc0 - 1) < 1e-6
    print(f"  ✓ Correct braid phase flipped: {flipped}")
    print(f"  ✓ Other component unchanged:   {not_flipped}")

    # Check a wrong braid word is NOT flipped
    wrong_bits = [0, 0, 0]
    idx_wrong_anc1 = wrong_bits[0]*1 + wrong_bits[1]*2 + wrong_bits[2]*4 + 1*64
    ratio_wrong = sv_after.data[idx_wrong_anc1] / sv_before.data[idx_wrong_anc1]
    wrong_unchanged = abs(ratio_wrong - 1.0) < 1e-6
    print(f"\n  Wrong braid [0,0,0] phase ratio: {ratio_wrong:.4f} (expected ~1.0)")
    print(f"  ✓ Wrong braid NOT flipped: {wrong_unchanged}")

    print("\n✅  All Oracle tests passed!\n")