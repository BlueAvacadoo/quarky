"""
MODULE 5: Grover Diffusion Operator
─────────────────────────────────────────────────
The Grover diffusion (inversion-about-mean) operator.
Acts only on the search register [q0, q1, q2].

Mathematical definition:
  D = H⊗³ · (2|0⟩⟨0| − I) · H⊗³

Circuit decomposition:
  1. H on all search qubits
  2. X on all search qubits          (map |000⟩ → |111⟩)
  3. Multi-controlled-Z on |111⟩     (phase flip |000⟩ after step 2)
  4. X on all search qubits          (unmap)
  5. H on all search qubits

The multi-controlled-Z is implemented as:
  H on last qubit → CCX (Toffoli) → H on last qubit
  which converts a controlled-Z into a controlled-X.

Effect on amplitudes:
  α_s  →  2⟨α⟩ − α_s
  (reflects each amplitude around the mean ⟨α⟩)

After k iterations, the amplitude of the marked state approaches 1.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister


# ── Diffusion Operator ─────────────────────────────────────────────────────

def grover_diffusion(circuit:   QuantumCircuit,
                     qr_search: QuantumRegister) -> None:
    """
    Apply the Grover diffusion (inversion-about-mean) operator to the
    search register. Acts only on qr_search, leaves other registers untouched.

    Args:
        circuit:   QuantumCircuit to modify in-place
        qr_search: 3-qubit search register [q0, q1, q2]
    """
    q0, q1, q2 = qr_search[0], qr_search[1], qr_search[2]

    # ── Step 1: H⊗³ ────────────────────────────────────────────────────
    circuit.h(qr_search)

    # ── Step 2: X⊗³  (map |000⟩ to |111⟩ for the phase flip) ──────────
    circuit.x(qr_search)

    # ── Step 3: Multi-controlled-Z on |111⟩ ────────────────────────────
    # Implement as H–CCX–H:
    #   CCX(q0, q1, q2) flips q2 when q0=q1=1
    #   Wrapping q2 with H converts the X flip into a Z phase flip
    circuit.h(q2)
    circuit.ccx(q0, q1, q2)    # Toffoli
    circuit.h(q2)

    # ── Step 4: X⊗³ (undo step 2) ──────────────────────────────────────
    circuit.x(qr_search)

    # ── Step 5: H⊗³ ────────────────────────────────────────────────────
    circuit.h(qr_search)

    circuit.barrier(label="diffuse✓")


# ── Amplitude Simulation (classical, for verification) ────────────────────

def simulate_diffusion_classically(amplitudes: list[complex]) -> list[complex]:
    """
    Classically simulate the diffusion operator on a list of amplitudes.
    Useful for verifying the quantum implementation without running a circuit.

    D[α]_s = 2⟨α⟩ − α_s,   where ⟨α⟩ = mean(amplitudes)

    Args:
        amplitudes: list of complex amplitudes (length must be a power of 2)

    Returns:
        new amplitudes after diffusion
    """
    mean = sum(amplitudes) / len(amplitudes)
    return [2 * mean - a for a in amplitudes]


def simulate_grover_iterations(n: int, n_iters: int,
                                 marked_idx: int) -> list[float]:
    """
    Classically simulate Grover iterations and return probability distribution.

    Args:
        n:          search space size (e.g. 8)
        n_iters:    number of Grover iterations
        marked_idx: index of the marked (correct) state

    Returns:
        list of probabilities after n_iters iterations
    """
    import math
    # Initial uniform superposition
    amp = 1.0 / math.sqrt(n)
    amplitudes = [amp] * n

    for _ in range(n_iters):
        # Oracle: negate the marked state
        amplitudes[marked_idx] *= -1

        # Diffusion: invert about mean
        amplitudes = simulate_diffusion_classically(amplitudes)

    return [abs(a)**2 for a in amplitudes]


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from qiskit.quantum_info import Statevector

    from circuit_init import build_initialized_circuit
    from oracle import oracle
    from classical_setup import setup_default

    print("\n🌊  Grover Diffusion — Self Test\n")

    # ── Test 1: Classical amplitude simulation ──
    print("── Test 1: Classical amplitude simulation ──")
    N       = 8
    SECRET  = [1, 0, 1]
    MARKED  = SECRET[0]*1 + SECRET[1]*2 + SECRET[2]*4   # little-endian index = 5

    for k in range(1, 5):
        probs = simulate_grover_iterations(N, k, MARKED)
        p_correct = probs[MARKED]
        print(f"  k={k}: P(correct) = {p_correct:.4f}  "
              f"{'← optimal' if k == 2 else ''}")

    # ── Test 2: Circuit diffusion on uniform state ──
    print("\n── Test 2: Circuit diffusion (no oracle) ──")
    # Without oracle, diffusion should leave the state roughly uniform
    # (inversion about mean of uniform = uniform)
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from circuit_init import N_SEARCH, N_STATE, N_ANCILLA

    qrs  = QuantumRegister(N_SEARCH,  "search")
    qrst = QuantumRegister(N_STATE,   "state")
    qra  = QuantumRegister(N_ANCILLA, "anc")
    qc   = QuantumCircuit(qrs, qrst, qra)
    qc.h(qrs)                          # uniform superposition
    qc.x(qra[0]); qc.h(qra[0])        # ancilla |−⟩

    sv_before = Statevector(qc).probabilities()

    grover_diffusion(qc, qrs)
    sv_after = Statevector(qc).probabilities()

    # Marginal probabilities on search register should still be uniform
    # (diffusion of uniform = uniform)
    print(f"  Max prob deviation after one diffusion (no oracle): "
          f"{max(abs(a-b) for a,b in zip(sv_before, sv_after)):.2e}")

    # ── Test 3: Full oracle + diffusion ──
    print("\n── Test 3: Oracle + Diffusion (1 full Grover step) ──")
    cfg = setup_default()
    circuit, qr_search, qr_state, qr_ancilla, cr = build_initialized_circuit()

    oracle(circuit, qr_search, qr_state, qr_ancilla, cfg.target_bits)
    grover_diffusion(circuit, qr_search)

    sv = Statevector(circuit)
    # Sum probabilities for each search state (marginalize over state+ancilla)
    total_qubits = circuit.num_qubits   # 7
    sv_arr = np.abs(sv.data)**2

    search_probs = {}
    for idx, p in enumerate(sv_arr):
        search_key = idx & 0b111          # low 3 bits = search register
        search_probs[search_key] = search_probs.get(search_key, 0) + p

    print(f"  Probabilities after 1 iteration:")
    for key in range(8):
        bits = [key & 1, (key >> 1) & 1, (key >> 2) & 1]
        marker = " ← SECRET" if bits == cfg.secret_bits else ""
        print(f"    {bits} (idx={key}): {search_probs.get(key, 0):.4f}{marker}")

    p_secret = search_probs.get(
        cfg.secret_bits[0] + cfg.secret_bits[1]*2 + cfg.secret_bits[2]*4, 0
    )
    print(f"\n  P(secret) after 1 step = {p_secret:.4f}")

    # ── Test 4: 2 full iterations ──
    print("\n── Test 4: 2 Full Grover Iterations ──")
    circuit2, qr_search2, qr_state2, qr_ancilla2, cr2 = build_initialized_circuit()

    for _ in range(2):
        oracle(circuit2, qr_search2, qr_state2, qr_ancilla2, cfg.target_bits)
        grover_diffusion(circuit2, qr_search2)

    sv2 = Statevector(circuit2)
    sv2_arr = np.abs(sv2.data)**2

    search_probs2 = {}
    for idx, p in enumerate(sv2_arr):
        search_key = idx & 0b111
        search_probs2[search_key] = search_probs2.get(search_key, 0) + p

    p_secret2 = search_probs2.get(
        cfg.secret_bits[0] + cfg.secret_bits[1]*2 + cfg.secret_bits[2]*4, 0
    )
    print(f"  P(secret) after 2 steps = {p_secret2:.4f}  (theoretical ≈ 0.945)")

    assert p_secret2 > 0.9, f"Expected P > 0.9, got {p_secret2:.4f}"
    print("\n✅  All Diffusion tests passed!\n")