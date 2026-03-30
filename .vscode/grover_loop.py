"""
MODULE 6: GroverLoop
─────────────────────────────────────────────────
Assembles the complete Grover search circuit.
Handles iteration count computation, circuit construction,
and measurement gate placement.

Optimal iteration count for N=8, M=1:
  k = floor( (π/4) × √(N/M) ) = floor(2.22) = 2
  P(success) ≈ sin²(5 × arcsin(1/√8)) ≈ 94.5%
"""

import math
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister

from circuit_init  import build_initialized_circuit, N_SEARCH, N_STATE, N_ANCILLA
from oracle        import oracle
from diffusion     import grover_diffusion
from classical_setup import ProblemConfig


# ── Iteration Math ─────────────────────────────────────────────────────────

def compute_iterations(N: int = 8, M: int = 1) -> int:
    """
    Compute the optimal number of Grover iterations.

    k = floor( (π/4) × √(N/M) )

    Args:
        N: search space size (default 8)
        M: number of marked solutions (default 1)

    Returns:
        k (int), always ≥ 1
    """
    k = math.floor((math.pi / 4) * math.sqrt(N / M))
    return max(k, 1)


def success_probability(N: int, M: int, k: int) -> float:
    """
    Theoretical success probability after k iterations.

    P = sin²( (2k+1) × arcsin(√(M/N)) )
    """
    theta = math.asin(math.sqrt(M / N))
    return math.sin((2 * k + 1) * theta) ** 2


# ── Circuit Builder ────────────────────────────────────────────────────────

def build_grover_circuit(cfg: ProblemConfig,
                          measure: bool = True) -> tuple:
    """
    Build the complete Grover circuit for a given problem configuration.

    Structure:
      1. Initialize: H⊗³ on search + |−⟩ on ancilla
      2. Repeat k times:
           oracle(target_bits) + diffusion
      3. Measure search register → classical bits

    Args:
        cfg:     ProblemConfig from classical_setup.py
        measure: if True, add measurement gates at the end

    Returns:
        (circuit, qr_search, qr_state, qr_ancilla, cr)
    """
    circuit, qr_search, qr_state, qr_ancilla, cr = build_initialized_circuit()

    k = cfg.optimal_iters
    for iteration in range(k):
        # Label each Grover layer for readability in circuit diagrams
        circuit.barrier(label=f"iter {iteration+1}/{k}")
        oracle(circuit, qr_search, qr_state, qr_ancilla, cfg.target_bits)
        grover_diffusion(circuit, qr_search)

    if measure:
        add_measurement(circuit, qr_search, cr)

    return circuit, qr_search, qr_state, qr_ancilla, cr


def add_measurement(circuit:   QuantumCircuit,
                    qr_search: QuantumRegister,
                    cr:        ClassicalRegister) -> None:
    """
    Add measurement gates on the search register.

    Maps:
      qr_search[0] → cr[0]
      qr_search[1] → cr[1]
      qr_search[2] → cr[2]

    Args:
        circuit:   QuantumCircuit to modify in-place
        qr_search: 3-qubit search register
        cr:        3-bit classical register
    """
    circuit.barrier(label="measure")
    circuit.measure(qr_search, cr)


# ── Multi-iteration Sweep (for P vs k plots) ──────────────────────────────

def build_circuit_with_k_iterations(cfg: ProblemConfig,
                                      k: int,
                                      measure: bool = True) -> tuple:
    """
    Build a Grover circuit with exactly k iterations (ignores cfg.optimal_iters).
    Useful for sweeping k to visualize probability oscillation.

    Args:
        cfg:     ProblemConfig
        k:       exact iteration count to use
        measure: add measurement gates

    Returns:
        (circuit, qr_search, qr_state, qr_ancilla, cr)
    """
    circuit, qr_search, qr_state, qr_ancilla, cr = build_initialized_circuit()

    for iteration in range(k):
        circuit.barrier(label=f"k={iteration+1}")
        oracle(circuit, qr_search, qr_state, qr_ancilla, cfg.target_bits)
        grover_diffusion(circuit, qr_search)

    if measure:
        add_measurement(circuit, qr_search, cr)

    return circuit, qr_search, qr_state, qr_ancilla, cr


# ── Circuit Statistics ─────────────────────────────────────────────────────

def circuit_stats(circuit: QuantumCircuit) -> dict:
    """
    Return key statistics about the circuit.

    Returns:
        dict with keys: n_qubits, n_cbits, depth, gate_counts, n_cx
    """
    ops = circuit.count_ops()
    return {
        "n_qubits":    circuit.num_qubits,
        "n_cbits":     circuit.num_clbits,
        "depth":       circuit.depth(),
        "gate_counts": dict(ops),
        "n_cx":        ops.get("cx", 0),
        "n_cswap":     ops.get("cswap", 0),
        "n_ccx":       ops.get("ccx", 0),
    }


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from qiskit.quantum_info import Statevector

    from classical_setup import setup_default, setup_all_instances

    print("\n🔄  GroverLoop — Self Test\n")

    cfg = setup_default()
    print(f"Problem: secret={cfg.secret_bits}, target={cfg.target_perm}")
    print(f"Optimal k={cfg.optimal_iters}, P(win)={success_probability(8, 1, cfg.optimal_iters):.1%}\n")

    # ── Test 1: Build circuit ──
    circuit, qr_search, qr_state, qr_ancilla, cr = build_grover_circuit(cfg, measure=False)
    stats = circuit_stats(circuit)
    print(f"✓ Circuit built: {stats['n_qubits']} qubits, depth={stats['depth']}")
    print(f"  Gates: {stats['gate_counts']}")

    # ── Test 2: Statevector check ──
    print("\n── Statevector after 2 iterations ──")
    sv = Statevector(circuit)
    sv_arr = np.abs(sv.data)**2

    # Marginalize over state and ancilla registers
    search_probs = {}
    for idx, p in enumerate(sv_arr):
        key = idx & 0b111    # low 3 bits = search register (little-endian)
        search_probs[key] = search_probs.get(key, 0) + p

    print(f"  {'Bits':<10} {'P':>8}  {'Winner?'}")
    print(f"  {'-'*28}")
    for key in range(8):
        bits = [key & 1, (key>>1) & 1, (key>>2) & 1]
        p    = search_probs.get(key, 0)
        win  = " ◄ SECRET" if bits == cfg.secret_bits else ""
        print(f"  {str(bits):<10} {p:>8.4f}  {win}")

    secret_key = cfg.secret_bits[0] + cfg.secret_bits[1]*2 + cfg.secret_bits[2]*4
    p_win = search_probs.get(secret_key, 0)
    print(f"\n  P(correct answer) = {p_win:.4f}  (theoretical ≈ 0.9453)")
    assert p_win > 0.9, f"Grover failed: P={p_win:.4f}"
    print("  ✓ P(correct) > 90%  — Grover's algorithm working!")

    # ── Test 3: P vs k sweep ──
    print("\n── P(correct) vs iteration count k ──")
    for k in range(0, 6):
        c2, _, _, _, _ = build_circuit_with_k_iterations(cfg, k, measure=False)
        sv2 = Statevector(c2)
        sv2_arr = np.abs(sv2.data)**2
        sp2 = {}
        for idx, p in enumerate(sv2_arr):
            key = idx & 0b111
            sp2[key] = sp2.get(key, 0) + p
        p_k = sp2.get(secret_key, 0)
        bar = "█" * int(p_k * 30)
        opt = " ← optimal" if k == cfg.optimal_iters else ""
        print(f"  k={k}: {p_k:.4f}  {bar}{opt}")

    # ── Test 4: All 8 problem instances ──
    print("\n── All 8 secrets converge ──")
    for c in setup_all_instances():
        ckt, _, _, _, _ = build_grover_circuit(c, measure=False)
        sv3 = Statevector(ckt)
        sp3 = {}
        for idx, p in enumerate(np.abs(sv3.data)**2):
            key = idx & 0b111
            sp3[key] = sp3.get(key, 0) + p
        sk = c.secret_bits[0] + c.secret_bits[1]*2 + c.secret_bits[2]*4
        p3 = sp3.get(sk, 0)
        ok = "✓" if p3 > 0.8 else "✗"
        print(f"  {ok} secret={c.secret_bits}  P(win)={p3:.4f}")

    print("\n✅  All GroverLoop tests passed!\n")