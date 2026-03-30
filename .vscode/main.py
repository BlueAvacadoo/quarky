"""
MODULE 8: Main
─────────────────────────────────────────────────
Top-level execution pipeline for the Quantum Knot Untier.
Wires together all modules end-to-end:

  Phase 0: Classical problem setup
  Phase 1: Build & initialize quantum circuit
  Phase 2: Apply k Grover iterations
  Phase 3: Measure search register
  Phase 4: Decode and verify results
  Phase 5: Display circuit and histograms

Run:
  python main.py                     # default secret [1,0,1]
  python main.py --secret 0 1 0      # custom secret
  python main.py --all               # sweep all 8 secrets
  python main.py --shots 4096        # more shots
"""

import argparse
import sys

from classical_setup import (
    setup_default,
    setup_custom,
    setup_all_instances,
    _success_probability,
)
from grover_loop import (
    build_grover_circuit,
    build_circuit_with_k_iterations,
    circuit_stats,
    compute_iterations,
    success_probability,
)
from result_decoder import (
    run_simulation,
    print_summary,
    verify_solution,
)


# ── Phase Runners ──────────────────────────────────────────────────────────

def phase_0_setup(secret_bits: list[int]):
    """Classical pre-computation."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  PHASE 0 — Classical Problem Setup")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cfg = setup_custom(secret_bits)
    print(cfg.summary())
    return cfg


def phase_1_build(cfg, draw: bool = True):
    """Build and initialize quantum circuit."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  PHASE 1 — Quantum Circuit Initialization")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    circuit, qr_search, qr_state, qr_ancilla, cr = build_grover_circuit(
        cfg, measure=True
    )
    stats = circuit_stats(circuit)

    print(f"  Qubits   : {stats['n_qubits']}  (3 search + 3 state + 1 ancilla)")
    print(f"  Cbits    : {stats['n_cbits']}")
    print(f"  Depth    : {stats['depth']}")
    print(f"  Gates    : {stats['gate_counts']}")

    if draw:
        print("\n  Circuit diagram:")
        print(circuit.draw(output="text", fold=100))

    return circuit, qr_search, qr_state, qr_ancilla, cr


def phase_2_3_grover_and_measure(circuit):
    """Phase 2+3: Grover iterations and measurement are baked into the circuit."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  PHASE 2+3 — Grover Iterations + Measurement")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (Iterations and measurement gates are embedded in the circuit)")
    print("  Ready to execute.\n")


def phase_4_run(circuit, cfg, shots: int = 2048):
    """Execute simulation and decode results."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  PHASE 4 — Simulation & Result Decoding")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Running {shots} shots on StatevectorSampler …")

    summary = run_simulation(circuit, cfg, shots=shots)
    print_summary(summary)
    return summary


def phase_5_verify(summary, cfg):
    """Classical post-verification and final verdict."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  PHASE 5 — Classical Verification")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ok, actual_perm = verify_solution(summary.winner_bits, cfg.target_perm)

    print(f"  Grover's answer  : {summary.winner_bits}")
    print(f"  Produces perm    : {actual_perm}")
    print(f"  Target tangle    : {cfg.target_perm}")
    print(f"  Match            : {'✅  KNOT UNTIED!' if ok else '❌  MISMATCH'}")
    print(f"  Confidence       : {summary.winner_fraction:.1%}")

    if ok:
        from braid_encoder import encode_braid_word
        word = encode_braid_word(summary.winner_bits)
        print(f"\n  🎉  The secret braid word is: {''.join(word)}")
        print(f"       In bits: {summary.winner_bits}")
    else:
        print(f"\n  ⚠️  Quantum answer was wrong this run.")
        print(f"      (Expected ~{success_probability(8, cfg.n_solutions, cfg.optimal_iters):.0%} success rate.)")
        print(f"      Try re-running or increasing shots.")

    return ok


# ── All-instances Sweep ────────────────────────────────────────────────────

def run_all_instances(shots: int = 1024):
    """Run Grover on all 8 possible secrets and report results."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  SWEEP — All 8 Secret Braid Words")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Secret':<10} {'Target':<12} {'Found':<10} {'P(win)':>8}  {'OK?'}")
    print(f"  {'-'*50}")

    all_ok = True
    for cfg in setup_all_instances():
        circuit, *_ = build_grover_circuit(cfg, measure=True)
        summary      = run_simulation(circuit, cfg, shots=shots)
        ok           = summary.winner_correct
        all_ok       = all_ok and ok
        marker       = "✅" if ok else "❌"
        print(
            f"  {str(cfg.secret_bits):<10} "
            f"{str(cfg.target_perm):<12} "
            f"{str(summary.winner_bits):<10} "
            f"{summary.winner_fraction:>8.1%}  "
            f"{marker}"
        )

    print(f"\n  Overall: {'✅ ALL PASSED' if all_ok else '⚠️  SOME FAILED'}")
    return all_ok


# ── Main Entry Point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Knot Untier using Grover's Algorithm"
    )
    parser.add_argument(
        "--secret", nargs=3, type=int, default=[1, 0, 1],
        metavar=("B0", "B1", "B2"),
        help="Secret braid word as 3 bits, e.g. --secret 1 0 1  (default: 1 0 1)"
    )
    parser.add_argument(
        "--shots", type=int, default=2048,
        help="Number of simulation shots (default: 2048)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 8 possible secrets instead of a single one"
    )
    parser.add_argument(
        "--no-draw", action="store_true",
        help="Skip circuit diagram output"
    )

    args = parser.parse_args()

    print("\n" + "═" * 58)
    print("  ⚛️   QUANTUM KNOT UNTIER")
    print("       Grover's Algorithm on Braid Groups")
    print("═" * 58)

    if args.all:
        run_all_instances(shots=args.shots)
        return

    # Single instance run
    secret = args.secret
    draw   = not args.no_draw

    cfg         = phase_0_setup(secret)
    circuit, *_ = phase_1_build(cfg, draw=draw)
    phase_2_3_grover_and_measure(circuit)
    summary     = phase_4_run(circuit, cfg, shots=args.shots)
    ok          = phase_5_verify(summary, cfg)

    print("\n" + "═" * 58)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()