"""
MODULE 7: ResultDecoder
─────────────────────────────────────────────────
Post-processing for Grover measurement results:
  - Decode raw bitstrings → braid words → permutations
  - Verify solutions classically
  - Analyze shot histograms
  - Format results for display and notebooks

Qiskit returns measurement results as bitstrings (strings of '0' and '1').
Bit order: Qiskit uses LITTLE-ENDIAN convention in count keys by default —
the rightmost character corresponds to qubit 0 (cr[0]).
We reverse the string before interpreting it.
"""

from collections import Counter
from dataclasses import dataclass, field

from braid_encoder import (
    encode_braid_word,
    apply_braid_word,
    INITIAL_STATE,
    GENERATOR_NAMES,
)
from classical_setup import ProblemConfig


# ── Result Data Classes ────────────────────────────────────────────────────

@dataclass
class SingleResult:
    """Decoded result from one measurement shot."""
    raw_bits:   list[int]           # [b0, b1, b2] in qubit order
    braid_word: list[str]           # ['σ₁', 'σ₂', ...]
    perm:       list[int]           # resulting permutation
    is_correct: bool                # does perm match target?
    raw_bitstring: str = ""         # original Qiskit string e.g. "101"


@dataclass
class RunSummary:
    """Summary statistics over all shots."""
    total_shots:     int
    histogram:       dict[str, int]         # bitstring → count
    winner_bits:     list[int]              # most frequent answer
    winner_braid:    list[str]
    winner_perm:     list[int]
    winner_correct:  bool
    winner_count:    int
    winner_fraction: float
    all_decoded:     dict[str, SingleResult] = field(default_factory=dict)
    secret_bits:     list[int] = field(default_factory=list)
    target_perm:     list[int] = field(default_factory=list)


# ── Decoding Functions ─────────────────────────────────────────────────────

def decode_bitstring(bitstring: str) -> list[int]:
    """
    Convert a Qiskit result bitstring to a list of qubit bits.

    Qiskit stores results with the RIGHTMOST character = qubit 0 (LSB).
    We reverse to get [q0, q1, q2] order.

    Args:
        bitstring: e.g. "101" (Qiskit format, right=q0)

    Returns:
        [int, int, int] in qubit order, e.g. [1, 0, 1]
    """
    # Reverse: rightmost char is q0
    return [int(c) for c in reversed(bitstring)]


def decode_single(bitstring: str,
                  target_perm: list[int]) -> SingleResult:
    """
    Fully decode one measurement bitstring.

    Args:
        bitstring:   Qiskit result string (e.g. "101")
        target_perm: the target permutation for correctness check

    Returns:
        SingleResult with all decoded fields
    """
    bits       = decode_bitstring(bitstring)
    braid_word = encode_braid_word(bits)
    perm       = apply_braid_word(bits, INITIAL_STATE.copy())
    is_correct = (perm == target_perm)
    return SingleResult(
        raw_bits=bits,
        braid_word=braid_word,
        perm=perm,
        is_correct=is_correct,
        raw_bitstring=bitstring,
    )


def decode_histogram(counts: dict[str, int],
                     cfg: ProblemConfig) -> RunSummary:
    """
    Decode and analyze a full shot histogram from Qiskit.

    Args:
        counts: Qiskit result counts dict, e.g. {"101": 952, "010": 30, ...}
        cfg:    ProblemConfig with target_perm and secret_bits

    Returns:
        RunSummary with all statistics
    """
    total = sum(counts.values())
    all_decoded = {}

    for bs, count in counts.items():
        decoded = decode_single(bs, cfg.target_perm)
        all_decoded[bs] = decoded

    # Find winner (most frequent)
    winner_bs    = max(counts, key=counts.get)
    winner_dec   = all_decoded[winner_bs]
    winner_count = counts[winner_bs]

    return RunSummary(
        total_shots=total,
        histogram=dict(sorted(counts.items(),
                               key=lambda x: x[1], reverse=True)),
        winner_bits=winner_dec.raw_bits,
        winner_braid=winner_dec.braid_word,
        winner_perm=winner_dec.perm,
        winner_correct=winner_dec.is_correct,
        winner_count=winner_count,
        winner_fraction=winner_count / total,
        all_decoded=all_decoded,
        secret_bits=cfg.secret_bits,
        target_perm=cfg.target_perm,
    )


# ── Simulation Runner (Qiskit Sampler) ────────────────────────────────────

def run_simulation(circuit, cfg: ProblemConfig,
                   shots: int = 2048) -> RunSummary:
    """
    Run the circuit on Qiskit's StatevectorSampler and return decoded results.

    Args:
        circuit: fully built QuantumCircuit with measurements
        cfg:     ProblemConfig
        shots:   number of simulation shots (default 2048)

    Returns:
        RunSummary
    """
    from qiskit.primitives import StatevectorSampler

    sampler = StatevectorSampler()
    job     = sampler.run([circuit], shots=shots)
    result  = job.result()

    # Extract counts from the first (and only) PUB result
    pub_result = result[0]
    counts_int = pub_result.data.out.get_counts()   # {int_key: count}

    # Convert integer keys to zero-padded bitstrings
    counts_str = {
        format(k, "03b")[::-1]: v          # reverse for Qiskit convention
        for k, v in counts_int.items()
    }
    # Re-reverse to standard Qiskit format (right=q0)
    counts_str = {bs[::-1]: v for bs, v in counts_str.items()}

    return decode_histogram(counts_str, cfg)


# ── Display / Formatting ───────────────────────────────────────────────────

def print_summary(summary: RunSummary) -> None:
    """Print a human-readable summary of run results."""
    print("=" * 58)
    print("  GROVER RESULT SUMMARY")
    print("=" * 58)
    print(f"  Total shots     : {summary.total_shots}")
    print(f"  Target tangle   : {summary.target_perm}")
    print(f"  Secret (hidden) : {summary.secret_bits}")
    print()
    print(f"  🏆 Winner answer : {summary.winner_bits}")
    print(f"     Braid word   : {''.join(summary.winner_braid)}")
    print(f"     Permutation  : {summary.winner_perm}")
    print(f"     Correct?     : {'✅ YES' if summary.winner_correct else '❌ NO'}")
    print(f"     Seen {summary.winner_count}/{summary.total_shots} shots "
          f"({summary.winner_fraction:.1%})")
    print()
    print("  Shot histogram:")
    print(f"  {'Bits':<10} {'Braid':<12} {'Perm':<12} {'Count':>6}  {'Bar'}")
    print(f"  {'-'*55}")

    max_count = max(summary.histogram.values())
    for bs, count in summary.histogram.items():
        decoded = summary.all_decoded.get(bs)
        if decoded is None:
            continue
        bar_len = int((count / max_count) * 20)
        bar     = "█" * bar_len
        mark    = " ◄" if decoded.is_correct else ""
        print(
            f"  {str(decoded.raw_bits):<10} "
            f"{''.join(decoded.braid_word):<12} "
            f"{str(decoded.perm):<12} "
            f"{count:>6}  {bar}{mark}"
        )
    print("=" * 58)


def verify_solution(bits: list[int],
                    target_perm: list[int]) -> tuple[bool, list[int]]:
    """
    Classically verify that a braid word produces the target permutation.

    Args:
        bits:        braid word as bits
        target_perm: expected permutation

    Returns:
        (is_correct, actual_perm)
    """
    actual = apply_braid_word(bits, INITIAL_STATE.copy())
    return (actual == target_perm), actual


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from classical_setup import setup_default
    from grover_loop import build_grover_circuit

    print("\n📊  ResultDecoder — Self Test\n")

    cfg     = setup_default()
    circuit, *_ = build_grover_circuit(cfg, measure=True)

    # ── Test 1: decode_bitstring ──
    assert decode_bitstring("101") == [1, 0, 1], "Decode failed"
    assert decode_bitstring("010") == [0, 1, 0], "Decode failed"
    assert decode_bitstring("000") == [0, 0, 0], "Decode failed"
    print("✓ decode_bitstring works correctly")

    # ── Test 2: decode_single ──
    r = decode_single("101", cfg.target_perm)
    print(f"✓ decode_single('101'): bits={r.raw_bits}, "
          f"braid={''.join(r.braid_word)}, perm={r.perm}, correct={r.is_correct}")

    # ── Test 3: verify_solution ──
    ok, perm = verify_solution(cfg.secret_bits, cfg.target_perm)
    assert ok, f"Verification failed: {perm} ≠ {cfg.target_perm}"
    print(f"✓ verify_solution({cfg.secret_bits}) → perm={perm}, correct={ok}")

    # ── Test 4: run simulation ──
    print("\n── Running simulation (2048 shots) ──")
    summary = run_simulation(circuit, cfg, shots=2048)
    print_summary(summary)

    # ── Assertions ──
    assert summary.winner_correct, \
        f"Winner {summary.winner_bits} is WRONG! expected {cfg.secret_bits}"
    assert summary.winner_fraction > 0.80, \
        f"Winner fraction too low: {summary.winner_fraction:.1%}"

    print(f"\n✅  Winner is correct!  P(win) = {summary.winner_fraction:.1%}\n")