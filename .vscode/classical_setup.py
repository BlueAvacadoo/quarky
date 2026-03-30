"""
MODULE 2: ClassicalSetup
─────────────────────────────────────────────────
Classical pre-computation that runs on the CPU before the quantum circuit.
Chooses the secret braid word, derives the visible tangle, and prepares
all data structures consumed by the oracle and result decoder.
"""

from braid_encoder import (
    compute_target_tangle,
    encode_braid_word,
    perm_to_bits,
    find_braid_for_perm,
    format_perm,
    INITIAL_STATE,
    NUM_MOVES,
    NUM_BRAID_WORDS,
)


# ── Problem Configuration ──────────────────────────────────────────────────

class ProblemConfig:
    """
    Holds all classical data for one instance of the knot-untying problem.

    Attributes:
        secret_bits  : the hidden braid word (what Grover's must find)
        secret_word  : human-readable generator sequence
        target_perm  : the visible tangle (output of the one-way function)
        target_bits  : 3-bit oracle comparison signature of target_perm
        all_solutions: all braid words that produce target_perm (≥1)
        n_qubits_search : number of search qubits (= NUM_MOVES)
        n_solutions  : number of valid solutions (usually 1 or 2)
        optimal_iters: Grover iteration count
    """

    def __init__(self, secret_bits: list[int]):
        assert len(secret_bits) == NUM_MOVES, \
            f"secret_bits must have length {NUM_MOVES}"
        assert all(b in (0, 1) for b in secret_bits), \
            "secret_bits must contain only 0s and 1s"

        self.secret_bits   = secret_bits
        self.secret_word   = encode_braid_word(secret_bits)
        self.target_perm   = compute_target_tangle(secret_bits)
        self.target_bits   = perm_to_bits(self.target_perm)
        self.all_solutions = find_braid_for_perm(self.target_perm)
        self.n_qubits_search = NUM_MOVES
        self.n_solutions   = len(self.all_solutions)
        self.optimal_iters = _compute_grover_iterations(
            N=NUM_BRAID_WORDS,
            M=self.n_solutions,
        )

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  QUANTUM KNOT UNTIER — Problem Configuration",
            "=" * 55,
            f"  Secret braid word : {''.join(self.secret_word)}",
            f"  Secret bits       : {self.secret_bits}",
            f"  Target tangle     : {format_perm(self.target_perm)}",
            f"  Oracle target bits: {self.target_bits}",
            f"  Search space      : {NUM_BRAID_WORDS} braid words",
            f"  Solutions found   : {self.n_solutions}",
            f"  All solutions     : {self.all_solutions}",
            f"  Grover iterations : {self.optimal_iters}",
            f"  Expected P(win)   : {_success_probability(NUM_BRAID_WORDS, self.n_solutions, self.optimal_iters):.1%}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"ProblemConfig(secret={self.secret_bits}, "
            f"target={self.target_perm}, iters={self.optimal_iters})"
        )


# ── Grover Math ────────────────────────────────────────────────────────────

def _compute_grover_iterations(N: int, M: int) -> int:
    """
    Compute the optimal number of Grover iterations.

    Formula: k = floor( (π/4) * sqrt(N/M) )

    Args:
        N: total search space size (8)
        M: number of marked solutions (1 or 2)

    Returns:
        optimal iteration count k
    """
    import math
    k = math.floor((math.pi / 4) * math.sqrt(N / M))
    return max(k, 1)           # at least 1 iteration


def _success_probability(N: int, M: int, k: int) -> float:
    """
    Theoretical success probability after k Grover iterations.

    P = sin²( (2k+1) * arcsin(√(M/N)) )
    """
    import math
    theta = math.asin(math.sqrt(M / N))
    return math.sin((2 * k + 1) * theta) ** 2


# ── Factory Functions ──────────────────────────────────────────────────────

def setup_default() -> ProblemConfig:
    """
    Default problem: secret braid word [1, 0, 1] → σ₂σ₁σ₂
    Target permutation: [2, 1, 0]
    """
    return ProblemConfig(secret_bits=[1, 0, 1])


def setup_custom(secret_bits: list[int]) -> ProblemConfig:
    """
    Create a problem with a user-specified secret braid word.

    Args:
        secret_bits: list of 3 bits, e.g. [0, 1, 0]
    """
    return ProblemConfig(secret_bits=secret_bits)


def setup_all_instances() -> list[ProblemConfig]:
    """
    Create all 8 problem instances (one for each possible secret).
    Useful for batch testing.
    """
    from itertools import product
    return [
        ProblemConfig(list(bits))
        for bits in product([0, 1], repeat=NUM_MOVES)
    ]


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n⚙️   ClassicalSetup — Self Test\n")

    # Test 1: default setup
    cfg = setup_default()
    print(cfg.summary())

    # Test 2: verify secret is among solutions
    assert cfg.secret_bits in cfg.all_solutions, \
        "Secret braid not in solution set!"
    print(f"\n✓ Secret bits {cfg.secret_bits} found in solutions {cfg.all_solutions}")

    # Test 3: verify target bits encode the permutation
    assert len(cfg.target_bits) == 3
    print(f"✓ Oracle target bits: {cfg.target_bits}")

    # Test 4: check grover iterations
    # NOTE: default secret [1,0,1] has M=2 solutions (both [0,1,0] and [1,0,1]
    # produce permutation [2,1,0]).  With M=2, N=8:
    #   k = floor(π/4 * sqrt(8/2)) = floor(π/4 * 2) = 1
    #   P = sin²(3 * arcsin(sqrt(2/8))) = sin²(90°) = 100%  ← perfect!
    assert cfg.optimal_iters >= 1, \
        f"Expected ≥1 iterations, got {cfg.optimal_iters}"
    print(f"✓ Grover iterations: {cfg.optimal_iters}  (M={cfg.n_solutions} solutions)")

    # Test 5: all 8 instances
    print("\n📋  All 8 problem instances:")
    print(f"{'Secret':<10} {'Tangle':<15} {'Iters':<8} {'P(win)'}")
    print("-" * 45)
    for c in setup_all_instances():
        p = _success_probability(8, c.n_solutions, c.optimal_iters)
        print(
            f"{''.join(map(str, c.secret_bits)):<10} "
            f"{str(c.target_perm):<15} "
            f"{c.optimal_iters:<8} "
            f"{p:.1%}"
        )

    print("\n✅  All ClassicalSetup tests passed!\n")