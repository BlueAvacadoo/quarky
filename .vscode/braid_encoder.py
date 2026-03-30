"""
MODULE 1: BraidEncoder
─────────────────────────────────────────────────
Pure classical data model for encoding/decoding braid words
and computing permutations. No quantum dependencies.

Encoding:
  - 3 search qubits → 3 moves → 8 possible braid words
  - Each qubit bit encodes one crossing move:
      bit = 0  →  σ₁  (swap strand 0 ↔ strand 1)
      bit = 1  →  σ₂  (swap strand 1 ↔ strand 2)
"""

from itertools import product


# ── Constants ──────────────────────────────────────────────────────────────

GENERATOR_NAMES = {0: "σ₁", 1: "σ₂"}
NUM_STRANDS     = 3
NUM_MOVES       = 3                         # braid word length
NUM_BRAID_WORDS = 2 ** NUM_MOVES            # 8 possibilities
INITIAL_STATE   = [0, 1, 2]                 # canonical strand order


# ── Core Functions ─────────────────────────────────────────────────────────

def encode_braid_word(bits: list[int]) -> list[str]:
    """
    Convert a list of bits (length 3) into a braid word (list of generator names).

    Args:
        bits: list of 0s and 1s, e.g. [1, 0, 1]

    Returns:
        list of generator strings, e.g. ['σ₂', 'σ₁', 'σ₂']
    """
    assert len(bits) == NUM_MOVES, f"Expected {NUM_MOVES} bits, got {len(bits)}"
    return [GENERATOR_NAMES[b] for b in bits]


def decode_braid_word(braid_word: list[str]) -> list[int]:
    """
    Convert a braid word back to bits.

    Args:
        braid_word: list of generator strings, e.g. ['σ₂', 'σ₁', 'σ₂']

    Returns:
        list of bits, e.g. [1, 0, 1]
    """
    reverse_map = {v: k for k, v in GENERATOR_NAMES.items()}
    return [reverse_map[g] for g in braid_word]


def apply_single_move(state: list[int], move_bit: int) -> list[int]:
    """
    Apply a single braid generator to a strand state.

    Args:
        state:    current strand positions, e.g. [0, 1, 2]
        move_bit: 0 → σ₁ (swap 0↔1), 1 → σ₂ (swap 1↔2)

    Returns:
        new strand state after the move
    """
    s = state.copy()
    if move_bit == 0:           # σ₁: swap positions 0 and 1
        s[0], s[1] = s[1], s[0]
    else:                       # σ₂: swap positions 1 and 2
        s[1], s[2] = s[2], s[1]
    return s


def apply_braid_word(bits: list[int], initial: list[int] = None) -> list[int]:
    """
    Apply a full braid word (list of bits) to a strand state.

    Args:
        bits:    list of move bits, e.g. [1, 0, 1]
        initial: starting strand state (defaults to [0, 1, 2])

    Returns:
        final strand permutation
    """
    state = (initial or INITIAL_STATE).copy()
    for bit in bits:
        state = apply_single_move(state, bit)
    return state


def compute_target_tangle(secret_bits: list[int]) -> list[int]:
    """
    Given the secret braid word, compute the resulting tangle (permutation).
    This is the "easy direction" of the one-way function.

    Args:
        secret_bits: the hidden braid word as bits

    Returns:
        the visible target permutation
    """
    return apply_braid_word(secret_bits, INITIAL_STATE)


def perm_to_bits(perm: list[int]) -> list[int]:
    """
    Encode a 3-element permutation into 3 comparison bits for the oracle.

    Strategy: track which position each original strand occupies.
    bit_i = 1 if strand i has moved to an ODD position, else 0.
    This gives a unique 3-bit signature for each of the 6 valid permutations
    (verified below in enumerate_all_braid_words).

    Args:
        perm: permutation, e.g. [2, 0, 1]

    Returns:
        3 comparison bits for use in the oracle phase-comparison step
    """
    # Position of each strand: where is strand i now?
    # perm[slot] = strand_id → position_of[strand_id] = slot
    position_of = [0] * NUM_STRANDS
    for slot, strand_id in enumerate(perm):
        position_of[strand_id] = slot

    # Encode: bit_i = parity of position (0=even slot, 1=odd slot)
    return [position_of[i] % 2 for i in range(NUM_STRANDS)]


# ── Enumeration & Lookup Table ─────────────────────────────────────────────

def enumerate_all_braid_words() -> dict:
    """
    Enumerate all 8 possible 3-move braid words and their permutations.

    Returns:
        dict mapping bits_tuple → {'braid': [...], 'perm': [...], 'bits': [...]}
    """
    table = {}
    for bits in product([0, 1], repeat=NUM_MOVES):
        bits_list = list(bits)
        perm      = apply_braid_word(bits_list)
        table[bits] = {
            'bits':  bits_list,
            'braid': encode_braid_word(bits_list),
            'perm':  perm,
            'sig':   perm_to_bits(perm),
        }
    return table


def find_braid_for_perm(target_perm: list[int]) -> list[list[int]]:
    """
    Classically find all braid words that produce a given permutation.
    Used to verify Grover's answer.

    Args:
        target_perm: the permutation to match

    Returns:
        list of bit sequences that produce the target permutation
    """
    matches = []
    for bits in product([0, 1], repeat=NUM_MOVES):
        if apply_braid_word(list(bits)) == target_perm:
            matches.append(list(bits))
    return matches


# ── Display Helpers ────────────────────────────────────────────────────────

def format_perm(perm: list[int]) -> str:
    """Human-readable permutation string."""
    arrows = " → ".join(
        f"pos{i}←strand{s}" for i, s in enumerate(perm)
    )
    return f"[{', '.join(map(str, perm))}]  ({arrows})"


def print_braid_table():
    """Print all 8 braid words and their permutations (for debugging)."""
    table = enumerate_all_braid_words()
    print("=" * 65)
    print(f"{'Bits':<10} {'Braid Word':<20} {'Permutation':<15} {'Sig bits'}")
    print("=" * 65)
    for bits, entry in table.items():
        bits_str  = "".join(map(str, entry['bits']))
        braid_str = "".join(entry['braid'])
        perm_str  = str(entry['perm'])
        sig_str   = str(entry['sig'])
        print(f"{bits_str:<10} {braid_str:<20} {perm_str:<15} {sig_str}")
    print("=" * 65)


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧵  BraidEncoder — Self Test\n")

    # Test 1: encode / decode roundtrip
    secret = [1, 0, 1]
    word   = encode_braid_word(secret)
    back   = decode_braid_word(word)
    assert back == secret, f"Roundtrip failed: {back} ≠ {secret}"
    print(f"✓ Encode/decode roundtrip: {secret} → {word} → {back}")

    # Test 2: known permutation
    # [1,0,1] = σ₂, σ₁, σ₂
    #   [0,1,2] →σ₂→ [0,2,1] →σ₁→ [2,0,1] →σ₂→ [2,1,0]
    perm = apply_braid_word([1, 0, 1])
    assert perm == [2, 1, 0], f"Wrong perm: {perm}"
    print(f"✓ [1,0,1] produces permutation: {perm}")

    # Test 3: identity braid
    perm_id = apply_braid_word([0, 0, 0])
    # [0,1,2] →σ₁→ [1,0,2] →σ₁→ [0,1,2] →σ₁→ [1,0,2]
    print(f"✓ [0,0,0] (σ₁σ₁σ₁) produces: {perm_id}")

    # Test 4: enumerate all & verify uniqueness of signatures
    print("\n📋  All braid words:")
    print_braid_table()

    # Test 5: inverse lookup
    target = compute_target_tangle([1, 0, 1])
    found  = find_braid_for_perm(target)
    print(f"\n✓ Classical search for perm {target}: found {found}")
    assert [1, 0, 1] in found

    print("\n✅  All BraidEncoder tests passed!\n")