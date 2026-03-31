# ⚛️ quarky — Quantum Knot Untier

**Grover's algorithm on braid groups.** A 48-hour hackathon quantum demo that
uses Grover's search to "untie" a mathematical knot by finding the secret braid
word that produced a given strand permutation.

---

## What It Does

Given a **target tangle** (a known permutation of 3 strands), the quantum circuit
finds the exact **braid word** (sequence of 3 crossing moves) that produced it —
without being told what the word is.

This is a quantum brute-force attack on a toy one-way function:
- **Easy:** apply braid word → get tangle
- **Hard (classically):** invert the tangle → find the braid word
- **Quantum speedup:** Grover's algorithm finds it in ~√8 ≈ 3 iterations vs 8 classical tries

---

## Module Structure

```
quarky/
├── braid_encoder.py      # Module 1: Classical data model (braid ↔ permutation)
├── classical_setup.py    # Module 2: Problem configuration & pre-computation
├── circuit_init.py       # Module 3: Quantum register construction
├── oracle.py             # Module 4: Controlled braid + phase kickback
├── diffusion.py          # Module 5: Grover diffusion operator
├── grover_loop.py        # Module 6: Full circuit assembly & iteration count
├── result_decoder.py     # Module 7: Measurement decoding & verification
├── main.py               # Module 8: Top-level execution pipeline
└── quarky_notebook.ipynb # Interactive Jupyter walkthrough + visualizations
```

---

## Quick Start

### Prerequisites
```bash
pip install qiskit>=1.4.0 qiskit-ibm-runtime>=0.36.0 matplotlib numpy
```

### Run the demo
```bash
# Default secret braid word [1, 0, 1] = σ₂σ₁σ₂
python main.py

# Custom secret
python main.py --secret 0 1 0

# More shots for higher confidence
python main.py --shots 4096

# Skip circuit diagram
python main.py --no-draw

# Run all 8 possible secrets
python main.py --all
```

### Jupyter Notebook
```bash
jupyter notebook quarky_notebook.ipynb
```

### Unit-test individual modules
```bash
python braid_encoder.py
python classical_setup.py
python circuit_init.py
python oracle.py
python diffusion.py
python grover_loop.py
python result_decoder.py
```

---

## Circuit Architecture

```
Qubit layout (7 total):
┌──────────────────────────────────────────────────────┐
│  q0  q1  q2   │   q3  q4  q5   │   q6               │
│  Search reg   │   State reg    │   Ancilla          │
│  (braid word) │   (strand pos) │   |−⟩ kickback     │
└──────────────────────────────────────────────────────┘

Circuit:
 Init → [Oracle → Diffusion] × 2 → Measure q0,q1,q2

Oracle breakdown:
  1. Controlled braid application (CSWAP gates controlled by search qubits)
  2. 3-controlled phase kickback (MCX on ancilla |−⟩)
  3. Uncompute braid (same CSWAPs in reverse → disentangle registers)
```

---

## Encoding

| Search bits | Braid word | Permutation |
|-------------|------------|-------------|
| `[0,0,0]`   | σ₁σ₁σ₁    | `[1,0,2]`   |
| `[1,0,0]`   | σ₂σ₁σ₁    | `[2,0,1]`   |
| `[0,1,0]`   | σ₁σ₂σ₁    | `[0,2,1]`   |
| `[1,1,0]`   | σ₂σ₂σ₁    | `[0,1,2]`   |
| `[0,0,1]`   | σ₁σ₁σ₂    | `[1,2,0]`   |
| **`[1,0,1]`** | **σ₂σ₁σ₂** | **`[2,1,0]`** ← default secret |
| `[0,1,1]`   | σ₁σ₂σ₂    | `[2,0,1]`   |
| `[1,1,1]`   | σ₂σ₂σ₂    | `[0,2,1]`   |

---

## Performance

| Iterations k | P(correct) | Notes              |
|--------------|------------|--------------------|
| k=0          | 12.5%      | Uniform guess      |
| k=1          | 78.1%      | —                  |
| **k=2**      | **94.5%**  | ← optimal (N=8)    |
| k=3          | 78.1%      | Overshoots         |
| k=4          | 12.5%      | Back to uniform    |

---

## Build Order (recommended for hacking)

```
1. braid_encoder.py      → test classical logic first
2. classical_setup.py    → configure your problem instance
3. circuit_init.py       → verify uniform superposition
4. oracle.py             → check phase flip on correct state
5. diffusion.py          → verify amplitude amplification
6. grover_loop.py        → assemble full circuit
7. result_decoder.py     → decode measurements
8. main.py               → run end-to-end
```

---


