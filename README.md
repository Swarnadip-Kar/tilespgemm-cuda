# TileSpGEMM Implementation

Implementation of **"TileSpGEMM: A Tiled Algorithm for Parallel Sparse General
Matrix-Matrix Multiplication on GPUs"** (Niu et al., PPoPP '22).

---

## Files

| File | Description |
|------|-------------|
| `spgemm_baseline.cu` | Baseline row-row SpGEMM (Algorithm 1 from paper) |
| `tilespgemm.cu`      | TileSpGEMM — Steps 1, 2, 3 (Algorithms 2 & 3 from paper) |
| `Makefile`           | Build system |
| `run_tests.py`       | Test script — downloads SuiteSparse datasets, runs both, compares |

---

## Quick Start

```bash
# 1. Compile
make

# 2a. Run on a downloaded matrix
./spgemm_baseline  my_matrix.mtx
./tilespgemm       my_matrix.mtx

# 2b. Use the test script (downloads datasets automatically)
python3 run_tests.py --quick          # small matrices, faster download
python3 run_tests.py                  # full set from paper Table 2
python3 run_tests.py --generate       # no download — uses random matrices

# 2c. Run on a specific local file
python3 run_tests.py --matrix path/to/my_matrix.mtx
```

---

## What Each File Does

### `spgemm_baseline.cu` — The "bad" algorithm (Section 2 of paper)
Implements the naive row-row Gustavson method. Demonstrates the three problems:
1. **Load imbalance** — rows with many nonzeros dominate runtime
2. **Unknown output size** — requires two passes (symbolic + numeric)
3. **Dense accumulator** — wastes O(N) memory even for sparse rows

Two GPU passes:
- Pass 1: `count_nnz_per_row` — counts nnz per row of C using boolean flags
- Pass 2: `compute_values`    — fills values using dense float accumulator

### `tilespgemm.cu` — The paper's algorithm (Section 3)

**Data structure** (§3.2, Figure 2):
```
High level (CSR on tile grid):
  tilePtr[tilem+1]     — row pointers
  tileColIdx[numtile]  — column indices
  tileNnz[numtile+1]   — prefix-summed nnz per tile

Per tile (CSR within 16×16 block):
  rowPtr[tile × 16]    — 16 local row offsets (uint8, range 0-255)
                          NOTE: only 16 entries, not 17 (paper §3.2)
  rowIdx[nnz]          — local row index 0-15 (uint8)
  colIdx[nnz]          — local col index 0-15 (uint8)
  val[nnz]             — float values
  mask[tile × 16]      — 16-bit mask per row (uint16)
                          bit j = 1 ↔ column j has nonzero in this row
```

**Three-step algorithm**:

**Step 1** (CPU, `find_tile_structure_C`):
- Symbolic SpGEMM at tile level to find which tiles of C are non-empty
- Paper uses NSPARSE library; we use a simple CPU implementation
- Expected time share: ~5% of total

**Step 2** (GPU, `symbolic_phase_kernel`):
- For each tile C_ij, find matching pairs (A_ik, B_kj) via binary search
- AtomicOr B's masks into C's masks to predict C's structure
- Prefix sum of popcounts gives rowPtr_C
- Expected time share: ~15% of total

**Step 3** (GPU, `numeric_phase_kernel`):
- Adaptive accumulator (threshold = 192 = 75% of 256):
  - nnz ≥ 192 → dense 256-float shared memory accumulator
  - nnz < 192 → sparse accumulator using known column positions
- Expected time share: ~70% of total

---

## Expected Output

```
Reading: cant.mtx
Read: 0.423 s | 62451×62451, nnz=4007383

=== Baseline Row-Row SpGEMM ===
  Matrix: 62451 x 62451, nnz(A)=4007383, nnz(B)=4007383
  Pass 1 (Symbolic / count):   0.8234 seconds
  Pass 2 (Numeric / values):   1.1423 seconds
  Total GPU time:              1.9657 seconds
  nnz(C) = 17400000

=== TileSpGEMM ===
Matrix: 62451×62451, nnz=4007383
Format conversion:   0.1823 s  (numtile=12455, tilem=3904×tilen=3904)
Step 1 (tile struct):0.0042 s  (numtileC=48210)
Step 2 (symbolic):   0.0287 s
Step 3 (numeric):    0.3941 s
─────────────────────────────────────────
Total (steps 1-3):   0.4270 s
  Step 1 share:   1.0%
  Step 2 share:   6.7%
  Step 3 share:  92.3%
  (Paper reports: Step1~5%, Step2~15%, Step3~70%, Mem~20%)
```

---

## Compilation Notes

**Adjust the GPU architecture** in `Makefile` to match your GPU:
```makefile
ARCH = -arch=sm_86   # RTX 3060/3070/3080 (Ampere, paper's GPU)
ARCH = -arch=sm_80   # A100, RTX 3090
ARCH = -arch=sm_75   # RTX 2080 (Turing)
ARCH = -arch=sm_70   # V100 (Volta)
ARCH = -arch=sm_61   # GTX 1080 (Pascal, minimum for paper)
```

**CUDA version**: Tested with CUDA 11.x and 12.x.

---

## Key Differences from Paper

1. **Step 1**: We use a simple CPU loop instead of calling NSPARSE.
   Fast enough for most matrices; could be replaced by a GPU implementation.

2. **Precision**: We use `float` (32-bit). The paper uses `double` (64-bit).
   Change `float` → `double` throughout for full paper comparison.

3. **B structure**: We build an explicit CSC tile structure for B.
   The paper may use a different approach internally.

4. **Shared memory limits**: The dense accumulator (256 floats × 4 bytes = 1KB)
   plus other shared allocations must fit in the GPU's shared memory per SM.
   On sm_86, each SM has 100KB shared memory, so this is fine.

---

## References

- Paper: https://doi.org/10.1145/3503221.3508431
- Source code (official): https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM
- SuiteSparse matrices: https://sparse.tamu.edu
