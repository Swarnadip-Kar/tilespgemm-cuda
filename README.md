# SpGEMM Benchmark Suite — RowSpGEMM vs TileSpGEMM

A complete benchmark comparing two Sparse General Matrix-Matrix Multiplication (SpGEMM) algorithms:

| Algorithm | Description |
|-----------|-------------|
| **RowSpGEMM** | Gustavson's row-row method via NVIDIA cuSPARSE (Algorithm 1 in paper) |
| **TileSpGEMM** | Tiled algorithm with 16×16 sparse tiles (3-step: tile structure → symbolic → numeric) |

Implements and benchmarks the algorithms from:
> *"TileSpGEMM: A Tiled Algorithm for Parallel Sparse General Matrix-Matrix Multiplication on GPUs"*, PPoPP '22

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA Toolkit | ≥ 11.0 | Includes `nvcc`, `cuSPARSE` |
| GPU | Compute Capability ≥ 6.1 | Pascal (GTX 1060+) or newer |
| Python | ≥ 3.8 | With `scipy`, `matplotlib`, `numpy` |
| GNU Make | any | |
| Linux | Ubuntu 18.04/20.04/22.04 tested | |

### Install Python dependencies

```bash
# Using pip (system or venv)
pip3 install scipy matplotlib numpy psutil

# Or using a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install scipy matplotlib numpy psutil
```

---

## Quick Start

```bash
# Clone / enter project directory
cd POP_project/

# Build binaries, run all benchmarks, generate all graphs:
make

# Or step by step:
make build       # Compile CUDA binaries only
make run         # Run benchmarks (produces results/results.json)
make graphs      # Generate graphs from results.json
```

That's it. After `make` completes, all graphs are in the `graphs/` folder.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make` | Full pipeline: build → benchmark → graphs |
| `make build` | Compile `build/row_spgemm` and `build/tile_spgemm` |
| `make run` | Run benchmarks on all matrices in `Dataset/` |
| `make graphs` | Generate all 5 plots from `results/results.json` |
| `make clean` | Remove `build/`, `results/`, `graphs/`, `csr_cache/` |
| `make clean-all` | Also remove extracted Dataset subdirectories |
| `make help` | Show detected paths and available targets |

### Override GPU Architecture

The Makefile auto-detects your GPU architecture via `nvidia-smi`. You can override:

```bash
make GPU_ARCH=sm_86    # Ampere (RTX 3000 series)
make GPU_ARCH=sm_89    # Ada Lovelace (RTX 4000 series)
make GPU_ARCH=sm_80    # A100
make GPU_ARCH=sm_75    # Turing (RTX 2000 series)
make GPU_ARCH=sm_70    # Volta (V100)
make GPU_ARCH=sm_61    # Pascal (GTX 1060/1080)
```

### Override CUDA Path

```bash
make CUDA_HOME=/usr/local/cuda-12.0
```

---

## Output Structure

```
POP_project/
├── build/
│   ├── row_spgemm          # RowSpGEMM binary
│   └── tile_spgemm         # TileSpGEMM binary
├── csr_cache/
│   ├── 1138_bus.csr        # Binary CSR files (auto-generated)
│   └── ...
├── results/
│   └── results.json        # All benchmark results
└── graphs/                 # All generated figures
    ├── fig1_gflops_comparison.png
    ├── fig2_peak_space_cost.png
    ├── fig3_runtime_breakdown.png
    ├── fig3b_runtime_breakdown_pct.png
    ├── fig4_conversion_vs_spgemm.png
    └── fig5_space_comparison.png
```

---

## Graphs Produced

| File | Description | Paper Figure |
|------|-------------|-------------|
| `fig1_gflops_comparison.png` | Performance (GFlops) — TileSpGEMM vs RowSpGEMM per matrix | Fig 7 |
| `fig2_peak_space_cost.png` | Peak memory usage (MB) during SpGEMM | Fig 9 |
| `fig3_runtime_breakdown.png` | TileSpGEMM Step1/Step2/Step3 time breakdown (ms) | Fig 10 |
| `fig3b_runtime_breakdown_pct.png` | Same breakdown as % of total | Fig 10 variant |
| `fig4_conversion_vs_spgemm.png` | CSR→Tiled conversion time vs SpGEMM runtime (scatter) | Fig 12 |
| `fig5_space_comparison.png` | Storage size: CSR vs Tiled format (MB) | Fig 11 |

All graphs include:
- Labelled X and Y axes with units
- Matrix names on X axis
- Value annotations on bars
- Legend identifying each algorithm

---

## Algorithm Details

### RowSpGEMM (Algorithm 1 from paper)

Uses NVIDIA cuSPARSE's `cusparseSpGEMM` which implements Gustavson's row-row method:
- Parallelises over rows of C
- Two-pass: work estimation (symbolic) + numeric compute
- Internal hash/dense-row sparse accumulator

**Three performance issues** (as identified in paper):
1. Load imbalance across rows
2. Large intermediate space allocation
3. Inefficient sparse accumulator for unpredictable row lengths

### TileSpGEMM (Algorithms 2 & 3 from paper)

Three-step approach with 16×16 sparse tiles:

**Step 1** — Tile structure of C: runs symbolic SpGEMM on compressed tile-level matrices A' and B' using cuSPARSE. Cost < 5% of total.

**Step 2** — Symbolic phase (GPU kernel): for each output tile C_ij, binary-search set intersection finds matching (A_ik, B_kj) pairs. AtomicOr on B's bitmasks builds C's bitmask, rowPtr, and nnz per tile.

**Step 3** — Numeric phase (CPU): adaptive accumulator:
- **Sparse** accumulator if nnz < 192 (75% of 256)
- **Dense** 256-element accumulator otherwise

All tile computations fit in on-chip memory (≤256 nonzeros per tile), avoiding global intermediate allocation.

### Correctness Check

When both algorithms run on the same matrix, RowSpGEMM verifies its output against TileSpGEMM's saved result (max element-wise difference reported to stderr).

---

## CSR Binary Format

Matrices are stored as binary files (`csr_cache/*.csr`):
```
int32  rows
int32  cols
int32  nnz
int32[rows+1]   rowPtr   (0-indexed)
int32[nnz]      colIdx   (0-indexed)
float64[nnz]    val
```

Conversion is handled automatically by `scripts/convert_mtx.py` using SciPy's `mmread`. Supports symmetric, general, pattern, and coordinate MTX files.

---

## Adding More Matrices

Download `.mtx.gz` or `.tar.gz` files from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) and place them in `Dataset/`. The benchmark automatically detects and processes all `*.tar.gz` archives.

```bash
# Example: download webbase-1M
wget https://sparse.tamu.edu/mat/Williams/webbase-1M.tar.gz -P Dataset/
make run graphs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvcc not found` | Add CUDA bin dir to PATH: `export PATH=/usr/local/cuda/bin:$PATH` |
| `libcusparse.so not found` at runtime | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| `CUDA error: no kernel image` | Set correct arch: `make GPU_ARCH=sm_XX` |
| Python packages missing | `pip3 install scipy matplotlib numpy psutil` |
| `nvidia-smi: command not found` | Set arch manually: `make GPU_ARCH=sm_86` |
| Out of GPU memory | Use smaller matrices; TileSpGEMM uses less memory than RowSpGEMM |

---

## Project Structure

```
POP_project/
├── Makefile                     # Main build + run orchestration
├── README.md                    # This file
├── Dataset/                     # Input .tar.gz archives
├── src/
│   ├── common.h                 # Shared types: CsrMatrix, CUDA macros, timing
│   ├── row_spgemm.cu            # RowSpGEMM via cuSPARSE
│   └── tile_spgemm.cu           # TileSpGEMM (3-step tiled algorithm)
└── scripts/
    ├── convert_mtx.py           # MTX → binary CSR converter
    ├── run_benchmarks.py        # Orchestrates extraction, conversion, benchmarking
    ├── make_all_graphs.py       # Calls all plot scripts
    ├── plot_gflops.py           # Fig 1: GFlops comparison
    ├── plot_peak_space.py       # Fig 2: Peak memory
    ├── plot_runtime_breakdown.py # Fig 3: Step breakdown
    ├── plot_conversion_time.py  # Fig 4: Conversion overhead
    └── plot_space_comparison.py # Fig 5: CSR vs Tiled storage
```
