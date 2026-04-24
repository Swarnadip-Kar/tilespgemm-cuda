# ═══════════════════════════════════════════════════════════════════════════
#  Makefile — SpGEMM Benchmark Suite: RowSpGEMM vs TileSpGEMM
#  Works on any Linux machine with CUDA installed.
#  Auto-detects CUDA paths, architecture, and Python interpreter.
# ═══════════════════════════════════════════════════════════════════════════

# ── Auto-detect CUDA installation ──────────────────────────────────────────
# Prefer CUDA_HOME env var, else fall back to nvcc location
NVCC          := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
  $(error nvcc not found. Please install CUDA or add it to PATH.)
endif

CUDA_BIN_DIR  := $(dir $(NVCC))
CUDA_HOME     ?= $(realpath $(CUDA_BIN_DIR)/..)

# cuSPARSE headers: try CUDA_HOME/include first, then system /usr/include
CUDA_INC_DIRS := $(wildcard $(CUDA_HOME)/include) \
                 $(wildcard /usr/include) \
                 $(wildcard /usr/local/cuda/include)
CUDA_INC_DIR  := $(firstword $(CUDA_INC_DIRS))

# cuSPARSE libraries: try CUDA_HOME/lib64, lib, then system paths
CUDA_LIB_DIRS := $(wildcard $(CUDA_HOME)/lib64) \
                 $(wildcard $(CUDA_HOME)/lib) \
                 $(wildcard /usr/lib/x86_64-linux-gnu) \
                 $(wildcard /usr/local/cuda/lib64) \
                 $(wildcard /usr/lib64)
CUDA_LIB_DIR  := $(firstword $(filter-out /usr/lib64,$(CUDA_LIB_DIRS) /usr/lib/x86_64-linux-gnu))

# ── Auto-detect GPU architecture ──────────────────────────────────────────
# Queries the first GPU; falls back to sm_61 (Pascal, GTX 1000 series)
GPU_ARCH      := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                   | head -1 | tr -d '.' | sed 's/^/sm_/' 2>/dev/null)
ifeq ($(GPU_ARCH),)
  GPU_ARCH    := sm_61
  $(info [Makefile] nvidia-smi not found, using default arch $(GPU_ARCH))
else
  $(info [Makefile] Detected GPU architecture: $(GPU_ARCH))
endif

# ── Python interpreter ─────────────────────────────────────────────────────
# Use the venv python if present, else system python3
VENV_PYTHON   := $(wildcard .venv/bin/python)
PYTHON        := $(if $(VENV_PYTHON),.venv/bin/python,$(shell which python3))

ifeq ($(PYTHON),)
  $(error python3 not found. Please install Python 3.)
endif
$(info [Makefile] Using Python: $(PYTHON))

# ── Directories ────────────────────────────────────────────────────────────
SRC_DIR       := src
BUILD_DIR     := build
SCRIPTS_DIR   := scripts
RESULTS_DIR   := results
GRAPHS_DIR    := graphs
CSR_CACHE     := csr_cache

# ── Compiler flags ─────────────────────────────────────────────────────────
NVCC_FLAGS    := -O3 -arch=$(GPU_ARCH) \
                 -std=c++14 \
                 -I$(CUDA_INC_DIR) -I$(SRC_DIR) \
                 -Xcompiler -Wall,-Wextra,-O3 \
                 --extended-lambda \
                 -lineinfo

NVCC_LDFLAGS  := -L$(CUDA_LIB_DIR) \
                 -lcusparse -lcudart \
                 -Xlinker -rpath,$(CUDA_LIB_DIR)

# ── Targets ────────────────────────────────────────────────────────────────
ROW_TARGET    := $(BUILD_DIR)/row_spgemm
TILE_TARGET   := $(BUILD_DIR)/tile_spgemm
RESULTS_JSON  := $(RESULTS_DIR)/results.json

.PHONY: all _build run graphs clean clean-all help

# Default target: build, benchmark, plot
all: _build run graphs
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  All done! Graphs are in: $(GRAPHS_DIR)/"
	@echo "═══════════════════════════════════════════════════════════════"

# ── Build binaries ─────────────────────────────────────────────────────────
_build: $(ROW_TARGET) $(TILE_TARGET)
	@echo "[Build] Both binaries ready in $(BUILD_DIR)/"
build: _build

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(ROW_TARGET): $(SRC_DIR)/row_spgemm.cu $(SRC_DIR)/common.h | $(BUILD_DIR)
	@echo "[Build] Compiling RowSpGEMM ($(GPU_ARCH)) ..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LDFLAGS)
	@echo "[Build] row_spgemm -> $@"

$(TILE_TARGET): $(SRC_DIR)/tile_spgemm.cu $(SRC_DIR)/common.h | $(BUILD_DIR)
	@echo "[Build] Compiling TileSpGEMM ($(GPU_ARCH)) ..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LDFLAGS)
	@echo "[Build] tile_spgemm -> $@"

# ── Run benchmarks ─────────────────────────────────────────────────────────
run: build
	@mkdir -p $(RESULTS_DIR) $(CSR_CACHE)
	@echo ""
	@echo "[Run] Starting benchmark suite ..."
	@echo "[Run] Matrices in Dataset/ will be extracted, converted, and benchmarked."
	@echo "[Run] Results will be saved to $(RESULTS_JSON)"
	@echo ""
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmarks.py
	@echo ""
	@echo "[Run] Benchmark complete. Results: $(RESULTS_JSON)"

# ── Generate graphs ────────────────────────────────────────────────────────
graphs: $(RESULTS_JSON)
	@mkdir -p $(GRAPHS_DIR)
	@echo ""
	@echo "[Graphs] Generating all 5 figures ..."
	$(PYTHON) $(SCRIPTS_DIR)/make_all_graphs.py $(RESULTS_JSON) $(GRAPHS_DIR)
	@echo "[Graphs] Done. See $(GRAPHS_DIR)/"
	@ls -1 $(GRAPHS_DIR)/*.png 2>/dev/null | sed 's/^/  /'

# ── If results.json doesn't exist, run benchmarks first ───────────────────
$(RESULTS_JSON): build
	@$(MAKE) run

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	@echo "[Clean] Removing build artifacts ..."
	rm -rf $(BUILD_DIR) $(RESULTS_DIR) $(GRAPHS_DIR) $(CSR_CACHE)
	@echo "[Clean] Done. Dataset extractions kept in Dataset/."

clean-all: clean
	@echo "[Clean-all] Removing extracted dataset files ..."
	find Dataset/ -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} +

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "SpGEMM Benchmark Suite — Makefile targets:"
	@echo ""
	@echo "  make           — Build, run benchmarks, and generate all graphs (default)"
	@echo "  make build     — Compile CUDA binaries only"
	@echo "  make run       — Run benchmarks (requires built binaries)"
	@echo "  make graphs    — Generate plots from existing results.json"
	@echo "  make clean     — Remove build/, results/, graphs/, csr_cache/"
	@echo "  make clean-all — Also remove extracted Dataset/ subdirectories"
	@echo "  make help      — Show this message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_HOME      — Override CUDA installation path (default: auto)"
	@echo "  GPU_ARCH       — Override GPU architecture (default: auto-detected)"
	@echo "                   Examples: sm_61, sm_70, sm_75, sm_80, sm_86, sm_89"
	@echo ""
	@echo "Detected:"
	@echo "  NVCC        = $(NVCC)"
	@echo "  CUDA_HOME   = $(CUDA_HOME)"
	@echo "  CUDA_INC    = $(CUDA_INC_DIR)"
	@echo "  CUDA_LIB    = $(CUDA_LIB_DIR)"
	@echo "  GPU_ARCH    = $(GPU_ARCH)"
	@echo "  PYTHON      = $(PYTHON)"
	@echo ""
