# Makefile for TileSpGEMM and Baseline SpGEMM
# Requires: CUDA Toolkit (nvcc), tested on CUDA 11.x and 12.x
#
# Usage:
#   make              — build both programs
#   make clean        — remove compiled binaries
#   make run MTX=path — compile and run both on a given .mtx file

NVCC       = nvcc
NVCC_FLAGS = -O2 -std=c++14
# Adjust arch for your GPU:
#   sm_60 = Pascal (GTX 1080)
#   sm_70 = Volta  (V100)
#   sm_75 = Turing (RTX 2080)
#   sm_80 = Ampere (A100, RTX 3090) — paper uses this
#   sm_86 = Ampere (RTX 3060/3070/3080)
ARCH       = -arch=sm_70 -gencode arch=compute_70,code=sm_70 \
                          -gencode arch=compute_80,code=sm_80 \
                          -gencode arch=compute_86,code=sm_86

TARGETS = spgemm_baseline tilespgemm

all: $(TARGETS)
	@echo ""
	@echo "Build complete. Run with:"
	@echo "  ./spgemm_baseline  <matrix.mtx>"
	@echo "  ./tilespgemm       <matrix.mtx>"

spgemm_baseline: spgemm_baseline.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<
	@echo "Built: spgemm_baseline"

tilespgemm: tilespgemm.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<
	@echo "Built: tilespgemm"

# Quick run target: make run MTX=datasets/cant.mtx
run: all
	@echo "======= Baseline ======="
	./spgemm_baseline $(MTX)
	@echo ""
	@echo "======= TileSpGEMM ======"
	./tilespgemm $(MTX)

clean:
	rm -f $(TARGETS)

.PHONY: all clean run
