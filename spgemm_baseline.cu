/*
 * spgemm_baseline.cu
 *
 * Baseline implementation of Sparse General Matrix-Matrix Multiplication
 * using Gustavson's row-row algorithm (Algorithm 1 from TileSpGEMM paper).
 *
 * This is the NAIVE baseline that TileSpGEMM is compared against.
 * It suffers from the three problems described in the paper:
 *   1. Load imbalance (rows have different numbers of nonzeros)
 *   2. Unknown output size (requires two passes)
 *   3. Poor sparse accumulator (dense array wastes memory for sparse rows)
 *
 * Approach:
 *   - Pass 1 (Symbolic): Count nonzeros per output row using a dense boolean array
 *   - Pass 2 (Numeric):  Fill values using a dense float accumulator
 *   - One CUDA thread per row of C
 *   - Dense scratch arrays of size N allocated per row in global memory
 *
 * Compile: nvcc -O2 -o spgemm_baseline spgemm_baseline.cu
 * Usage:   ./spgemm_baseline <input.mtx>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/* ============================================================
 *  Error checking macro
 * ============================================================ */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                   \
    } while (0)

/* ============================================================
 *  CSR Sparse Matrix Structure
 *  Standard Compressed Sparse Row format.
 *  rowPtr[i]..rowPtr[i+1]-1 gives the range of nonzeros in row i.
 * ============================================================ */
typedef struct {
    int   nrows;       /* number of rows */
    int   ncols;       /* number of columns */
    int   nnz;         /* number of nonzeros */
    int  *rowPtr;      /* row pointers, size nrows+1 */
    int  *colIdx;      /* column indices, size nnz */
    float *val;        /* values, size nnz */
} CSRMatrix;

/* ============================================================
 *  Timing utility
 * ============================================================ */
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ============================================================
 *  MTX File Reader
 *  Reads Matrix Market format (.mtx) files.
 *  Handles both 'real' (with values) and 'pattern' (binary) matrices.
 * ============================================================ */
CSRMatrix* read_mtx(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }

    char line[256];
    int is_pattern = 0, is_symmetric = 0;

    /* Read header line */
    fgets(line, sizeof(line), f);
    if (strstr(line, "pattern"))   is_pattern = 1;
    if (strstr(line, "symmetric")) is_symmetric = 1;

    /* Skip comments */
    while (fgets(line, sizeof(line), f) && line[0] == '%');

    int nrows, ncols, nnz_file;
    sscanf(line, "%d %d %d", &nrows, &ncols, &nnz_file);

    /* For symmetric matrices, each off-diagonal entry appears twice */
    int max_nnz = is_symmetric ? nnz_file * 2 : nnz_file;

    int   *row_tmp = (int*)  malloc(max_nnz * sizeof(int));
    int   *col_tmp = (int*)  malloc(max_nnz * sizeof(int));
    float *val_tmp = (float*)malloc(max_nnz * sizeof(float));

    int count = 0;
    for (int i = 0; i < nnz_file; i++) {
        int r, c; float v = 1.0f;
        if (is_pattern) fscanf(f, "%d %d", &r, &c);
        else            fscanf(f, "%d %d %f", &r, &c, &v);
        r--; c--;  /* convert 1-indexed to 0-indexed */

        row_tmp[count] = r; col_tmp[count] = c; val_tmp[count] = v;
        count++;

        if (is_symmetric && r != c) {
            row_tmp[count] = c; col_tmp[count] = r; val_tmp[count] = v;
            count++;
        }
    }
    fclose(f);

    /* Build CSR from COO */
    CSRMatrix *mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    mat->nrows = nrows; mat->ncols = ncols; mat->nnz = count;
    mat->rowPtr = (int*)  calloc(nrows + 1, sizeof(int));
    mat->colIdx = (int*)  malloc(count * sizeof(int));
    mat->val    = (float*)malloc(count * sizeof(float));

    /* Count entries per row */
    for (int i = 0; i < count; i++) mat->rowPtr[row_tmp[i] + 1]++;
    /* Prefix sum */
    for (int i = 1; i <= nrows; i++) mat->rowPtr[i] += mat->rowPtr[i-1];
    /* Fill colIdx and val (using rowPtr as temporary write positions) */
    int *pos = (int*)calloc(nrows, sizeof(int));
    for (int i = 0; i < count; i++) {
        int r = row_tmp[i];
        int p = mat->rowPtr[r] + pos[r]++;
        mat->colIdx[p] = col_tmp[i];
        mat->val[p]    = val_tmp[i];
    }
    free(pos); free(row_tmp); free(col_tmp); free(val_tmp);
    return mat;
}

void free_csr(CSRMatrix *m) {
    if (!m) return;
    free(m->rowPtr); free(m->colIdx); free(m->val); free(m);
}

/* ============================================================
 *  KERNEL: Pass 1 — Count nonzeros per row of C
 *
 *  Each thread handles one row of A.
 *  Uses a dense boolean flag array (size ncols) per thread.
 *  This is the "worst case" accumulator from the paper — it works
 *  but wastes O(N) memory and has poor data locality.
 *
 *  flag[j] = 1 means column j has a nonzero in the current row of C.
 * ============================================================ */
__global__ void count_nnz_per_row(
    const int  *A_rowPtr, const int *A_colIdx,   /* CSR of A */
    const int  *B_rowPtr, const int *B_colIdx,   /* CSR of B */
    int        *C_rowNnz,                        /* output: nnz per row */
    int        *flag_global,                     /* scratch: ncols per thread */
    int         nrows, int ncols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    /* Each thread gets its own slice of the flag array */
    int *flag = flag_global + (long long)row * ncols;

    /* Initialize flags to 0 (they were already zero-initialized on host) */
    int count = 0;

    /* Iterate over nonzeros in row `row` of A */
    for (int p = A_rowPtr[row]; p < A_rowPtr[row + 1]; p++) {
        int k = A_colIdx[p];  /* A[row, k] is nonzero */

        /* Iterate over nonzeros in row k of B */
        for (int q = B_rowPtr[k]; q < B_rowPtr[k + 1]; q++) {
            int j = B_colIdx[q];  /* B[k, j] is nonzero, so C[row,j] will be nonzero */
            if (flag[j] == 0) {
                flag[j] = 1;
                count++;
            }
        }
    }

    /* Reset flag array for reuse */
    for (int p = A_rowPtr[row]; p < A_rowPtr[row + 1]; p++) {
        int k = A_colIdx[p];
        for (int q = B_rowPtr[k]; q < B_rowPtr[k + 1]; q++) {
            flag[B_colIdx[q]] = 0;
        }
    }

    C_rowNnz[row] = count;
}

/* ============================================================
 *  KERNEL: Pass 2 — Compute values of C
 *
 *  Same thread-per-row strategy.
 *  Uses a dense float accumulator array (size ncols) per thread.
 *  After accumulation, compresses to sparse form.
 *
 *  Performance issue (from paper §2.2):
 *  - Load imbalance: rows with many nonzeros take much longer
 *  - Memory: O(N) per thread = O(N²) total for N threads
 * ============================================================ */
__global__ void compute_values(
    const int   *A_rowPtr, const int *A_colIdx, const float *A_val,
    const int   *B_rowPtr, const int *B_colIdx, const float *B_val,
    const int   *C_rowPtr,                      /* prefix-summed row pointers */
    int         *C_colIdx,                      /* output column indices */
    float       *C_val,                         /* output values */
    float       *acc_global,                    /* scratch: ncols floats per thread */
    int         *flag_global,                   /* scratch: ncols flags per thread */
    int          nrows, int ncols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    float *acc  = acc_global  + (long long)row * ncols;
    int   *flag = flag_global + (long long)row * ncols;

    /* Accumulate products into dense array */
    for (int p = A_rowPtr[row]; p < A_rowPtr[row + 1]; p++) {
        int   k   = A_colIdx[p];
        float a_v = A_val[p];
        for (int q = B_rowPtr[k]; q < B_rowPtr[k + 1]; q++) {
            int j = B_colIdx[q];
            acc[j]  += a_v * B_val[q];
            flag[j]  = 1;    /* mark as nonzero */
        }
    }

    /* Compress dense accumulator → sparse row of C */
    int write_pos = C_rowPtr[row];
    for (int p = A_rowPtr[row]; p < A_rowPtr[row + 1]; p++) {
        int k = A_colIdx[p];
        for (int q = B_rowPtr[k]; q < B_rowPtr[k + 1]; q++) {
            int j = B_colIdx[q];
            if (flag[j]) {                /* first time seeing column j */
                C_colIdx[write_pos] = j;
                C_val[write_pos]    = acc[j];
                write_pos++;
                acc[j]  = 0.0f;
                flag[j] = 0;
            }
        }
    }
}

/* ============================================================
 *  CPU prefix sum (exclusive scan)
 *  Used to convert row nnz counts → row pointers
 * ============================================================ */
static void prefix_sum(int *arr, int n) {
    int running = 0;
    for (int i = 0; i < n; i++) {
        int cur = arr[i];
        arr[i]  = running;
        running += cur;
    }
    arr[n] = running;
}

/* ============================================================
 *  Main SpGEMM function (wrapper around the two GPU passes)
 * ============================================================ */
CSRMatrix* spgemm_baseline(CSRMatrix *A, CSRMatrix *B) {
    int nrows = A->nrows, ncols = B->ncols;
    long long scratch_size = (long long)nrows * ncols;

    printf("  Matrix: %d x %d, nnz(A)=%d, nnz(B)=%d\n",
           nrows, ncols, A->nnz, B->nnz);
    printf("  Scratch memory needed: %.2f MB\n",
           scratch_size * (sizeof(int) + sizeof(float)) / 1e6);

    if (scratch_size > 2e9) {
        printf("  [SKIP] Matrix too large for dense-row baseline (>2GB scratch)\n");
        return NULL;
    }

    /* --- Upload A, B to GPU --- */
    int *d_A_rowPtr, *d_A_colIdx; float *d_A_val;
    int *d_B_rowPtr, *d_B_colIdx; float *d_B_val;

    CUDA_CHECK(cudaMalloc(&d_A_rowPtr, (nrows+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_colIdx, A->nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_val,    A->nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_rowPtr, (B->nrows+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B_colIdx, B->nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B_val,    B->nnz*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A_rowPtr, A->rowPtr, (nrows+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_colIdx, A->colIdx, A->nnz*sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_val,    A->val,    A->nnz*sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_rowPtr, B->rowPtr, (B->nrows+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_colIdx, B->colIdx, B->nnz*sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_val,    B->val,    B->nnz*sizeof(float),     cudaMemcpyHostToDevice));

    /* --- Allocate scratch (dense per-row arrays) --- */
    int   *d_flag; float *d_acc;
    CUDA_CHECK(cudaMalloc(&d_flag, scratch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_acc,  scratch_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flag, 0, scratch_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_acc,  0, scratch_size * sizeof(float)));

    /* --- Allocate row nnz counter --- */
    int *d_C_rowNnz;
    CUDA_CHECK(cudaMalloc(&d_C_rowNnz, (nrows+1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_C_rowNnz, 0, (nrows+1) * sizeof(int)));

    int block_size = 128;
    int grid_size  = (nrows + block_size - 1) / block_size;

    /* ---- PASS 1: Count nnz per row (symbolic phase) ---- */
    double t0 = get_time();
    count_nnz_per_row<<<grid_size, block_size>>>(
        d_A_rowPtr, d_A_colIdx,
        d_B_rowPtr, d_B_colIdx,
        d_C_rowNnz, d_flag,
        nrows, ncols);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_symbolic = get_time() - t0;
    printf("  Pass 1 (Symbolic / count):   %.4f seconds\n", t_symbolic);

    /* Prefix sum on CPU to build rowPtr_C */
    int *h_C_rowPtr = (int*)malloc((nrows+1) * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_C_rowPtr, d_C_rowNnz, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost));
    prefix_sum(h_C_rowPtr, nrows);
    int C_nnz = h_C_rowPtr[nrows];
    printf("  nnz(C) = %d\n", C_nnz);

    /* Allocate output C arrays */
    int   *d_C_rowPtr, *d_C_colIdx; float *d_C_val;
    CUDA_CHECK(cudaMalloc(&d_C_rowPtr, (nrows+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C_colIdx, C_nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C_val,    C_nnz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_C_rowPtr, h_C_rowPtr, (nrows+1)*sizeof(int), cudaMemcpyHostToDevice));

    /* Reset scratch for pass 2 */
    CUDA_CHECK(cudaMemset(d_flag, 0, scratch_size * sizeof(int)));

    /* ---- PASS 2: Compute values (numeric phase) ---- */
    double t1 = get_time();
    compute_values<<<grid_size, block_size>>>(
        d_A_rowPtr, d_A_colIdx, d_A_val,
        d_B_rowPtr, d_B_colIdx, d_B_val,
        d_C_rowPtr, d_C_colIdx, d_C_val,
        d_acc, d_flag,
        nrows, ncols);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_numeric = get_time() - t1;
    printf("  Pass 2 (Numeric / values):   %.4f seconds\n", t_numeric);
    printf("  Total GPU time:              %.4f seconds\n", t_symbolic + t_numeric);

    /* Copy result to host */
    CSRMatrix *C = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    C->nrows = nrows; C->ncols = ncols; C->nnz = C_nnz;
    C->rowPtr = h_C_rowPtr;
    C->colIdx = (int*)  malloc(C_nnz * sizeof(int));
    C->val    = (float*)malloc(C_nnz * sizeof(float));
    CUDA_CHECK(cudaMemcpy(C->colIdx, d_C_colIdx, C_nnz*sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C->val,    d_C_val,    C_nnz*sizeof(float), cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_A_rowPtr); cudaFree(d_A_colIdx); cudaFree(d_A_val);
    cudaFree(d_B_rowPtr); cudaFree(d_B_colIdx); cudaFree(d_B_val);
    cudaFree(d_flag); cudaFree(d_acc);
    cudaFree(d_C_rowNnz); cudaFree(d_C_rowPtr);
    cudaFree(d_C_colIdx); cudaFree(d_C_val);

    return C;
}

/* ============================================================
 *  Main entry point
 * ============================================================ */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx>\n", argv[0]);
        printf("Computes C = A * A (A squared) using baseline row-row SpGEMM\n");
        return 1;
    }

    printf("=== Baseline Row-Row SpGEMM ===\n");
    printf("Reading matrix: %s\n", argv[1]);
    double t_read = get_time();
    CSRMatrix *A = read_mtx(argv[1]);
    if (!A) return 1;
    printf("Read time: %.3f s | Matrix: %dx%d, nnz=%d\n",
           get_time() - t_read, A->nrows, A->ncols, A->nnz);

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    printf("--- Computing C = A * A ---\n");
    CSRMatrix *C = spgemm_baseline(A, A);

    if (C) {
        printf("\n[RESULT] nnz(C) = %d\n", C->nnz);
        free_csr(C);
    }
    free_csr(A);
    return 0;
}
