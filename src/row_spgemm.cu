/*
 * row_spgemm.cu  —  Custom Row-Row SpGEMM (GPU Hash Accumulator)
 * ─────────────────────────────────────────────────────────────────────────────
 * No cuSPARSE — fully custom CUDA implementation of SpGEMM (C = A²).
 *
 * Algorithm: Hash-Based Row SpGEMM with open-addressing global hash tables
 * Phase 1: upper_nnz_kernel  — per-row upper bound → size hash tables
 * Phase 2: spgemm_hash_kernel — 1 warp/row, atomicCAS+atomicAdd into hash
 * Phase 3: Thrust inclusive_scan — rowPtrC (prefix sum)
 * Phase 4: collect_kernel — hash tables → sorted CSR output
 *
 * Hash table design:
 * - int   hash_keys[]  initialized to 0xFFFFFFFF = -1  (HASH_EMPTY)
 * - double hash_vals[] initialized to 0x00             (0.0)
 * - Open addressing with linear probing
 * - atomicCAS claims key slot; atomicAdd accumulates value (sm_89 supports
 * native double atomicAdd)
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static inline double wtime(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define WARP_SIZE        32
#define WARPS_PER_BLK    8
#define HASH_EMPTY       (-1)
#define MIN_HASH_CAP     32
// #define MAX_HASH_CAP     (1 << 17)   /* 128K entries/row max */ // BUG FIX: Removed MAX_HASH_CAP to allow larger hash tables when needed (with memory check)

typedef struct {
    int rows, cols, nnz;
    int    *rowPtr, *colIdx; double *val;
    int    *d_rowPtr, *d_colIdx; double *d_val;
} CsrMatrix;

static int csr_load_binary(const char *path, CsrMatrix *A) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    if (fread(&A->rows, sizeof(int), 1, fp) != 1 ||
        fread(&A->cols, sizeof(int), 1, fp) != 1 ||
        fread(&A->nnz,  sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    A->rowPtr = (int*)    malloc((size_t)(A->rows+1)*sizeof(int));
    A->colIdx = (int*)    malloc((size_t)A->nnz*sizeof(int));
    A->val    = (double*) malloc((size_t)A->nnz*sizeof(double));
    if (!A->rowPtr || !A->colIdx || !A->val) { fclose(fp); return -1; }
    fread(A->rowPtr, sizeof(int),    (size_t)(A->rows+1), fp);
    fread(A->colIdx, sizeof(int),    (size_t)A->nnz,      fp);
    fread(A->val,    sizeof(double), (size_t)A->nnz,      fp);
    fclose(fp);
    A->d_rowPtr = NULL; A->d_colIdx = NULL; A->d_val = NULL;
    return 0;
}

static void csr_upload(CsrMatrix *A) {
    CUDA_CHECK(cudaMalloc(&A->d_rowPtr, (size_t)(A->rows+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A->d_colIdx, (size_t)A->nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A->d_val,    (size_t)A->nnz*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A->d_rowPtr, A->rowPtr, (size_t)(A->rows+1)*sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A->d_colIdx, A->colIdx, (size_t)A->nnz*sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A->d_val,    A->val,    (size_t)A->nnz*sizeof(double),    cudaMemcpyHostToDevice));
}

static inline int next_pow2(int x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 1 — upper-bound nnz per row of C
 * upper_nnz[i] = Σ_{k: A[i,k]≠0} nnz(B row k)
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__
void upper_nnz_kernel(
    const int *A_rowPtr, const int *A_colIdx,
    const int *B_rowPtr, int *upper_nnz, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    int s = 0;
    for (int jp = A_rowPtr[i]; jp < A_rowPtr[i+1]; jp++)
        s += B_rowPtr[A_colIdx[jp]+1] - B_rowPtr[A_colIdx[jp]];
    upper_nnz[i] = s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 2 — Hash-based SpGEMM  (1 warp per row of C)
 *
 * Each lane takes every WARP_SIZE-th nonzero of A[i,*].
 * For each A[i,k] iterates over B[k,*]; inserts (j, A*B) into hash table.
 * atomicCAS(&key, HASH_EMPTY, j)  → claim slot
 * atomicAdd(&val, prod)           → accumulate (atomic double, sm≥6.0)
 * After fill: all lanes count occupied slots, warp-reduce → nnzC[i].
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__
void spgemm_hash_kernel(
    const int *A_rowPtr, const int *A_colIdx, const double *A_val,
    const int *B_rowPtr, const int *B_colIdx, const double *B_val,
    int *hash_keys, double *hash_vals,
    const int *hash_off, const int *hash_cap,
    int *nnzC, int rows)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int wid  = tid  / WARP_SIZE;
    int lane = tid  % WARP_SIZE;
    if (wid >= rows) return;

    int hoff  = hash_off[wid];
    int hcap  = hash_cap[wid];
    int hmask = hcap - 1;

    for (int jp = A_rowPtr[wid] + lane; jp < A_rowPtr[wid+1]; jp += WARP_SIZE) {
        int    k    = A_colIdx[jp];
        double aval = A_val[jp];
        for (int kp = B_rowPtr[k]; kp < B_rowPtr[k+1]; kp++) {
            int    j    = B_colIdx[kp];
            double prod = aval * B_val[kp];
            int slot = j & hmask;
            while (true) {
                int old = atomicCAS(&hash_keys[hoff + slot], HASH_EMPTY, j);
                if (old == HASH_EMPTY || old == j) {
                    atomicAdd(&hash_vals[hoff + slot], prod);
                    break;
                }
                slot = (slot + 1) & hmask;
            }
        }
    }
    __syncwarp();

    /* Warp-parallel count of occupied slots */
    int cnt = 0;
    for (int s = lane; s < hcap; s += WARP_SIZE)
        cnt += (hash_keys[hoff + s] != HASH_EMPTY) ? 1 : 0;
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        cnt += __shfl_down_sync(0xFFFFFFFFu, cnt, off);
    if (lane == 0) nnzC[wid] = cnt;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 4 — Collect hash entries → CSR (one thread/row, insertion sort)
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__
void collect_kernel(
    const int *hash_keys, const double *hash_vals,
    const int *hash_off,  const int *hash_cap,
    const int *rowPtrC, int *colIdxC, double *valC, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    int hoff = hash_off[i], hcap = hash_cap[i];
    int out  = rowPtrC[i], cnt = 0;

    for (int s = 0; s < hcap; s++) {
        if (hash_keys[hoff + s] != HASH_EMPTY) {
            colIdxC[out + cnt] = hash_keys[hoff + s];
            valC[out + cnt]    = hash_vals[hoff + s];
            cnt++;
        }
    }
    /* Insertion sort by column (cnt is small for sparse rows) */
    for (int a = 1; a < cnt; a++) {
        int    kc = colIdxC[out + a];
        double vv = valC[out + a];
        int b = a - 1;
        while (b >= 0 && colIdxC[out + b] > kc) {
            colIdxC[out + b + 1] = colIdxC[out + b];
            valC[out + b + 1]    = valC[out + b];
            b--;
        }
        colIdxC[out + b + 1] = kc;
        valC[out + b + 1]    = vv;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: row_spgemm <csr_binary> <matrix_name>\n");
        return 1;
    }
    const char *csr_path = argv[1];
    const char *mat_name = argv[2];

    fprintf(stderr, "[RowSpGEMM] Loading '%s' from %s ...\n", mat_name, csr_path);
    CsrMatrix A;
    if (csr_load_binary(csr_path, &A) != 0) return 1;
    fprintf(stderr, "[RowSpGEMM] Matrix: %d x %d, nnz=%d\n", A.rows, A.cols, A.nnz);
    csr_upload(&A);

    CUDA_CHECK(cudaDeviceSynchronize());
    
    // =========================================================================
    // START END-TO-END ALGORITHM TIMER
    // =========================================================================
    double t_start_algo = wtime();

    /* ── Phase 1: upper-bound nnz per row ─────────────────────────────── */
    int *d_upper_nnz;
    CUDA_CHECK(cudaMalloc(&d_upper_nnz, (size_t)A.rows * sizeof(int)));
    {
        int blk = 256, grd = (A.rows + blk - 1) / blk;
        upper_nnz_kernel<<<grd, blk>>>(
            A.d_rowPtr, A.d_colIdx, A.d_rowPtr, d_upper_nnz, A.rows);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }
    int *h_upper = (int*)malloc((size_t)A.rows * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_upper, d_upper_nnz,
                          (size_t)A.rows * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_upper_nnz);

    /* Compute hash_cap[i] and hash_off[i] for all rows */
    int *h_hcap = (int*)malloc((size_t)A.rows * sizeof(int));
    int *h_hoff = (int*)malloc((size_t)(A.rows + 1) * sizeof(int));
    if (!h_hcap || !h_hoff) { fprintf(stderr, "[RowSpGEMM] OOM\n"); return 1; }

    size_t total_hash_entries = 0;
    h_hoff[0] = 0;
    for (int i = 0; i < A.rows; i++) {
        int cap = next_pow2(h_upper[i] * 2);
        if (cap < MIN_HASH_CAP) cap = MIN_HASH_CAP;
        
        /* BUG FIX: Removed 'if (cap > MAX_HASH_CAP)' to prevent infinite probing loops */
        
        h_hcap[i] = cap;
        total_hash_entries += (size_t)cap;
        h_hoff[i + 1] = h_hoff[i] + cap;
    }
    free(h_upper);

    size_t hash_bytes_total = total_hash_entries * (sizeof(int) + sizeof(double));
    fprintf(stderr, "[RowSpGEMM] Hash tables: %zu entries, %.2f GB\n",
            total_hash_entries, hash_bytes_total / 1.0e9);

    #define HASH_MEM_BUDGET ((size_t)4 * 1024 * 1024 * 1024)   /* 4 GB */

    /* BUG FIX: Straight skip if memory exceeds budget (No Batching) */
    if (hash_bytes_total > HASH_MEM_BUDGET) {
        fprintf(stderr, "[RowSpGEMM] Memory (%.2f GB) exceeds 4GB Budget. Skipping.\n", hash_bytes_total / 1.0e9);
        printf("{\"algo\":\"RowSpGEMM\",\"matrix\":\"%s\",\"skipped\":true}\n", mat_name);
        
        /* Cleanup and exit gracefully */
        free(h_hcap); free(h_hoff);
        cudaFree(A.d_rowPtr); cudaFree(A.d_colIdx); cudaFree(A.d_val);
        free(A.rowPtr); free(A.colIdx); free(A.val);
        return 0;
    }

    /* ── Alloc + init global hash tables ──────────────────────────────── */
    int *d_hcap, *d_hoff;
    CUDA_CHECK(cudaMalloc(&d_hcap, (size_t)A.rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hoff, (size_t)(A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_hcap, h_hcap, (size_t)A.rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hoff, h_hoff, (size_t)(A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    int *d_hkeys; double *d_hvals;
    CUDA_CHECK(cudaMalloc(&d_hkeys, total_hash_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hvals, total_hash_entries * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hkeys, 0xFF, total_hash_entries * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hvals, 0x00, total_hash_entries * sizeof(double)));

    int *d_nnzC;
    CUDA_CHECK(cudaMalloc(&d_nnzC, (size_t)A.rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nnzC, 0, (size_t)A.rows * sizeof(int)));

    /* ── Phase 2: Hash kernel ─────────────────────────────────────────── */
    {
        int threads = WARPS_PER_BLK * WARP_SIZE;
        int blocks  = (A.rows + WARPS_PER_BLK - 1) / WARPS_PER_BLK;
        spgemm_hash_kernel<<<blocks, threads>>>(
            A.d_rowPtr, A.d_colIdx, A.d_val,
            A.d_rowPtr, A.d_colIdx, A.d_val,
            d_hkeys, d_hvals, d_hoff, d_hcap,
            d_nnzC, A.rows);
        CUDA_CHECK(cudaGetLastError());
    }

    /* ── Phase 3: Build rowPtrC ───────────────────────────────────────── */
    int *d_rowPtrC;
    CUDA_CHECK(cudaMalloc(&d_rowPtrC, (size_t)(A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_rowPtrC, 0, sizeof(int))); /* Ensure first element is 0 */
    thrust::inclusive_scan(
        thrust::device_ptr<int>(d_nnzC),
        thrust::device_ptr<int>(d_nnzC + A.rows),
        thrust::device_ptr<int>(d_rowPtrC + 1));
        
    int nnzC = 0;
    CUDA_CHECK(cudaMemcpy(&nnzC, d_rowPtrC + A.rows, sizeof(int), cudaMemcpyDeviceToHost));

    /* ── Phase 4: Collect to CSR ──────────────────────────────────────── */
    size_t safe_nnzC = (nnzC > 0) ? (size_t)nnzC : 1;
    int    *d_ciC;
    double *d_vC;
    CUDA_CHECK(cudaMalloc(&d_ciC, safe_nnzC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vC,  safe_nnzC * sizeof(double)));
    {
        int blk = 256, grd = (A.rows + blk - 1) / blk;
        collect_kernel<<<grd, blk>>>(
            d_hkeys, d_hvals, d_hoff, d_hcap,
            d_rowPtrC, d_ciC, d_vC, A.rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all GPU work is totally finished
    
    // =========================================================================
    // END END-TO-END ALGORITHM TIMER
    // =========================================================================
    double time_ms = (wtime() - t_start_algo) * 1e3;

    /* ── Timing + stats ──────────────────────────────────────────────── */
    long long flops = 0;
    for (int i = 0; i < A.rows; i++)
        for (int jp = A.rowPtr[i]; jp < A.rowPtr[i+1]; jp++)
            flops += 2LL * (A.rowPtr[A.colIdx[jp]+1] - A.rowPtr[A.colIdx[jp]]);
    
    double gflops = (flops / 1e9) / (time_ms / 1e3);

    size_t mem_bytes =
        (size_t)(A.rows + 1) * sizeof(int) * 2 +
        (size_t)A.nnz * (sizeof(int) + sizeof(double)) * 2 +
        hash_bytes_total + 
        (size_t)nnzC * (sizeof(int) + sizeof(double)) +
        (size_t)(A.rows + 1) * sizeof(int);

    fprintf(stderr, "[RowSpGEMM] nnzC = %d\n", nnzC);
    fprintf(stderr,
            "[RowSpGEMM] total algorithm time = %.3f ms  %.4f GFlops\n",
            time_ms, gflops);

    printf("{\"algo\":\"RowSpGEMM\",\"matrix\":\"%s\","
           "\"time_ms\":%.4f,\"gflops\":%.6f,"
           "\"mem_bytes\":%zu,\"nnz_C\":%d,\"flops\":%lld}\n",
           mat_name, time_ms, gflops, mem_bytes, nnzC, flops);

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    cudaFree(A.d_rowPtr); cudaFree(A.d_colIdx); cudaFree(A.d_val);
    cudaFree(d_hkeys); cudaFree(d_hvals); cudaFree(d_hcap); cudaFree(d_hoff);
    cudaFree(d_nnzC); cudaFree(d_rowPtrC); cudaFree(d_ciC); cudaFree(d_vC);
    free(A.rowPtr); free(A.colIdx); free(A.val);
    free(h_hcap); free(h_hoff);
    
    return 0;
}