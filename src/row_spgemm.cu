/*
 * row_spgemm.cu  —  Custom Row-Row SpGEMM (GPU Hash Accumulator)
 * ─────────────────────────────────────────────────────────────────────────────
 * No cuSPARSE — fully custom CUDA implementation of SpGEMM (C = A²).
 *
 * Algorithm: Hash-Based Row SpGEMM with open-addressing global hash tables
 *   Phase 1: upper_nnz_kernel  — per-row upper bound → size hash tables
 *   Phase 2: spgemm_hash_kernel — 1 warp/row, atomicCAS+atomicAdd into hash
 *   Phase 3: Thrust inclusive_scan — rowPtrC (prefix sum)
 *   Phase 4: collect_kernel — hash tables → sorted CSR output
 *
 * Hash table design:
 *   - int   hash_keys[]  initialized to 0xFFFFFFFF = -1  (HASH_EMPTY)
 *   - double hash_vals[] initialized to 0x00             (0.0)
 *   - Open addressing with linear probing
 *   - atomicCAS claims key slot; atomicAdd accumulates value (sm_89 supports
 *     native double atomicAdd)
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
#define MAX_HASH_CAP     (1 << 17)   /* 128K entries/row max */

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
 *  PHASE 1 — upper-bound nnz per row of C
 *  upper_nnz[i] = Σ_{k: A[i,k]≠0} nnz(B row k)
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
 *  PHASE 2 — Hash-based SpGEMM  (1 warp per row of C)
 *
 *  Each lane takes every WARP_SIZE-th nonzero of A[i,*].
 *  For each A[i,k] iterates over B[k,*]; inserts (j, A*B) into hash table.
 *    atomicCAS(&key, HASH_EMPTY, j)  → claim slot
 *    atomicAdd(&val, prod)           → accumulate (atomic double, sm≥6.0)
 *  After fill: all lanes count occupied slots, warp-reduce → nnzC[i].
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
 *  PHASE 4 — Collect hash entries → CSR (one thread/row, insertion sort)
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
 *  MAIN
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

    /* Compute hash_cap[i] for all rows.
       Use size_t accumulation to detect overflow / large matrices.        */
    int    *h_hcap = (int*)malloc((size_t)A.rows * sizeof(int));
    if (!h_hcap) { fprintf(stderr, "[RowSpGEMM] OOM\n"); return 1; }

    /* Running total in size_t — avoids int overflow (94k rows × 128K cap > INT_MAX) */
    size_t total_hash_entries = 0;
    for (int i = 0; i < A.rows; i++) {
        int cap = next_pow2(h_upper[i] * 2);
        if (cap < MIN_HASH_CAP) cap = MIN_HASH_CAP;
        if (cap > MAX_HASH_CAP) cap = MAX_HASH_CAP;
        h_hcap[i] = cap;
        total_hash_entries += (size_t)cap;
    }
    free(h_upper);

    size_t hash_bytes_total = total_hash_entries * (sizeof(int) + sizeof(double));
    fprintf(stderr, "[RowSpGEMM] Hash tables: %zu entries, %.2f GB\n",
            total_hash_entries, hash_bytes_total / 1.0e9);

    /* Memory budget for hash tables per batch (leave headroom for CSR arrays) */
#define HASH_MEM_BUDGET ((size_t)4 * 1024 * 1024 * 1024)   /* 4 GB */

    /* ─── Host output accumulators (filled batch by batch) ────────────── */
    int    *h_nnzC    = (int*)calloc((size_t)A.rows, sizeof(int));
    /* Growing host buffer for C nonzeros */
    size_t  h_C_cap   = (size_t)A.nnz * 4 + 1024;   /* rough initial capacity */
    int    *h_ciC_all = (int*)   malloc(h_C_cap * sizeof(int));
    double *h_vC_all  = (double*)malloc(h_C_cap * sizeof(double));
    size_t  h_C_used  = 0;
    if (!h_nnzC || !h_ciC_all || !h_vC_all) {
        fprintf(stderr, "[RowSpGEMM] OOM (host output buffers)\n"); return 1; }

    /* CUDA events for timing the entire compute (hash + collect, all batches) */
    cudaEvent_t ev0, ev1, ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2)); CUDA_CHECK(cudaEventCreate(&ev3));

    double t_hash_total_ms = 0.0, t_collect_total_ms = 0.0;

    /* ════════════════════════════════════════════════════════════════════
     *  Batched processing — process a range of rows at a time so that
     *  hash table memory stays within HASH_MEM_BUDGET.
     * ════════════════════════════════════════════════════════════════════ */
    int i0 = 0;
    while (i0 < A.rows) {
        /* Find largest i1 such that sum(hcap[i0..i1)) × 12 ≤ budget */
        size_t batch_hash_entries = 0;
        int i1 = i0;
        while (i1 < A.rows) {
            size_t new_entries = batch_hash_entries + (size_t)h_hcap[i1];
            if (new_entries * (sizeof(int) + sizeof(double)) > HASH_MEM_BUDGET
                    && i1 > i0) break;
            batch_hash_entries = new_entries;
            i1++;
        }

        int batch_rows = i1 - i0;
        fprintf(stderr, "[RowSpGEMM] Batch rows [%d,%d): %d rows, %.2f GB hash\n",
                i0, i1, batch_rows,
                batch_hash_entries * (sizeof(int) + sizeof(double)) / 1.0e9);

        /* Build batch-local int hoff[batch_rows+1] — fits in int since
           batch_hash_entries ≤ HASH_MEM_BUDGET/(12) ≤ 4G/12 ≈ 357M < INT_MAX  */
        int *h_hoff_b = (int*)malloc((size_t)(batch_rows + 1) * sizeof(int));
        if (!h_hoff_b) { fprintf(stderr, "[RowSpGEMM] OOM hoff\n"); return 1; }
        h_hoff_b[0] = 0;
        for (int i = 0; i < batch_rows; i++)
            h_hoff_b[i + 1] = h_hoff_b[i] + h_hcap[i0 + i];

        /* Upload batch hcap, hoff to device */
        int *d_hcap_b, *d_hoff_b;
        CUDA_CHECK(cudaMalloc(&d_hcap_b, (size_t)batch_rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_hoff_b, (size_t)(batch_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_hcap_b, h_hcap + i0,
                              (size_t)batch_rows * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_hoff_b, h_hoff_b,
                              (size_t)(batch_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        free(h_hoff_b);

        /* Alloc + init batch hash tables */
        int    *d_hkeys_b;
        double *d_hvals_b;
        CUDA_CHECK(cudaMalloc(&d_hkeys_b, batch_hash_entries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_hvals_b, batch_hash_entries * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hkeys_b, 0xFF, batch_hash_entries * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_hvals_b, 0x00, batch_hash_entries * sizeof(double)));

        /* Batch nnzC counter */
        int *d_nnzC_b;
        CUDA_CHECK(cudaMalloc(&d_nnzC_b, (size_t)batch_rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_nnzC_b, 0, (size_t)batch_rows * sizeof(int)));

        /* ── Hash kernel (B = A, pass shifted rowPtr pointer) ── */
        {
            int threads = WARPS_PER_BLK * WARP_SIZE;
            int blocks  = (batch_rows + WARPS_PER_BLK - 1) / WARPS_PER_BLK;
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(ev0));
            spgemm_hash_kernel<<<blocks, threads>>>(
                A.d_rowPtr + i0, A.d_colIdx, A.d_val,  /* A rows [i0,i1) */
                A.d_rowPtr,      A.d_colIdx, A.d_val,  /* B = A (full)   */
                d_hkeys_b, d_hvals_b, d_hoff_b, d_hcap_b,
                d_nnzC_b, batch_rows);
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            CUDA_CHECK(cudaGetLastError());
        }
        float ft_hash; CUDA_CHECK(cudaEventElapsedTime(&ft_hash, ev0, ev1));
        t_hash_total_ms += (double)ft_hash;

        /* Download nnzC for this batch */
        int *h_nnzC_b = (int*)malloc((size_t)batch_rows * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_nnzC_b, d_nnzC_b,
                              (size_t)batch_rows * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < batch_rows; i++) h_nnzC[i0 + i] = h_nnzC_b[i];

        /* Build batch-local rowPtrC (device + host) */
        int *d_rowPtrC_b;
        CUDA_CHECK(cudaMalloc(&d_rowPtrC_b, (size_t)(batch_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_rowPtrC_b, 0, sizeof(int)));
        thrust::inclusive_scan(
            thrust::device_ptr<int>(d_nnzC_b),
            thrust::device_ptr<int>(d_nnzC_b + batch_rows),
            thrust::device_ptr<int>(d_rowPtrC_b + 1));
        int batch_nnzC = 0;
        CUDA_CHECK(cudaMemcpy(&batch_nnzC, d_rowPtrC_b + batch_rows,
                              sizeof(int), cudaMemcpyDeviceToHost));

        /* Grow host output buffer if needed */
        if (h_C_used + (size_t)batch_nnzC > h_C_cap) {
            h_C_cap = (h_C_used + (size_t)batch_nnzC) * 2 + 1024;
            h_ciC_all = (int*)   realloc(h_ciC_all, h_C_cap * sizeof(int));
            h_vC_all  = (double*)realloc(h_vC_all,  h_C_cap * sizeof(double));
            if (!h_ciC_all || !h_vC_all) {
                fprintf(stderr, "[RowSpGEMM] OOM growing output buffer\n"); return 1; }
        }

        /* Alloc batch device CSR output, run collect */
        size_t bnnzC_safe = (batch_nnzC > 0) ? (size_t)batch_nnzC : 1;
        int    *d_ciC_b;
        double *d_vC_b;
        CUDA_CHECK(cudaMalloc(&d_ciC_b, bnnzC_safe * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vC_b,  bnnzC_safe * sizeof(double)));
        {
            int blk = 256, grd = (batch_rows + blk - 1) / blk;
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(ev2));
            collect_kernel<<<grd, blk>>>(
                d_hkeys_b, d_hvals_b, d_hoff_b, d_hcap_b,
                d_rowPtrC_b, d_ciC_b, d_vC_b, batch_rows);
            CUDA_CHECK(cudaEventRecord(ev3));
            CUDA_CHECK(cudaEventSynchronize(ev3));
            CUDA_CHECK(cudaGetLastError());
        }
        float ft_collect; CUDA_CHECK(cudaEventElapsedTime(&ft_collect, ev2, ev3));
        t_collect_total_ms += (double)ft_collect;

        /* Download batch CSR to host buffer */
        if (batch_nnzC > 0) {
            CUDA_CHECK(cudaMemcpy(h_ciC_all + h_C_used, d_ciC_b,
                                  (size_t)batch_nnzC * sizeof(int),    cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vC_all  + h_C_used, d_vC_b,
                                  (size_t)batch_nnzC * sizeof(double), cudaMemcpyDeviceToHost));
        }
        h_C_used += (size_t)batch_nnzC;

        /* Free batch GPU memory */
        cudaFree(d_hkeys_b); cudaFree(d_hvals_b);
        cudaFree(d_hcap_b);  cudaFree(d_hoff_b);
        cudaFree(d_nnzC_b);  cudaFree(d_rowPtrC_b);
        cudaFree(d_ciC_b);   cudaFree(d_vC_b);
        free(h_nnzC_b);

        i0 = i1;
    } /* end batch loop */

    /* ── Build final rowPtrC and upload output to device ─────────────── */
    int nnzC = (int)h_C_used;
    int *h_rowPtrC = (int*)malloc((size_t)(A.rows + 1) * sizeof(int));
    h_rowPtrC[0] = 0;
    for (int i = 0; i < A.rows; i++)
        h_rowPtrC[i + 1] = h_rowPtrC[i] + h_nnzC[i];

    fprintf(stderr, "[RowSpGEMM] nnzC = %d\n", nnzC);

    /* ── Timing + stats ──────────────────────────────────────────────── */
    double time_ms = t_hash_total_ms + t_collect_total_ms;

    long long flops = 0;
    for (int i = 0; i < A.rows; i++)
        for (int jp = A.rowPtr[i]; jp < A.rowPtr[i+1]; jp++)
            flops += 2LL * (A.rowPtr[A.colIdx[jp]+1] - A.rowPtr[A.colIdx[jp]]);
    double gflops = (flops / 1e9) / (time_ms / 1e3);

    size_t mem_bytes =
        (size_t)(A.rows + 1) * sizeof(int) * 2 +
        (size_t)A.nnz * (sizeof(int) + sizeof(double)) * 2 +
        /* peak hash per batch — report worst-case single batch */
        (total_hash_entries < HASH_MEM_BUDGET / (sizeof(int) + sizeof(double))
             ? total_hash_entries
             : HASH_MEM_BUDGET / (sizeof(int) + sizeof(double)))
            * (sizeof(int) + sizeof(double)) +
        (size_t)nnzC * (sizeof(int) + sizeof(double)) +
        (size_t)(A.rows + 1) * sizeof(int);

    fprintf(stderr,
            "[RowSpGEMM] hash=%.3f ms  collect=%.3f ms  total=%.3f ms  %.4f GFlops\n",
            t_hash_total_ms, t_collect_total_ms, time_ms, gflops);

    printf("{\"algo\":\"RowSpGEMM\",\"matrix\":\"%s\","
           "\"time_ms\":%.4f,\"gflops\":%.6f,"
           "\"mem_bytes\":%zu,\"nnz_C\":%d,\"flops\":%lld}\n",
           mat_name, time_ms, gflops, mem_bytes, nnzC, flops);

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    cudaFree(A.d_rowPtr); cudaFree(A.d_colIdx); cudaFree(A.d_val);
    free(A.rowPtr); free(A.colIdx); free(A.val);
    free(h_hcap);
    free(h_nnzC); free(h_ciC_all); free(h_vC_all); free(h_rowPtrC);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaEventDestroy(ev2); cudaEventDestroy(ev3);
    return 0;
}