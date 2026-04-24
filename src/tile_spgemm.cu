/*
 * tile_spgemm.cu  —  OPTIMIZED TileSpGEMM (fully GPU-accelerated)
 * ─────────────────────────────────────────────────────────────────────────────
 * Optimizations over the baseline:
 *   1. Step 3 (numeric) fully on GPU  — 256 threads per block, one thread per
 *      output slot C[r][c]; all 256 threads work in parallel with zero
 *      inter-thread conflicts (no atomics needed in the numeric path).
 *   2. Precomputed d_tile_row array — O(1) tile-row lookup, eliminates
 *      expensive binary search inside Step 2 / Step 3 kernels.
 *   3. Tile nnz prefix sums on GPU — fast O(1) per-tile flat-array offset.
 *   4. Thrust inclusive_scan for C output allocation — stays fully on-device.
 *   5. Sort-based CSR → Tiled conversion — O(nnz) memory (no dense
 *      tilem×tilen map that blows up for large matrices).
 *   6. Step 2 kernel: fixed shared memory size, removed dead code.
 *   7. CPU numeric step eliminated entirely.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "common.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  TILE CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════ */
#define TILE_DIM        16
#define TILE_SIZE       256         /* TILE_DIM * TILE_DIM                   */
#define WARP_SIZE       32
#define WARPS_PER_BLOCK 8           /* Step-2 kernel warps per block         */
/* Max matched intermediate tile-pairs per output C-tile.
   Supports matrices with up to TILE_DIM * MAX_PAIRS = 4096 columns.        */
#define MAX_PAIRS       256

/* ═══════════════════════════════════════════════════════════════════════════
 *  TILED SPARSE FORMAT
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int   tilem, tilen, numTiles, nnz;
    /* host arrays */
    int            *h_tilePtr;
    int            *h_tileColIdx;
    int            *h_tileNnz;
    int            *h_tileNnzPrefix;   /* [numTiles+1] exclusive prefix sum  */
    unsigned char  *h_rowPtr;          /* [numTiles * TILE_DIM]              */
    unsigned char  *h_rowIdx;          /* [nnz]                              */
    unsigned char  *h_colIdx;          /* [nnz]                              */
    double         *h_val;             /* [nnz]                              */
    unsigned short *h_mask;            /* [numTiles * TILE_DIM] row bitmasks */
    /* device mirrors */
    int            *d_tilePtr;
    int            *d_tileColIdx;
    int            *d_tileNnz;
    int            *d_tileNnzPrefix;
    unsigned char  *d_rowPtr;
    unsigned char  *d_rowIdx;
    unsigned char  *d_colIdx;
    double         *d_val;
    unsigned short *d_mask;
} TiledMatrix;

/* ─── Sort key for O(nnz) CSR → Tiled conversion ────────────────────────── */
typedef struct { int tr, tc, lr, lc, orig; } TileElem;

static int cmp_tile_elem(const void *a, const void *b)
{
    const TileElem *ea = (const TileElem *)a;
    const TileElem *eb = (const TileElem *)b;
    if (ea->tr != eb->tr) return ea->tr - eb->tr;
    if (ea->tc != eb->tc) return ea->tc - eb->tc;
    if (ea->lr != eb->lr) return ea->lr - eb->lr;
    return ea->lc - eb->lc;
}

/* ─── Convert CSR → TiledMatrix (sort-based, O(nnz) memory) ─────────────── */
static void csr_to_tiled(const CsrMatrix *A, TiledMatrix *T)
{
    int tilem = (A->rows + TILE_DIM - 1) / TILE_DIM;
    int tilen = (A->cols + TILE_DIM - 1) / TILE_DIM;
    T->tilem = tilem;
    T->tilen = tilen;
    T->nnz   = A->nnz;

    /* Collect all (tr, tc, lr, lc, orig_idx) and sort */
    TileElem *elems = (TileElem *)malloc((size_t)A->nnz * sizeof(TileElem));
    int idx = 0;
    for (int i = 0; i < A->rows; i++) {
        int tr = i / TILE_DIM, lr = i % TILE_DIM;
        for (int jp = A->rowPtr[i]; jp < A->rowPtr[i+1]; jp++) {
            int j  = A->colIdx[jp];
            elems[idx++] = (TileElem){ tr, j / TILE_DIM, lr, j % TILE_DIM, jp };
        }
    }
    qsort(elems, A->nnz, sizeof(TileElem), cmp_tile_elem);

    /* Pass 1: count unique tiles per tile-row */
    T->h_tilePtr = (int *)calloc(tilem + 1, sizeof(int));
    {
        int cur_tr = -1, cur_tc = -1;
        for (int k = 0; k < A->nnz; k++) {
            if (elems[k].tr != cur_tr || elems[k].tc != cur_tc) {
                T->h_tilePtr[elems[k].tr + 1]++;
                cur_tr = elems[k].tr; cur_tc = elems[k].tc;
            }
        }
    }
    for (int tr = 0; tr < tilem; tr++)
        T->h_tilePtr[tr+1] += T->h_tilePtr[tr];
    T->numTiles = T->h_tilePtr[tilem];

    /* Allocate structure arrays */
    T->h_tileColIdx    = (int *)malloc(T->numTiles * sizeof(int));
    T->h_tileNnz       = (int *)calloc(T->numTiles, sizeof(int));
    T->h_tileNnzPrefix = (int *)malloc((T->numTiles + 1) * sizeof(int));
    T->h_rowPtr  = (unsigned char *) calloc(T->numTiles * TILE_DIM, sizeof(unsigned char));
    T->h_rowIdx  = (unsigned char *) malloc((size_t)A->nnz * sizeof(unsigned char));
    T->h_colIdx  = (unsigned char *) malloc((size_t)A->nnz * sizeof(unsigned char));
    T->h_val     = (double *)        malloc((size_t)A->nnz * sizeof(double));
    T->h_mask    = (unsigned short *)calloc(T->numTiles * TILE_DIM, sizeof(unsigned short));

    /* Pass 2a: fill tileColIdx and tileNnz counts */
    {
        int cur_tr = -1, cur_tc = -1, tp = -1;
        for (int k = 0; k < A->nnz; k++) {
            if (elems[k].tr != cur_tr || elems[k].tc != cur_tc) {
                tp++;
                T->h_tileColIdx[tp] = elems[k].tc;
                cur_tr = elems[k].tr; cur_tc = elems[k].tc;
            }
            T->h_tileNnz[tp]++;
        }
    }

    /* Build tileNnzPrefix */
    T->h_tileNnzPrefix[0] = 0;
    for (int t = 0; t < T->numTiles; t++)
        T->h_tileNnzPrefix[t+1] = T->h_tileNnzPrefix[t] + T->h_tileNnz[t];

    /* Pass 2b: fill data using prefix-initialized write cursors */
    int *wc = (int *)malloc(T->numTiles * sizeof(int));
    for (int t = 0; t < T->numTiles; t++) wc[t] = T->h_tileNnzPrefix[t];
    {
        int cur_tr = -1, cur_tc = -1, tp = -1;
        for (int k = 0; k < A->nnz; k++) {
            TileElem *e = &elems[k];
            if (e->tr != cur_tr || e->tc != cur_tc) {
                tp++; cur_tr = e->tr; cur_tc = e->tc;
            }
            int pos      = wc[tp]++;
            T->h_rowIdx[pos] = (unsigned char)e->lr;
            T->h_colIdx[pos] = (unsigned char)e->lc;
            T->h_val[pos]    = A->val[e->orig];
            T->h_mask[tp * TILE_DIM + e->lr] |= (unsigned short)(1u << e->lc);
        }
    }
    free(wc);

    /* Build rowPtr (per-row prefix sums within each tile) */
    int *row_counts = (int *)calloc(T->numTiles * TILE_DIM, sizeof(int));
    for (int t = 0; t < T->numTiles; t++) {
        int off = T->h_tileNnzPrefix[t], nnz_t = T->h_tileNnz[t];
        for (int k = 0; k < nnz_t; k++)
            row_counts[t * TILE_DIM + T->h_rowIdx[off + k]]++;
    }
    for (int t = 0; t < T->numTiles; t++) {
        unsigned char acc = 0;
        for (int r = 0; r < TILE_DIM; r++) {
            T->h_rowPtr[t * TILE_DIM + r] = acc;
            acc += (unsigned char)row_counts[t * TILE_DIM + r];
        }
    }
    free(row_counts);
    free(elems);

    T->d_tilePtr = NULL; T->d_tileColIdx = NULL; T->d_tileNnz = NULL;
    T->d_tileNnzPrefix = NULL;
    T->d_rowPtr = NULL;  T->d_rowIdx = NULL;   T->d_colIdx = NULL;
    T->d_val    = NULL;  T->d_mask   = NULL;
}

/* ─── Upload TiledMatrix to GPU ─────────────────────────────────────────── */
static double tiled_upload(TiledMatrix *T)
{
    double t0 = wtime();
    CUDA_CHECK(cudaMalloc(&T->d_tilePtr,       (T->tilem+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_tileColIdx,     T->numTiles*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_tileNnz,        T->numTiles*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_tileNnzPrefix,  (T->numTiles+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_rowPtr,         T->numTiles*TILE_DIM*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&T->d_rowIdx,         T->nnz*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&T->d_colIdx,         T->nnz*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&T->d_val,            T->nnz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&T->d_mask,           T->numTiles*TILE_DIM*sizeof(unsigned short)));

    CUDA_CHECK(cudaMemcpy(T->d_tilePtr,       T->h_tilePtr,       (T->tilem+1)*sizeof(int),                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_tileColIdx,    T->h_tileColIdx,    T->numTiles*sizeof(int),                    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_tileNnz,       T->h_tileNnz,       T->numTiles*sizeof(int),                    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_tileNnzPrefix, T->h_tileNnzPrefix, (T->numTiles+1)*sizeof(int),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_rowPtr,        T->h_rowPtr,        T->numTiles*TILE_DIM*sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_rowIdx,        T->h_rowIdx,        T->nnz*sizeof(unsigned char),               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_colIdx,        T->h_colIdx,        T->nnz*sizeof(unsigned char),               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_val,           T->h_val,           T->nnz*sizeof(double),                      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_mask,          T->h_mask,          T->numTiles*TILE_DIM*sizeof(unsigned short),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    return (wtime() - t0) * 1e3;
}

static void tiled_free(TiledMatrix *T)
{
    free(T->h_tilePtr);  free(T->h_tileColIdx); free(T->h_tileNnz);
    free(T->h_tileNnzPrefix);
    free(T->h_rowPtr);   free(T->h_rowIdx);     free(T->h_colIdx);
    free(T->h_val);      free(T->h_mask);
    if (T->d_tilePtr)       cudaFree(T->d_tilePtr);
    if (T->d_tileColIdx)    cudaFree(T->d_tileColIdx);
    if (T->d_tileNnz)       cudaFree(T->d_tileNnz);
    if (T->d_tileNnzPrefix) cudaFree(T->d_tileNnzPrefix);
    if (T->d_rowPtr)        cudaFree(T->d_rowPtr);
    if (T->d_rowIdx)        cudaFree(T->d_rowIdx);
    if (T->d_colIdx)        cudaFree(T->d_colIdx);
    if (T->d_val)           cudaFree(T->d_val);
    if (T->d_mask)          cudaFree(T->d_mask);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  STEP 1 – Symbolic SpGEMM on tile-level matrices using cuSPARSE
 * ═══════════════════════════════════════════════════════════════════════════ */
static double step1_tile_structure(const TiledMatrix *A, const TiledMatrix *B,
                                    int **h_tilePtrC_out,
                                    int **h_tileColIdxC_out,
                                    int  *numTilesC_out)
{
    int rowsAp = A->tilem, colsAp = A->tilen, nnzAp = A->numTiles;
    int rowsBp = B->tilem, colsBp = B->tilen, nnzBp = B->numTiles;

    double *h_valAp = (double *)malloc(nnzAp * sizeof(double));
    double *h_valBp = (double *)malloc(nnzBp * sizeof(double));
    for (int i = 0; i < nnzAp; i++) h_valAp[i] = 1.0;
    for (int i = 0; i < nnzBp; i++) h_valBp[i] = 1.0;

    int *d_rpAp, *d_ciAp; double *d_vAp;
    int *d_rpBp, *d_ciBp; double *d_vBp;
    CUDA_CHECK(cudaMalloc(&d_rpAp, (rowsAp+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ciAp, nnzAp*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vAp,  nnzAp*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rpBp, (rowsBp+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ciBp, nnzBp*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vBp,  nnzBp*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rpAp, A->h_tilePtr,    (rowsAp+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ciAp, A->h_tileColIdx, nnzAp*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vAp,  h_valAp,         nnzAp*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rpBp, B->h_tilePtr,    (rowsBp+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ciBp, B->h_tileColIdx, nnzBp*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vBp,  h_valBp,         nnzBp*sizeof(double),     cudaMemcpyHostToDevice));
    free(h_valAp); free(h_valBp);

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t matAp, matBp, matCp;
    CUSPARSE_CHECK(cusparseCreateCsr(&matAp, rowsAp, colsAp, nnzAp,
        d_rpAp, d_ciAp, d_vAp, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matBp, rowsBp, colsBp, nnzBp,
        d_rpBp, d_ciBp, d_vBp, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    int *d_rpCp;
    CUDA_CHECK(cudaMalloc(&d_rpCp, (rowsAp+1)*sizeof(int)));
    CUSPARSE_CHECK(cusparseCreateCsr(&matCp, rowsAp, colsBp, 0, d_rpCp,
        NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    double alpha = 1.0, beta = 0.0;
    cusparseSpGEMMDescr_t desc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&desc));
    size_t bs1 = 0; void *b1 = NULL;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAp, matBp, &beta, matCp, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &bs1, NULL));
    CUDA_CHECK(cudaMalloc(&b1, bs1 ? bs1 : 1));
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAp, matBp, &beta, matCp, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &bs1, b1));
    size_t bs2 = 0; void *b2 = NULL;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAp, matBp, &beta, matCp, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &bs2, NULL));
    CUDA_CHECK(cudaMalloc(&b2, bs2 ? bs2 : 1));

    double t0 = wtime();
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAp, matBp, &beta, matCp, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &bs2, b2));
    int64_t nr, nc, nnzCp;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matCp, &nr, &nc, &nnzCp));
    int *d_ciCp; double *d_vCp;
    CUDA_CHECK(cudaMalloc(&d_ciCp, nnzCp * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vCp,  nnzCp * sizeof(double)));
    CUSPARSE_CHECK(cusparseCsrSetPointers(matCp, d_rpCp, d_ciCp, d_vCp));
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAp, matBp, &beta, matCp, CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT, desc));
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed_ms = (wtime() - t0) * 1e3;

    *h_tilePtrC_out    = (int *)malloc((rowsAp+1) * sizeof(int));
    *h_tileColIdxC_out = (int *)malloc(nnzCp * sizeof(int));
    CUDA_CHECK(cudaMemcpy(*h_tilePtrC_out,    d_rpCp, (rowsAp+1)*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*h_tileColIdxC_out, d_ciCp, nnzCp*sizeof(int),      cudaMemcpyDeviceToHost));
    *numTilesC_out = (int)nnzCp;

    cudaFree(b1); cudaFree(b2);
    cudaFree(d_rpAp); cudaFree(d_ciAp); cudaFree(d_vAp);
    cudaFree(d_rpBp); cudaFree(d_ciBp); cudaFree(d_vBp);
    cudaFree(d_rpCp); cudaFree(d_ciCp); cudaFree(d_vCp);
    cusparseSpGEMM_destroyDescr(desc);
    cusparseDestroySpMat(matAp); cusparseDestroySpMat(matBp);
    cusparseDestroySpMat(matCp);
    cusparseDestroy(handle);
    return elapsed_ms;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  STEP 2 – Symbolic phase kernel
 *  One warp per output C-tile.
 *  Each lane handles one A_ik tile; binary-searches matching B_kj.
 *  AtomicOr on per-warp shared mask → C bitmask + rowPtr + nnz per tile.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__
void step2_symbolic_kernel(
    const int            *A_tilePtr,
    const int            *A_tileColIdx,
    const int            *A_tileNnz,
    const unsigned char  *A_rowPtr,
    const unsigned short *A_mask,
    const int            *B_tilePtr,
    const int            *B_tileColIdx,
    const int            *B_tileNnz,
    const unsigned char  *B_rowPtr,
    const unsigned short *B_mask,
    const int            *C_tileColIdx,
    const int            *d_tile_row,   /* [numTilesC] precomputed tile-row  */
    int                  *C_tileNnz,
    unsigned char        *C_rowPtr,
    unsigned short       *C_mask,
    int numTilesC)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int wid  = tid  / WARP_SIZE;
    int lane = tid  % WARP_SIZE;
    int wb   = threadIdx.x / WARP_SIZE;

    if (wid >= numTilesC) return;

    __shared__ unsigned int s_mask[WARPS_PER_BLOCK][TILE_DIM];
    if (lane < TILE_DIM) s_mask[wb][lane] = 0u;
    __syncwarp();

    int tile_i = d_tile_row[wid];
    int tile_j = C_tileColIdx[wid];

    int lenA  = A_tilePtr[tile_i+1] - A_tilePtr[tile_i];
    int lenB  = B_tilePtr[tile_j+1] - B_tilePtr[tile_j];
    int baseA = A_tilePtr[tile_i];
    int baseB = B_tilePtr[tile_j];

    for (int ia = lane; ia < lenA; ia += WARP_SIZE) {
        int col_a = A_tileColIdx[baseA + ia];
        int lo = 0, hi = lenB - 1, found_b = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int v   = B_tileColIdx[baseB + mid];
            if      (v == col_a) { found_b = mid; break; }
            else if (v <  col_a)   lo = mid + 1;
            else                   hi = mid - 1;
        }
        if (found_b < 0) continue;

        int posA = baseA + ia;
        int posB = baseB + found_b;
        (void)A_tileNnz; (void)B_tileNnz; (void)A_rowPtr; (void)B_rowPtr;

        for (int r = 0; r < TILE_DIM; r++) {
            unsigned short maskA_row = A_mask[posA * TILE_DIM + r];
            while (maskA_row) {
                int c = __ffs((int)(unsigned int)maskA_row) - 1;
                maskA_row &= (unsigned short)(maskA_row - 1);
                atomicOr(&s_mask[wb][r],
                         (unsigned int)B_mask[posB * TILE_DIM + c]);
            }
        }
    }
    __syncwarp();

    if (lane < TILE_DIM)
        C_mask[wid * TILE_DIM + lane] =
            (unsigned short)(s_mask[wb][lane] & 0xFFFFu);
    __syncwarp();

    if (lane == 0) {
        int total = 0;
        unsigned char acc = 0;
        for (int r = 0; r < TILE_DIM; r++) {
            C_rowPtr[wid * TILE_DIM + r] = acc;
            int cnt  = __popc(s_mask[wb][r]);
            total   += cnt;
            acc     += (unsigned char)cnt;
        }
        C_tileNnz[wid] = total;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  STEP 3 – Numeric phase kernel (fully GPU, no CPU fallback)
 *
 *  Grid:  numTilesC blocks (one block per output tile)
 *  Block: TILE_SIZE = 256 threads (one thread per output slot C[r][c])
 *
 *  Thread slot = r*16+c independently computes:
 *    C[r][c] = Σ_{matched pairs p} Σ_{ka: A[r][ca]≠0} A[r][ca] * B[ca][c]
 *
 *  No atomics — each thread writes to a unique output position.
 *  Thread 0 performs all binary searches once; result shared via s_posA/s_posB.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__
void step3_numeric_kernel(
    /* A */
    const int            *A_tilePtr,
    const int            *A_tileColIdx,
    const int            *A_tileNnzPrefix,
    const unsigned char  *A_rowPtr,
    const unsigned char  *A_colIdx,
    const double         *A_val,
    /* B */
    const int            *B_tilePtr,
    const int            *B_tileColIdx,
    const int            *B_tileNnzPrefix,
    const unsigned char  *B_rowPtr,
    const unsigned char  *B_colIdx,
    const double         *B_val,
    /* C structure */
    const int            *C_tileColIdx,
    const int            *C_tileNnzPrefix,
    const unsigned char  *C_rowPtr,
    const unsigned short *C_mask,
    const int            *d_tile_row,
    /* C outputs */
    unsigned char        *C_rowIdx_out,
    unsigned char        *C_colIdx_out,
    double               *C_val_out,
    int numTilesC)
{
    int tile_idx = blockIdx.x;
    int slot     = threadIdx.x;

    if (tile_idx >= numTilesC) return;

    int r = slot >> 4;
    int c = slot & 15;

    int tile_i = d_tile_row[tile_idx];
    int tile_j = C_tileColIdx[tile_idx];
    int lenA   = A_tilePtr[tile_i+1] - A_tilePtr[tile_i];
    int lenB   = B_tilePtr[tile_j+1] - B_tilePtr[tile_j];
    int baseA  = A_tilePtr[tile_i];
    int baseB  = B_tilePtr[tile_j];

    /* Thread 0 finds all matched pairs; stored in shared memory */
    __shared__ int s_posA[MAX_PAIRS];
    __shared__ int s_posB[MAX_PAIRS];
    __shared__ int s_npairs;

    if (slot == 0) {
        int np = 0;
        for (int ia = 0; ia < lenA; ia++) {
            int col_a = A_tileColIdx[baseA + ia];
            int lo = 0, hi = lenB - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                int v   = B_tileColIdx[baseB + mid];
                if (v == col_a) {
                    if (np < MAX_PAIRS) {
                        s_posA[np] = baseA + ia;
                        s_posB[np] = baseB + mid;
                        np++;
                    }
                    break;
                } else if (v < col_a) lo = mid + 1;
                else                   hi = mid - 1;
            }
        }
        s_npairs = np;
    }
    __syncthreads();

    /* Each thread independently accumulates C[r][c] */
    double sum = 0.0;
    int    np  = s_npairs;

    for (int p = 0; p < np; p++) {
        int posA   = s_posA[p];
        int posB   = s_posB[p];
        int offA   = A_tileNnzPrefix[posA];
        int nnzA_t = A_tileNnzPrefix[posA+1] - offA;
        int offB   = B_tileNnzPrefix[posB];
        int nnzB_t = B_tileNnzPrefix[posB+1] - offB;

        /* Iterate nonzeros in A tile's row r */
        int raStart = (int)A_rowPtr[posA * TILE_DIM + r];
        int raEnd   = (r < TILE_DIM-1)
                    ? (int)A_rowPtr[posA * TILE_DIM + r + 1]
                    : nnzA_t;

        for (int ka = raStart; ka < raEnd; ka++) {
            int    ca = (int)A_colIdx[offA + ka];
            double va = A_val[offA + ka];

            /* Find B[ca][c] — scan B tile's row ca (at most 16 entries) */
            int bcStart = (int)B_rowPtr[posB * TILE_DIM + ca];
            int bcEnd   = (ca < TILE_DIM-1)
                        ? (int)B_rowPtr[posB * TILE_DIM + ca + 1]
                        : nnzB_t;

            for (int kb = bcStart; kb < bcEnd; kb++) {
                if ((int)B_colIdx[offB + kb] == c) {
                    sum += va * B_val[offB + kb];
                    break;
                }
            }
        }
    }

    /* Write output only if (r,c) is structurally present in C */
    unsigned short row_mask = C_mask[tile_idx * TILE_DIM + r];
    if (!((row_mask >> c) & 1u)) return;

    int outBase = C_tileNnzPrefix[tile_idx];
    int rstart  = (int)C_rowPtr[tile_idx * TILE_DIM + r];
    int pos     = __popc((unsigned int)(row_mask & ((1u << c) - 1u)));

    int outIdx = outBase + rstart + pos;
    C_rowIdx_out[outIdx] = (unsigned char)r;
    C_colIdx_out[outIdx] = (unsigned char)c;
    C_val_out[outIdx]    = sum;
}

/* ─── Convert tiled C back to CSR (for --save-c correctness check) ────────  */
static void tiled_C_to_csr(
    const int *h_tilePtrC, const int *h_tileColIdxC,
    const int *h_tileNnzC,
    const unsigned char *h_rowIdxC, const unsigned char *h_colIdxC,
    const double *h_valC,
    int numTilesC, int tilem, int tilen,
    int rows, int cols,
    int **h_rowPtrOut, int **h_colIdxOut, double **h_valOut, int *nnzOut)
{
    int *rowCnt = (int *)calloc(rows, sizeof(int));
    int *prefix = (int *)malloc((numTilesC+1)*sizeof(int));
    prefix[0] = 0;
    for (int t = 0; t < numTilesC; t++) prefix[t+1] = prefix[t] + h_tileNnzC[t];
    int total = prefix[numTilesC];

    for (int wid = 0; wid < numTilesC; wid++) {
        int tile_i = -1;
        for (int tr = 0; tr < tilem; tr++)
            if (h_tilePtrC[tr] <= wid && wid < h_tilePtrC[tr+1]) { tile_i = tr; break; }
        if (tile_i < 0) continue;
        int base = prefix[wid];
        for (int k = 0; k < h_tileNnzC[wid]; k++) {
            int gr = tile_i * TILE_DIM + (int)h_rowIdxC[base + k];
            if (gr < rows) rowCnt[gr]++;
        }
    }
    *nnzOut      = total;
    *h_rowPtrOut = (int *)malloc((rows+1)*sizeof(int));
    (*h_rowPtrOut)[0] = 0;
    for (int r = 0; r < rows; r++) (*h_rowPtrOut)[r+1] = (*h_rowPtrOut)[r] + rowCnt[r];
    *h_colIdxOut = (int *)   malloc(total*sizeof(int));
    *h_valOut    = (double *)malloc(total*sizeof(double));
    int *cursor  = (int *)calloc(rows, sizeof(int));
    for (int wid = 0; wid < numTilesC; wid++) {
        int tile_i = -1;
        for (int tr = 0; tr < tilem; tr++)
            if (h_tilePtrC[tr] <= wid && wid < h_tilePtrC[tr+1]) { tile_i = tr; break; }
        if (tile_i < 0) continue;
        int tile_j = h_tileColIdxC[wid];
        int base   = prefix[wid];
        for (int k = 0; k < h_tileNnzC[wid]; k++) {
            int gr = tile_i * TILE_DIM + (int)h_rowIdxC[base + k];
            int gc = tile_j * TILE_DIM + (int)h_colIdxC[base + k];
            if (gr >= rows || gc >= cols) continue;
            int p = (*h_rowPtrOut)[gr] + cursor[gr]++;
            (*h_colIdxOut)[p] = gc;
            (*h_valOut)[p]    = h_valC[base + k];
        }
    }
    free(rowCnt); free(prefix); free(cursor);
    (void)tilen;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: tile_spgemm <csr_binary> <matrix_name> [--save-c <out.bin>]\n");
        return 1;
    }
    const char *csr_path  = argv[1];
    const char *mat_name  = argv[2];
    int do_save           = (argc >= 5 && strcmp(argv[3], "--save-c") == 0);
    const char *save_path = do_save ? argv[4] : NULL;

    /* ── Load CSR ──────────────────────────────────────────────────────── */
    fprintf(stderr, "[TileSpGEMM] Loading '%s' from %s ...\n", mat_name, csr_path);
    CsrMatrix A;
    if (csr_load_binary(csr_path, &A) != 0) return 1;
    fprintf(stderr, "[TileSpGEMM] %d x %d, nnz=%d\n", A.rows, A.cols, A.nnz);

    /* ── CSR → Tiled conversion ──────────────────────────────────────── */
    fprintf(stderr, "[TileSpGEMM] CSR → Tiled ...\n");
    double t_conv_start = wtime();
    TiledMatrix TA;
    csr_to_tiled(&A, &TA);
    double t_conv_ms = (wtime() - t_conv_start) * 1e3;
    fprintf(stderr, "[TileSpGEMM] %d tiles, conv=%.3f ms\n", TA.numTiles, t_conv_ms);

    size_t tiled_bytes =
        (size_t)(TA.tilem+1)*sizeof(int) +
        (size_t)TA.numTiles * sizeof(int) * 2 +
        (size_t)TA.numTiles * TILE_DIM * (sizeof(unsigned char)+sizeof(unsigned short)) +
        (size_t)TA.nnz * (2*sizeof(unsigned char)+sizeof(double));
    size_t csr_bytes =
        (size_t)(A.rows+1)*sizeof(int) +
        (size_t)A.nnz * (sizeof(int)+sizeof(double));

    /* ── Upload to GPU ──────────────────────────────────────────────── */
    fprintf(stderr, "[TileSpGEMM] Uploading to GPU ...\n");
    double t_upload_ms = tiled_upload(&TA);
    /* B = A for squaring — share all device pointers */
    TiledMatrix TB;
    memcpy(&TB, &TA, sizeof(TiledMatrix));

    /* ═══ STEP 1 ═══════════════════════════════════════════════════════ */
    fprintf(stderr, "[TileSpGEMM] Step 1: tile-level symbolic (cuSPARSE) ...\n");
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    int *h_tilePtrC = NULL, *h_tileColIdxC = NULL;
    int  numTilesC  = 0;
    double t_step1_ms = step1_tile_structure(&TA, &TB,
                                              &h_tilePtrC, &h_tileColIdxC,
                                              &numTilesC);
    fprintf(stderr, "[TileSpGEMM] Step 1: %d C-tiles, %.3f ms\n",
            numTilesC, t_step1_ms);

    /* ── Precompute tile-row lookup: d_tile_row[wid] = tile row of C-tile wid */
    int *h_tile_row = (int *)malloc(numTilesC * sizeof(int));
    for (int tr = 0; tr < TA.tilem; tr++)
        for (int wid = h_tilePtrC[tr]; wid < h_tilePtrC[tr+1]; wid++)
            h_tile_row[wid] = tr;
    int *d_tile_row;
    CUDA_CHECK(cudaMalloc(&d_tile_row, numTilesC * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tile_row, h_tile_row, numTilesC*sizeof(int),
                          cudaMemcpyHostToDevice));
    free(h_tile_row);

    /* ── Allocate C symbolic device arrays ─────────────────────────── */
    int            *d_tilePtrC, *d_tileColIdxC, *d_tileNnzC;
    unsigned char  *d_rowPtrC;
    unsigned short *d_maskC;
    CUDA_CHECK(cudaMalloc(&d_tilePtrC,    (TA.tilem+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tileColIdxC,  numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tileNnzC,     numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowPtrC,      numTilesC*TILE_DIM*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_maskC,        numTilesC*TILE_DIM*sizeof(unsigned short)));
    CUDA_CHECK(cudaMemcpy(d_tilePtrC,    h_tilePtrC,    (TA.tilem+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tileColIdxC, h_tileColIdxC, numTilesC*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tileNnzC, 0, numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_rowPtrC,  0, numTilesC*TILE_DIM*sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_maskC,    0, numTilesC*TILE_DIM*sizeof(unsigned short)));

    /* ═══ STEP 2 ═══════════════════════════════════════════════════════ */
    fprintf(stderr, "[TileSpGEMM] Step 2: GPU symbolic ...\n");
    {
        int threads = WARPS_PER_BLOCK * WARP_SIZE;
        int blocks  = (numTilesC + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (blocks < 1) blocks = 1;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev0));
        step2_symbolic_kernel<<<blocks, threads>>>(
            TA.d_tilePtr, TA.d_tileColIdx, TA.d_tileNnz,
            TA.d_rowPtr, TA.d_mask,
            TB.d_tilePtr, TB.d_tileColIdx, TB.d_tileNnz,
            TB.d_rowPtr, TB.d_mask,
            d_tileColIdxC, d_tile_row,
            d_tileNnzC, d_rowPtrC, d_maskC,
            numTilesC);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());
    }
    float f2; CUDA_CHECK(cudaEventElapsedTime(&f2, ev0, ev1));
    double t_step2_ms = (double)f2;
    fprintf(stderr, "[TileSpGEMM] Step 2: %.3f ms\n", t_step2_ms);

    /* ── GPU prefix sum (thrust) — fully on-device ─────────────────── */
    int *d_tileNnzPrefixC;
    CUDA_CHECK(cudaMalloc(&d_tileNnzPrefixC, (numTilesC+1)*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tileNnzPrefixC, 0, sizeof(int)));
    if (numTilesC > 0) {
        thrust::inclusive_scan(
            thrust::device_ptr<int>(d_tileNnzC),
            thrust::device_ptr<int>(d_tileNnzC + numTilesC),
            thrust::device_ptr<int>(d_tileNnzPrefixC + 1));
    }
    int nnzC_total = 0;
    CUDA_CHECK(cudaMemcpy(&nnzC_total, d_tileNnzPrefixC + numTilesC,
                          sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stderr, "[TileSpGEMM] C nnz=%d\n", nnzC_total);

    /* ── Allocate C numeric output arrays ─────────────────────────── */
    size_t nnzC_safe = (nnzC_total > 0) ? (size_t)nnzC_total : 1;
    unsigned char *d_rowIdxC, *d_colIdxC;
    double        *d_valC;
    CUDA_CHECK(cudaMalloc(&d_rowIdxC, nnzC_safe * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_colIdxC, nnzC_safe * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_valC,    nnzC_safe * sizeof(double)));

    /* ═══ STEP 3 ═══════════════════════════════════════════════════════ */
    fprintf(stderr, "[TileSpGEMM] Step 3: GPU numeric (256 threads/tile) ...\n");
    {
        int blocks = (numTilesC > 0) ? numTilesC : 1;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev0));
        step3_numeric_kernel<<<blocks, TILE_SIZE>>>(
            TA.d_tilePtr, TA.d_tileColIdx, TA.d_tileNnzPrefix,
            TA.d_rowPtr,  TA.d_colIdx, TA.d_val,
            TB.d_tilePtr, TB.d_tileColIdx, TB.d_tileNnzPrefix,
            TB.d_rowPtr,  TB.d_colIdx, TB.d_val,
            d_tileColIdxC, d_tileNnzPrefixC, d_rowPtrC, d_maskC, d_tile_row,
            d_rowIdxC, d_colIdxC, d_valC,
            numTilesC);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());
    }
    float f3; CUDA_CHECK(cudaEventElapsedTime(&f3, ev0, ev1));
    double t_step3_ms = (double)f3;
    fprintf(stderr, "[TileSpGEMM] Step 3: %.3f ms\n", t_step3_ms);

    /* ── Stats ─────────────────────────────────────────────────────── */
    double t_total_ms = t_step1_ms + t_step2_ms + t_step3_ms;

    long long flops = 0;
    for (int i = 0; i < A.rows; i++)
        for (int jp = A.rowPtr[i]; jp < A.rowPtr[i+1]; jp++)
            flops += 2LL * (A.rowPtr[A.colIdx[jp]+1] - A.rowPtr[A.colIdx[jp]]);
    double gflops = (flops / 1e9) / (t_total_ms / 1e3);

    size_t peak_bytes =
        2 * tiled_bytes +
        (size_t)numTilesC *
            (sizeof(int)*2 + TILE_DIM*(sizeof(unsigned char)+sizeof(unsigned short))) +
        (size_t)nnzC_total * (2*sizeof(unsigned char) + sizeof(double));

    fprintf(stderr, "[TileSpGEMM] Total: %.3f ms  %.6f GFlops  upload=%.3f ms\n",
            t_total_ms, gflops, t_upload_ms);

    printf("{\"algo\":\"TileSpGEMM\",\"matrix\":\"%s\","
           "\"time_ms\":%.4f,\"gflops\":%.6f,"
           "\"mem_bytes\":%zu,\"nnz_C\":%d,\"flops\":%lld,"
           "\"step1_ms\":%.4f,\"step2_ms\":%.4f,\"step3_ms\":%.4f,"
           "\"conversion_ms\":%.4f,"
           "\"tiled_bytes\":%zu,\"csr_bytes\":%zu}\n",
           mat_name, t_total_ms, gflops, peak_bytes,
           nnzC_total, flops,
           t_step1_ms, t_step2_ms, t_step3_ms,
           t_conv_ms, tiled_bytes, csr_bytes);

    /* ── Optional: save C as CSR binary ────────────────────────────── */
    if (do_save && nnzC_total > 0) {
        fprintf(stderr, "[TileSpGEMM] Saving C to %s ...\n", save_path);
        int           *h_tileNnzC_h  = (int *)malloc(numTilesC*sizeof(int));
        unsigned char *h_rowIdxC_h   = (unsigned char *)malloc((size_t)nnzC_total);
        unsigned char *h_colIdxC_h   = (unsigned char *)malloc((size_t)nnzC_total);
        double        *h_valC_h      = (double *)malloc((size_t)nnzC_total*sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_tileNnzC_h, d_tileNnzC,  numTilesC*sizeof(int),                 cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rowIdxC_h,  d_rowIdxC,   (size_t)nnzC_total*sizeof(unsigned char), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_colIdxC_h,  d_colIdxC,   (size_t)nnzC_total*sizeof(unsigned char), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_valC_h,     d_valC,      (size_t)nnzC_total*sizeof(double),         cudaMemcpyDeviceToHost));

        int *h_rpC = NULL, *h_ciC = NULL; double *h_vC = NULL; int nnzCSR = 0;
        tiled_C_to_csr(h_tilePtrC, h_tileColIdxC, h_tileNnzC_h,
                       h_rowIdxC_h, h_colIdxC_h, h_valC_h,
                       numTilesC, TA.tilem, TA.tilen, A.rows, A.cols,
                       &h_rpC, &h_ciC, &h_vC, &nnzCSR);
        FILE *fp = fopen(save_path, "wb");
        if (fp) {
            fwrite(&A.rows, sizeof(int),    1,       fp);
            fwrite(&A.cols, sizeof(int),    1,       fp);
            fwrite(&nnzCSR, sizeof(int),    1,       fp);
            fwrite(h_rpC,   sizeof(int),    A.rows+1,fp);
            fwrite(h_ciC,   sizeof(int),    nnzCSR,  fp);
            fwrite(h_vC,    sizeof(double), nnzCSR,  fp);
            fclose(fp);
        }
        free(h_tileNnzC_h); free(h_rowIdxC_h); free(h_colIdxC_h); free(h_valC_h);
        free(h_rpC); free(h_ciC); free(h_vC);
    }

    /* ── Cleanup ───────────────────────────────────────────────────── */
    free(h_tilePtrC); free(h_tileColIdxC);
    cudaFree(d_tile_row);
    cudaFree(d_tilePtrC); cudaFree(d_tileColIdxC);
    cudaFree(d_tileNnzC); cudaFree(d_rowPtrC); cudaFree(d_maskC);
    cudaFree(d_tileNnzPrefixC);
    cudaFree(d_rowIdxC); cudaFree(d_colIdxC); cudaFree(d_valC);

    /* TB shares TA's device/host pointers — null before tiled_free */
    TB.d_tilePtr = NULL; TB.d_tileColIdx = NULL; TB.d_tileNnz = NULL;
    TB.d_tileNnzPrefix = NULL;
    TB.d_rowPtr = NULL;  TB.d_rowIdx = NULL;   TB.d_colIdx = NULL;
    TB.d_val    = NULL;  TB.d_mask   = NULL;
    TB.h_tilePtr = NULL; TB.h_tileColIdx = NULL; TB.h_tileNnz = NULL;
    TB.h_tileNnzPrefix = NULL;
    TB.h_rowPtr = NULL;  TB.h_rowIdx = NULL;   TB.h_colIdx = NULL;
    TB.h_val    = NULL;  TB.h_mask   = NULL;
    tiled_free(&TA);

    free(A.rowPtr); free(A.colIdx); free(A.val);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
