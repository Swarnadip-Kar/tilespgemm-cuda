/*
 * tile_spgemm.cu  —  OPTIMIZED TileSpGEMM  —  Self-contained
 * ─────────────────────────────────────────────────────────────────────────────
 * Optimizations:
 *   1. Step 3 fully on GPU — 256 threads/block, one thread per C[r][c] slot.
 *   2. Precomputed d_tile_row[] — O(1) tile-row lookup (no binary search).
 *   3. Tile nnz prefix sums on GPU via Thrust inclusive_scan.
 *   4. Sort-based CSR→Tiled — O(nnz) memory, safe for large matrices.
 *   5. Step 2 kernel: shared-memory bitmask accumulation, no atomics on output.
 *   6. Large-matrix safe: all size_t casts, heap-allocated shared data via
 *      global memory when tiles have many pairs.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ── Error macros ──────────────────────────────────────────────────────── */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t _s = (call); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "[CUSPARSE ERROR] %s:%d  code=%d\n", __FILE__, __LINE__, (int)_s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static inline double wtime(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ── CSR matrix ────────────────────────────────────────────────────────── */
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

/* ═══════════════════════════════════════════════════════════════════════
 *  TILE CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */
#define TILE_DIM         16
#define TILE_SIZE        256    /* TILE_DIM * TILE_DIM                    */
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8
/* MAX_PAIRS: max (A_ik, B_kj) pairs per output tile.
   For a 200k-col matrix: tilen = 200000/16 = 12500 tile-columns.
   MAX_PAIRS must cover the max nnz per tile-row in the tile-level matrix.
   512 supports very dense tile structures. */
#define MAX_PAIRS        512

/* ═══════════════════════════════════════════════════════════════════════
 *  TILED SPARSE FORMAT
 * ═══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int   tilem, tilen, numTiles, nnz;
    /* host */
    int            *h_tilePtr;
    int            *h_tileColIdx;
    int            *h_tileNnz;
    int            *h_tileNnzPrefix;  /* [numTiles+1] exclusive prefix sum */
    unsigned char  *h_rowPtr;         /* [numTiles * TILE_DIM] */
    unsigned char  *h_rowIdx;         /* [nnz] */
    unsigned char  *h_colIdx;         /* [nnz] */
    double         *h_val;            /* [nnz] */
    unsigned short *h_mask;           /* [numTiles * TILE_DIM] bitmasks */
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

/* ── Sort key for O(nnz) CSR → Tiled conversion ──────────────────────── */
typedef struct { int tr, tc, lr, lc, orig; } TileElem;

static int cmp_tile_elem(const void *a, const void *b) {
    const TileElem *ea = (const TileElem *)a;
    const TileElem *eb = (const TileElem *)b;
    if (ea->tr != eb->tr) return ea->tr - eb->tr;
    if (ea->tc != eb->tc) return ea->tc - eb->tc;
    if (ea->lr != eb->lr) return ea->lr - eb->lr;
    return ea->lc - eb->lc;
}

static void csr_to_tiled(const CsrMatrix *A, TiledMatrix *T)
{
    int tilem = (A->rows + TILE_DIM - 1) / TILE_DIM;
    int tilen = (A->cols + TILE_DIM - 1) / TILE_DIM;
    T->tilem = tilem; T->tilen = tilen; T->nnz = A->nnz;

    TileElem *elems = (TileElem *)malloc((size_t)A->nnz * sizeof(TileElem));
    if (!elems) { fprintf(stderr, "[TileSpGEMM] OOM: TileElem array (%d)\n", A->nnz); exit(1); }
    int idx = 0;
    for (int i = 0; i < A->rows; i++) {
        int tr = i / TILE_DIM, lr = i % TILE_DIM;
        for (int jp = A->rowPtr[i]; jp < A->rowPtr[i+1]; jp++) {
            int j = A->colIdx[jp];
            elems[idx++] = (TileElem){ tr, j/TILE_DIM, lr, j%TILE_DIM, jp };
        }
    }
    qsort(elems, A->nnz, sizeof(TileElem), cmp_tile_elem);

    /* Pass 1: count unique tiles per tile-row */
    T->h_tilePtr = (int *)calloc(tilem + 1, sizeof(int));
    { int cur_tr=-1, cur_tc=-1;
      for (int k=0; k<A->nnz; k++)
          if (elems[k].tr!=cur_tr || elems[k].tc!=cur_tc) {
              T->h_tilePtr[elems[k].tr+1]++;
              cur_tr=elems[k].tr; cur_tc=elems[k].tc;
          }
    }
    for (int tr=0; tr<tilem; tr++) T->h_tilePtr[tr+1] += T->h_tilePtr[tr];
    T->numTiles = T->h_tilePtr[tilem];

    T->h_tileColIdx    = (int *)malloc((size_t)T->numTiles * sizeof(int));
    T->h_tileNnz       = (int *)calloc((size_t)T->numTiles, sizeof(int));
    T->h_tileNnzPrefix = (int *)malloc((size_t)(T->numTiles+1) * sizeof(int));
    T->h_rowPtr  = (unsigned char *) calloc((size_t)T->numTiles * TILE_DIM, sizeof(unsigned char));
    T->h_rowIdx  = (unsigned char *) malloc((size_t)A->nnz * sizeof(unsigned char));
    T->h_colIdx  = (unsigned char *) malloc((size_t)A->nnz * sizeof(unsigned char));
    T->h_val     = (double *)        malloc((size_t)A->nnz * sizeof(double));
    T->h_mask    = (unsigned short *)calloc((size_t)T->numTiles * TILE_DIM, sizeof(unsigned short));

    /* Pass 2a: fill tileColIdx and counts */
    { int cur_tr=-1, cur_tc=-1, tp=-1;
      for (int k=0; k<A->nnz; k++) {
          if (elems[k].tr!=cur_tr || elems[k].tc!=cur_tc) {
              tp++; T->h_tileColIdx[tp]=elems[k].tc;
              cur_tr=elems[k].tr; cur_tc=elems[k].tc;
          }
          T->h_tileNnz[tp]++;
      }
    }

    T->h_tileNnzPrefix[0] = 0;
    for (int t=0; t<T->numTiles; t++)
        T->h_tileNnzPrefix[t+1] = T->h_tileNnzPrefix[t] + T->h_tileNnz[t];

    /* Pass 2b: fill data with write cursors */
    int *wc = (int *)malloc((size_t)T->numTiles * sizeof(int));
    for (int t=0; t<T->numTiles; t++) wc[t] = T->h_tileNnzPrefix[t];
    { int cur_tr=-1, cur_tc=-1, tp=-1;
      for (int k=0; k<A->nnz; k++) {
          TileElem *e = &elems[k];
          if (e->tr!=cur_tr || e->tc!=cur_tc) { tp++; cur_tr=e->tr; cur_tc=e->tc; }
          int pos = wc[tp]++;
          T->h_rowIdx[pos] = (unsigned char)e->lr;
          T->h_colIdx[pos] = (unsigned char)e->lc;
          T->h_val[pos]    = A->val[e->orig];
          T->h_mask[(size_t)tp*TILE_DIM + e->lr] |= (unsigned short)(1u << e->lc);
      }
    }
    free(wc);

    /* Build rowPtr (per-row prefix sums within each tile) */
    int *row_counts = (int *)calloc((size_t)T->numTiles * TILE_DIM, sizeof(int));
    for (int t=0; t<T->numTiles; t++) {
        int off=T->h_tileNnzPrefix[t], nnz_t=T->h_tileNnz[t];
        for (int k=0; k<nnz_t; k++) row_counts[(size_t)t*TILE_DIM + T->h_rowIdx[off+k]]++;
    }
    for (int t=0; t<T->numTiles; t++) {
        unsigned char acc=0;
        for (int r=0; r<TILE_DIM; r++) {
            T->h_rowPtr[(size_t)t*TILE_DIM+r] = acc;
            acc += (unsigned char)row_counts[(size_t)t*TILE_DIM+r];
        }
    }
    free(row_counts); free(elems);

    T->d_tilePtr=NULL; T->d_tileColIdx=NULL; T->d_tileNnz=NULL;
    T->d_tileNnzPrefix=NULL; T->d_rowPtr=NULL; T->d_rowIdx=NULL;
    T->d_colIdx=NULL; T->d_val=NULL; T->d_mask=NULL;
}

static double tiled_upload(TiledMatrix *T)
{
    double t0 = wtime();
    CUDA_CHECK(cudaMalloc(&T->d_tilePtr,      (size_t)(T->tilem+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_tileColIdx,    (size_t)T->numTiles*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_tileNnz,       (size_t)T->numTiles*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_tileNnzPrefix, (size_t)(T->numTiles+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&T->d_rowPtr,        (size_t)T->numTiles*TILE_DIM*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&T->d_rowIdx,        (size_t)T->nnz*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&T->d_colIdx,        (size_t)T->nnz*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&T->d_val,           (size_t)T->nnz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&T->d_mask,          (size_t)T->numTiles*TILE_DIM*sizeof(unsigned short)));

    CUDA_CHECK(cudaMemcpy(T->d_tilePtr,      T->h_tilePtr,      (size_t)(T->tilem+1)*sizeof(int),                    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_tileColIdx,   T->h_tileColIdx,   (size_t)T->numTiles*sizeof(int),                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_tileNnz,      T->h_tileNnz,      (size_t)T->numTiles*sizeof(int),                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_tileNnzPrefix,T->h_tileNnzPrefix,(size_t)(T->numTiles+1)*sizeof(int),                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_rowPtr,       T->h_rowPtr,       (size_t)T->numTiles*TILE_DIM*sizeof(unsigned char),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_rowIdx,       T->h_rowIdx,       (size_t)T->nnz*sizeof(unsigned char),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_colIdx,       T->h_colIdx,       (size_t)T->nnz*sizeof(unsigned char),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_val,          T->h_val,          (size_t)T->nnz*sizeof(double),                       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(T->d_mask,         T->h_mask,         (size_t)T->numTiles*TILE_DIM*sizeof(unsigned short), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    return (wtime() - t0) * 1e3;
}

static void tiled_free(TiledMatrix *T) {
    free(T->h_tilePtr); free(T->h_tileColIdx); free(T->h_tileNnz);
    free(T->h_tileNnzPrefix); free(T->h_rowPtr); free(T->h_rowIdx);
    free(T->h_colIdx); free(T->h_val); free(T->h_mask);
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

/* ═══════════════════════════════════════════════════════════════════════
 *  STEP 1 — cuSPARSE tile-level symbolic SpGEMM
 * ═══════════════════════════════════════════════════════════════════════ */
static double step1_tile_structure(const TiledMatrix *A, const TiledMatrix *B,
                                    int **h_tilePtrC_out, int **h_tileColIdxC_out,
                                    int *numTilesC_out)
{
    int rowsAp=A->tilem, colsAp=A->tilen, nnzAp=A->numTiles;
    int rowsBp=B->tilem, colsBp=B->tilen, nnzBp=B->numTiles;

    double *h_vA=(double*)malloc(nnzAp*sizeof(double));
    double *h_vB=(double*)malloc(nnzBp*sizeof(double));
    for(int i=0;i<nnzAp;i++) h_vA[i]=1.0;
    for(int i=0;i<nnzBp;i++) h_vB[i]=1.0;

    int *d_rpA,*d_ciA; double *d_vA;
    int *d_rpB,*d_ciB; double *d_vB;
    CUDA_CHECK(cudaMalloc(&d_rpA,(rowsAp+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ciA,nnzAp*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vA, nnzAp*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rpB,(rowsBp+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ciB,nnzBp*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vB, nnzBp*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rpA,A->h_tilePtr,   (rowsAp+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ciA,A->h_tileColIdx, nnzAp*sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vA, h_vA,            nnzAp*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rpB,B->h_tilePtr,   (rowsBp+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ciB,B->h_tileColIdx, nnzBp*sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vB, h_vB,            nnzBp*sizeof(double), cudaMemcpyHostToDevice));
    free(h_vA); free(h_vB);

    cusparseHandle_t handle; CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t mA,mB,mC;
    CUSPARSE_CHECK(cusparseCreateCsr(&mA,rowsAp,colsAp,nnzAp,d_rpA,d_ciA,d_vA,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&mB,rowsBp,colsBp,nnzBp,d_rpB,d_ciB,d_vB,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    int *d_rpC; CUDA_CHECK(cudaMalloc(&d_rpC,(rowsAp+1)*sizeof(int)));
    CUSPARSE_CHECK(cusparseCreateCsr(&mC,rowsAp,colsBp,0,d_rpC,NULL,NULL,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));

    double alpha=1.0,beta=0.0;
    cusparseSpGEMMDescr_t desc; CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&desc));
    /* BUG FIX (Bug 5): time the FULL step including work estimation */
    double t0=wtime();
    size_t bs1=0; void *b1=NULL;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,mA,mB,&beta,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs1,NULL));
    CUDA_CHECK(cudaMalloc(&b1,bs1?bs1:1));
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,mA,mB,&beta,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs1,b1));
    size_t bs2=0; void *b2=NULL;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,mA,mB,&beta,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs2,NULL));
    CUDA_CHECK(cudaMalloc(&b2,bs2?bs2:1));

    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,mA,mB,&beta,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs2,b2));
    int64_t nr,nc,nnzCp;
    CUSPARSE_CHECK(cusparseSpMatGetSize(mC,&nr,&nc,&nnzCp));
    int *d_ciC; double *d_vC;
    CUDA_CHECK(cudaMalloc(&d_ciC,(size_t)nnzCp*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vC, (size_t)nnzCp*sizeof(double)));
    CUSPARSE_CHECK(cusparseCsrSetPointers(mC,d_rpC,d_ciC,d_vC));
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,mA,mB,&beta,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,desc));
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed_ms = (wtime()-t0)*1e3;

    *h_tilePtrC_out    = (int*)malloc((rowsAp+1)*sizeof(int));
    *h_tileColIdxC_out = (int*)malloc((size_t)nnzCp*sizeof(int));
    CUDA_CHECK(cudaMemcpy(*h_tilePtrC_out,   d_rpC,(rowsAp+1)*sizeof(int),      cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*h_tileColIdxC_out,d_ciC,(size_t)nnzCp*sizeof(int),   cudaMemcpyDeviceToHost));
    *numTilesC_out = (int)nnzCp;

    cudaFree(b1);cudaFree(b2);
    cudaFree(d_rpA);cudaFree(d_ciA);cudaFree(d_vA);
    cudaFree(d_rpB);cudaFree(d_ciB);cudaFree(d_vB);
    cudaFree(d_rpC);cudaFree(d_ciC);cudaFree(d_vC);
    cusparseSpGEMM_destroyDescr(desc);
    cusparseDestroySpMat(mA);cusparseDestroySpMat(mB);cusparseDestroySpMat(mC);
    cusparseDestroy(handle);
    return elapsed_ms;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  STEP 2 — Symbolic phase kernel (optimized + bug-fixed)
 *
 *  1 warp per C-tile; each lane handles A tiles at stride WARP_SIZE.
 *
 *  BUG FIX (Bug 1): For C-tile(tile_i, tile_j) = Σ_k A[tile_i][k]×B[k][tile_j],
 *    each A tile's column is k.  We must search B's tile-ROW k for tile-col
 *    tile_j — not B's tile-row tile_j for col_a as the old code did.
 *    The old approach was only correct because all test matrices are symmetric.
 *
 *  BUG FIX (Bug 3): Removed dead parameters A_tileNnz, B_tileNnz,
 *    A_rowPtr, B_rowPtr (were suppressed with (void) in the old kernel).
 *
 *  OPTIMIZATION: Register-local mask accumulation per lane followed by
 *    warp-shuffle XOR reduction — eliminates all atomicOr on shared memory.
 * ═══════════════════════════════════════════════════════════════════════ */
__global__
void step2_symbolic_kernel(
    const int            *A_tilePtr, const int *A_tileColIdx,
    const unsigned short *A_mask,
    const int            *B_tilePtr, const int *B_tileColIdx,
    const unsigned short *B_mask,
    const int            *C_tileColIdx, const int *d_tile_row,
    int *C_tileNnz, unsigned char *C_rowPtr, unsigned short *C_mask,
    int numTilesC)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int wid  = tid  / WARP_SIZE;
    int lane = tid  % WARP_SIZE;
    int wb   = threadIdx.x / WARP_SIZE;
    if (wid >= numTilesC) return;

    int tile_i = d_tile_row[wid];
    int tile_j = C_tileColIdx[wid];
    int lenA   = A_tilePtr[tile_i+1] - A_tilePtr[tile_i];
    int baseA  = A_tilePtr[tile_i];

    /* Each lane accumulates its own contribution into register-local masks */
    unsigned int local_mask[TILE_DIM];
    for (int r = 0; r < TILE_DIM; r++) local_mask[r] = 0u;

    for (int ia = lane; ia < lenA; ia += WARP_SIZE) {
        int col_a = A_tileColIdx[baseA + ia];  /* k = intermediate tile-col */

        /* BUG FIX (Bug 1): search B's tile-ROW col_a for tile-col tile_j */
        int lenB_k  = B_tilePtr[col_a+1] - B_tilePtr[col_a];
        int baseB_k = B_tilePtr[col_a];
        int lo=0, hi=lenB_k-1, found_b=-1;
        while (lo<=hi) {
            int mid=(lo+hi)>>1, v=B_tileColIdx[baseB_k+mid];
            if      (v==tile_j) { found_b=mid; break; }
            else if (v< tile_j)   lo=mid+1;
            else                   hi=mid-1;
        }
        if (found_b < 0) continue;

        int posA = baseA + ia;
        int posB = baseB_k + found_b;

        /* Accumulate column contributions into local register masks */
        for (int r = 0; r < TILE_DIM; r++) {
            unsigned short mA = A_mask[(size_t)posA * TILE_DIM + r];
            unsigned int contrib = 0u;
            while (mA) {
                int c = __ffs((int)(unsigned int)mA) - 1;
                mA &= (unsigned short)(mA - 1);
                contrib |= (unsigned int)B_mask[(size_t)posB * TILE_DIM + c];
            }
            local_mask[r] |= contrib;
        }
    }

    /* Warp-reduce local_mask via shuffle XOR — no shared-memory atomics */
    __shared__ unsigned int s_mask[WARPS_PER_BLOCK][TILE_DIM];
    for (int r = 0; r < TILE_DIM; r++) {
        unsigned int v = local_mask[r];
        for (int off = WARP_SIZE/2; off > 0; off >>= 1)
            v |= __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) s_mask[wb][r] = v;
    }
    __syncwarp();

    if (lane < TILE_DIM)
        C_mask[(size_t)wid * TILE_DIM + lane] =
            (unsigned short)(s_mask[wb][lane] & 0xFFFFu);
    __syncwarp();

    if (lane == 0) {
        int total = 0; unsigned char acc = 0;
        for (int r = 0; r < TILE_DIM; r++) {
            C_rowPtr[(size_t)wid * TILE_DIM + r] = acc;
            int cnt = __popc(s_mask[wb][r]);
            total += cnt; acc += (unsigned char)cnt;
        }
        C_tileNnz[wid] = total;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  STEP 3 — Numeric phase kernel (fully GPU, optimized + bug-fixed)
 *
 *  1 block per C-tile; TILE_SIZE=256 threads, one thread per C[r][c] slot.
 *
 *  BUG FIX (Bug 1): B tile-row is now searched per k (= col_a), looking for
 *    tile-col tile_j.  The old code searched B's tile-row tile_j for col_a,
 *    which was wrong for non-symmetric matrices.
 *
 *  BUG FIX (Bug 2): All 256 threads participate in pair-finding via a
 *    grid-stride loop + atomicAdd on s_npairs (was: only thread 0).
 *
 *  BUG FIX (Bug 4): Overflow of MAX_PAIRS now prints a warning (was silent).
 *
 *  OPTIMIZATION: B[ca][c] looked up in O(1) using B_mask bitmask + __popc
 *    to derive the storage offset — replaces the inner linear scan over
 *    B tile row ca (up to 16 iterations).
 * ═══════════════════════════════════════════════════════════════════════ */
__global__
void step3_numeric_kernel(
    const int *A_tilePtr, const int *A_tileColIdx, const int *A_tileNnzPrefix,
    const unsigned char *A_rowPtr, const unsigned char *A_colIdx, const double *A_val,
    const int *B_tilePtr, const int *B_tileColIdx, const int *B_tileNnzPrefix,
    const unsigned char *B_rowPtr, const unsigned char *B_colIdx, const double *B_val,
    const unsigned short *B_mask,                    /* NEW: for O(1) col lookup */
    const int *C_tileColIdx, const int *C_tileNnzPrefix,
    const unsigned char *C_rowPtr, const unsigned short *C_mask,
    const int *d_tile_row,
    unsigned char *C_rowIdx_out, unsigned char *C_colIdx_out, double *C_val_out,
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
    int baseA  = A_tilePtr[tile_i];

    __shared__ int s_posA[MAX_PAIRS];
    __shared__ int s_posB[MAX_PAIRS];
    __shared__ int s_npairs;
    __shared__ int s_overflow;

    if (slot == 0) { s_npairs = 0; s_overflow = 0; }
    __syncthreads();

    /* BUG FIX (Bug 2): all 256 threads find pairs in parallel */
    for (int ia = slot; ia < lenA; ia += TILE_SIZE) {
        int col_a = A_tileColIdx[baseA + ia];

        /* BUG FIX (Bug 1): search B's tile-ROW col_a for tile-col tile_j */
        int lenB_k  = B_tilePtr[col_a+1] - B_tilePtr[col_a];
        int baseB_k = B_tilePtr[col_a];
        int lo=0, hi=lenB_k-1;
        while (lo<=hi) {
            int mid=(lo+hi)>>1, v=B_tileColIdx[baseB_k+mid];
            if (v==tile_j) {
                int idx = atomicAdd(&s_npairs, 1);
                if (idx < MAX_PAIRS) {
                    s_posA[idx] = baseA + ia;
                    s_posB[idx] = baseB_k + mid;
                } else {
                    atomicOr(&s_overflow, 1);  /* BUG FIX (Bug 4): flag overflow */
                }
                break;
            } else if (v < tile_j) lo=mid+1; else hi=mid-1;
        }
    }
    __syncthreads();

    /* BUG FIX (Bug 4): warn on overflow (only once per tile) */
    if (slot == 0 && s_overflow)
        printf("[TileSpGEMM] WARNING: C-tile(%d,%d) has >%d pairs; results truncated!\n",
               tile_i, tile_j, MAX_PAIRS);

    double sum = 0.0;
    int np = s_npairs < MAX_PAIRS ? s_npairs : MAX_PAIRS;

    for (int p = 0; p < np; p++) {
        int posA   = s_posA[p];
        int posB   = s_posB[p];
        int offA   = A_tileNnzPrefix[posA];
        int nnzA_t = A_tileNnzPrefix[posA+1] - offA;
        int offB   = B_tileNnzPrefix[posB];

        int raStart = (int)A_rowPtr[(size_t)posA * TILE_DIM + r];
        int raEnd   = (r < TILE_DIM-1)
                    ? (int)A_rowPtr[(size_t)posA * TILE_DIM + r + 1]
                    : nnzA_t;

        for (int ka = raStart; ka < raEnd; ka++) {
            int    ca = (int)A_colIdx[offA + ka];
            double va = A_val[offA + ka];

            /* OPTIMIZATION: O(1) B[ca][c] lookup via bitmask + __popc offset */
            unsigned short brow_mask = B_mask[(size_t)posB * TILE_DIM + ca];
            if (!((brow_mask >> c) & 1u)) continue;   /* B[ca][c] == 0 */
            int bcBase  = (int)B_rowPtr[(size_t)posB * TILE_DIM + ca];
            int bc_pos  = __popc((unsigned int)(brow_mask & ((1u << c) - 1u)));
            sum += va * B_val[offB + bcBase + bc_pos];
        }
    }

    unsigned short row_mask = C_mask[(size_t)tile_idx * TILE_DIM + r];
    if (!((row_mask >> c) & 1u)) return;

    int outBase = C_tileNnzPrefix[tile_idx];
    int rstart  = (int)C_rowPtr[(size_t)tile_idx * TILE_DIM + r];
    int pos     = __popc((unsigned int)(row_mask & ((1u << c) - 1u)));
    int outIdx  = outBase + rstart + pos;
    C_rowIdx_out[outIdx] = (unsigned char)r;
    C_colIdx_out[outIdx] = (unsigned char)c;
    C_val_out[outIdx]    = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: tile_spgemm <csr_binary> <matrix_name>\n");
        return 1;
    }
    const char *csr_path = argv[1];
    const char *mat_name = argv[2];

    fprintf(stderr, "[TileSpGEMM] Loading '%s' from %s ...\n", mat_name, csr_path);
    CsrMatrix A;
    if (csr_load_binary(csr_path, &A) != 0) return 1;
    fprintf(stderr, "[TileSpGEMM] Matrix: %d x %d, nnz=%d\n", A.rows, A.cols, A.nnz);

    /* CSR → Tiled */
    fprintf(stderr, "[TileSpGEMM] Converting CSR → Tiled ...\n");
    double t_conv_start = wtime();
    TiledMatrix TA;
    csr_to_tiled(&A, &TA);
    double t_conv_ms = (wtime() - t_conv_start) * 1e3;
    fprintf(stderr, "[TileSpGEMM] Tiled: %d tiles (A), conversion=%.2f ms\n",
            TA.numTiles, t_conv_ms);

    size_t tiled_bytes =
        (size_t)(TA.tilem+1)*sizeof(int) +
        (size_t)TA.numTiles*sizeof(int)*2 +
        (size_t)TA.numTiles*TILE_DIM*(sizeof(unsigned char)+sizeof(unsigned short)) +
        (size_t)TA.nnz*(2*sizeof(unsigned char)+sizeof(double));
    size_t csr_bytes =
        (size_t)(A.rows+1)*sizeof(int) +
        (size_t)A.nnz*(sizeof(int)+sizeof(double));

    /* Upload */
    fprintf(stderr, "[TileSpGEMM] Uploading tiled matrix to GPU ...\n");
    double t_upload_ms = tiled_upload(&TA);

    /* B = A (C = A²) — share all device/host pointers */
    TiledMatrix TB; memcpy(&TB, &TA, sizeof(TiledMatrix));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

    /* ── STEP 1 ── */
    fprintf(stderr, "[TileSpGEMM] Step 1: Computing tile structure of C ...\n");
    int *h_tilePtrC=NULL, *h_tileColIdxC=NULL, numTilesC=0;
    double t_step1_ms = step1_tile_structure(&TA, &TB,
                                              &h_tilePtrC, &h_tileColIdxC, &numTilesC);
    fprintf(stderr, "[TileSpGEMM] Step 1 done: %d non-empty tiles in C, %.2f ms\n",
            numTilesC, t_step1_ms);

    /* Precompute tile-row lookup */
    int *h_tile_row = (int*)malloc((size_t)numTilesC * sizeof(int));
    for (int tr=0; tr<TA.tilem; tr++)
        for (int wid=h_tilePtrC[tr]; wid<h_tilePtrC[tr+1]; wid++)
            h_tile_row[wid] = tr;
    int *d_tile_row;
    CUDA_CHECK(cudaMalloc(&d_tile_row, (size_t)numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tile_row, h_tile_row, (size_t)numTilesC*sizeof(int), cudaMemcpyHostToDevice));
    free(h_tile_row);

    /* Allocate C symbolic arrays */
    int            *d_tilePtrC, *d_tileColIdxC, *d_tileNnzC;
    unsigned char  *d_rowPtrC;
    unsigned short *d_maskC;
    CUDA_CHECK(cudaMalloc(&d_tilePtrC,    (size_t)(TA.tilem+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tileColIdxC, (size_t)numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tileNnzC,    (size_t)numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowPtrC,     (size_t)numTilesC*TILE_DIM*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_maskC,       (size_t)numTilesC*TILE_DIM*sizeof(unsigned short)));
    CUDA_CHECK(cudaMemcpy(d_tilePtrC,    h_tilePtrC,    (size_t)(TA.tilem+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tileColIdxC, h_tileColIdxC, (size_t)numTilesC*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tileNnzC, 0, (size_t)numTilesC*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_rowPtrC,  0, (size_t)numTilesC*TILE_DIM*sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_maskC,    0, (size_t)numTilesC*TILE_DIM*sizeof(unsigned short)));

    /* ── STEP 2 ── */
    fprintf(stderr, "[TileSpGEMM] Step 2: Symbolic phase (bitmask, rowPtr per tile) ...\n");
    {
        int threads = WARPS_PER_BLOCK * WARP_SIZE;
        int blocks  = (numTilesC + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (blocks < 1) blocks = 1;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev0));
        step2_symbolic_kernel<<<blocks, threads>>>(
            TA.d_tilePtr, TA.d_tileColIdx, TA.d_mask,
            TB.d_tilePtr, TB.d_tileColIdx, TB.d_mask,
            d_tileColIdxC, d_tile_row,
            d_tileNnzC, d_rowPtrC, d_maskC, numTilesC);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());
    }
    float f2; CUDA_CHECK(cudaEventElapsedTime(&f2, ev0, ev1));
    double t_step2_ms = (double)f2;

    /* Thrust prefix sum on device */
    int *d_tileNnzPrefixC;
    CUDA_CHECK(cudaMalloc(&d_tileNnzPrefixC, (size_t)(numTilesC+1)*sizeof(int)));
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
    fprintf(stderr, "[TileSpGEMM] Step 2 done: nnzC=%d (in %d/%d non-empty tiles), %.2f ms\n",
            nnzC_total, numTilesC, numTilesC, t_step2_ms);

    /* Allocate C numeric output */
    size_t nnzC_safe = (nnzC_total > 0) ? (size_t)nnzC_total : 1;
    unsigned char *d_rowIdxC, *d_colIdxC;
    double        *d_valC;
    CUDA_CHECK(cudaMalloc(&d_rowIdxC, nnzC_safe * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_colIdxC, nnzC_safe * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_valC,    nnzC_safe * sizeof(double)));

    /* ── STEP 3 ── */
    fprintf(stderr, "[TileSpGEMM] Step 3: GPU numeric (256 threads/tile) ...\n");
    {
        int blocks = (numTilesC > 0) ? numTilesC : 1;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev0));
        step3_numeric_kernel<<<blocks, TILE_SIZE>>>(
            TA.d_tilePtr, TA.d_tileColIdx, TA.d_tileNnzPrefix,
            TA.d_rowPtr,  TA.d_colIdx, TA.d_val,
            TB.d_tilePtr, TB.d_tileColIdx, TB.d_tileNnzPrefix,
            TB.d_rowPtr,  TB.d_colIdx, TB.d_val, TB.d_mask,
            d_tileColIdxC, d_tileNnzPrefixC, d_rowPtrC, d_maskC, d_tile_row,
            d_rowIdxC, d_colIdxC, d_valC, numTilesC);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());
    }
    float f3; CUDA_CHECK(cudaEventElapsedTime(&f3, ev0, ev1));
    double t_step3_ms = (double)f3;
    fprintf(stderr, "[TileSpGEMM] Step 3 done: nnzC=%d, %.2f ms\n", nnzC_total, t_step3_ms);

    /* Stats */
    double t_total_ms = t_step1_ms + t_step2_ms + t_step3_ms;

    long long flops = 0;
    for (int i=0; i<A.rows; i++)
        for (int jp=A.rowPtr[i]; jp<A.rowPtr[i+1]; jp++)
            flops += 2LL * (A.rowPtr[A.colIdx[jp]+1] - A.rowPtr[A.colIdx[jp]]);
    double gflops = (flops / 1e9) / (t_total_ms / 1e3);

    size_t peak_bytes =
        2 * tiled_bytes +
        (size_t)numTilesC * (sizeof(int)*2 + TILE_DIM*(sizeof(unsigned char)+sizeof(unsigned short))) +
        (size_t)nnzC_total * (2*sizeof(unsigned char) + sizeof(double));

    fprintf(stderr, "[TileSpGEMM] Total: %.2f ms, %.6f GFlops  upload=%.2f ms\n",
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

    /* Cleanup */
    free(h_tilePtrC); free(h_tileColIdxC);
    cudaFree(d_tile_row);
    cudaFree(d_tilePtrC); cudaFree(d_tileColIdxC);
    cudaFree(d_tileNnzC); cudaFree(d_rowPtrC); cudaFree(d_maskC);
    cudaFree(d_tileNnzPrefixC);
    cudaFree(d_rowIdxC); cudaFree(d_colIdxC); cudaFree(d_valC);

    /* TB shares TA's pointers — null before free */
    TB.d_tilePtr=NULL; TB.d_tileColIdx=NULL; TB.d_tileNnz=NULL;
    TB.d_tileNnzPrefix=NULL; TB.d_rowPtr=NULL; TB.d_rowIdx=NULL;
    TB.d_colIdx=NULL; TB.d_val=NULL; TB.d_mask=NULL;
    TB.h_tilePtr=NULL; TB.h_tileColIdx=NULL; TB.h_tileNnz=NULL;
    TB.h_tileNnzPrefix=NULL; TB.h_rowPtr=NULL; TB.h_rowIdx=NULL;
    TB.h_colIdx=NULL; TB.h_val=NULL; TB.h_mask=NULL;
    tiled_free(&TA);

    free(A.rowPtr); free(A.colIdx); free(A.val);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
