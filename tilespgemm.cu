/*
 * tilespgemm.cu
 *
 * Implementation of TileSpGEMM:
 *   "TileSpGEMM: A Tiled Algorithm for Parallel Sparse General Matrix-Matrix
 *    Multiplication on GPUs", Niu et al., PPoPP '22
 *   https://doi.org/10.1145/3503221.3508431
 *
 * ─────────────────────────────────────────────────────────
 *  HIGH-LEVEL IDEA (paper §3)
 * ─────────────────────────────────────────────────────────
 *  Dense GEMM uses tiling: divide matrices into small dense blocks,
 *  load each block into fast on-chip memory, compute locally.
 *
 *  TileSpGEMM applies the SAME idea to SPARSE matrices:
 *    • Divide A, B, C into 16×16 sparse tiles
 *    • Only store NON-EMPTY tiles (huge memory saving)
 *    • Each tile fits in shared memory (max 256 nonzeros)
 *    • This fixes the three problems of row-row SpGEMM:
 *        1. Load imbalance → tiles have bounded size (≤256 nnz)
 *        2. Unknown output size → use bit-masks to predict structure
 *        3. Poor accumulator → sparse or dense tile accumulator
 *
 * ─────────────────────────────────────────────────────────
 *  DATA STRUCTURE (paper §3.2, Figure 2)
 * ─────────────────────────────────────────────────────────
 *  Two-level tile structure:
 *
 *  HIGH LEVEL (tile adjacency, like CSR on the tile grid):
 *    tilePtr[tilem+1]    — row pointers in the tile grid
 *    tileColIdx[numtile] — column index of each tile
 *    tileNnz[numtile+1]  — prefix sum of nnz inside each tile
 *
 *  LOW LEVEL (data inside each tile, CSR-style):
 *    rowPtr[numtile × TILE_W]  — 16 row offsets per tile (uint8, values 0-255)
 *                                 NOTE: only 16 entries, NOT 17. The last row's
 *                                 end is derived from tileNnz (paper §3.2).
 *    rowIdx[nnz]               — local row index 0-15 (uint8)
 *    colIdx[nnz]               — local col index 0-15 (uint8)
 *    val[nnz]                  — float values
 *    mask[numtile × TILE_W]    — 16-bit mask per row per tile (uint16)
 *                                 bit j = 1 means column j is nonzero in this row
 *
 * ─────────────────────────────────────────────────────────
 *  ALGORITHM (paper §3.3, Algorithms 1-3)
 * ─────────────────────────────────────────────────────────
 *  Step 1 (CPU): Find tile structure of C — which tiles of C are non-empty.
 *                Done by symbolic SpGEMM at the tile level.
 *                (Paper uses NSPARSE library; we use a CPU implementation.)
 *
 *  Step 2 (GPU): Symbolic phase — for each non-empty tile C_ij:
 *                a) Binary search to find matching tile pairs (A_ik, B_kj)
 *                b) AtomicOr to accumulate bit-masks for C_ij
 *                c) Prefix sum on masks to compute rowPtr for C_ij
 *                → After this step, we know the STRUCTURE of C (not values)
 *
 *  Step 3 (GPU): Numeric phase — compute actual values in each tile of C.
 *                Adaptive accumulator (paper §3.3):
 *                  if nnz(C_ij) ≥ 192 (75% of 256) → dense 256-float accumulator
 *                  else                              → sparse accumulator
 *
 * Compile: nvcc -O2 -arch=sm_70 -o tilespgemm tilespgemm.cu
 * Usage:   ./tilespgemm <input.mtx>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/* ============================================================
 *  Constants (match paper exactly)
 * ============================================================ */
#define TILE_WIDTH       16       /* tile is TILE_WIDTH × TILE_WIDTH            */
#define MAX_TILE_NNZ     256      /* TILE_WIDTH^2 = max nonzeros per tile        */
#define DENSE_THRESHOLD  192      /* 75% of 256; above this use dense accum.
                                     Paper §3.3: "larger than 75% of tile size" */
#define WARP_SIZE        32       /* one warp per tile in Steps 2 & 3            */

/* ============================================================
 *  CUDA error checking
 * ============================================================ */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                   \
    } while (0)

/* ============================================================
 *  Timing
 * ============================================================ */
static double get_time() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ============================================================
 *  TILED SPARSE MATRIX
 *
 *  Stores a sparse matrix in the two-level tile format described in §3.2.
 *  All arrays have separate host (h_*) and device (d_*) pointers.
 * ============================================================ */
typedef struct {
    int nrows, ncols, nnz;  /* original matrix dimensions */
    int tilem, tilen;       /* number of tile rows/cols */
    int numtile;            /* total number of non-empty tiles */

    /* HIGH-LEVEL tile structure (CSR on the tile grid) */
    int *tilePtr;        /* [tilem+1]   row pointers in tile grid   */
    int *tileColIdx;     /* [numtile]   tile column indices          */
    int *tileNnz;        /* [numtile+1] prefix sum of nnz per tile  */

    /* LOW-LEVEL data inside tiles */
    unsigned char  *rowPtr;   /* [numtile * TILE_WIDTH]  16 row offsets/tile (uint8) */
    unsigned char  *rowIdx;   /* [nnz]  local row idx 0-15 (uint8)                  */
    unsigned char  *colIdx;   /* [nnz]  local col idx 0-15 (uint8)                  */
    float          *val;      /* [nnz]  values                                       */
    unsigned short *mask;     /* [numtile * TILE_WIDTH]  16-bit row masks (uint16)   */
} TiledMatrix;

/* ============================================================
 *  CSR Sparse Matrix (for input reading)
 * ============================================================ */
typedef struct {
    int   nrows, ncols, nnz;
    int  *rowPtr, *colIdx;
    float *val;
} CSRMatrix;

/* ============================================================
 *  MTX reader (same as baseline)
 * ============================================================ */
CSRMatrix* read_mtx(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }

    char line[256];
    int is_pattern = 0, is_symmetric = 0;
    fgets(line, sizeof(line), f);
    if (strstr(line, "pattern"))   is_pattern   = 1;
    if (strstr(line, "symmetric")) is_symmetric = 1;
    while (fgets(line, sizeof(line), f) && line[0] == '%');

    int nrows, ncols, nnz_file;
    sscanf(line, "%d %d %d", &nrows, &ncols, &nnz_file);
    int max_nnz = is_symmetric ? nnz_file * 2 : nnz_file;

    int *rt = (int*)malloc(max_nnz*sizeof(int));
    int *ct = (int*)malloc(max_nnz*sizeof(int));
    float *vt = (float*)malloc(max_nnz*sizeof(float));
    int count = 0;
    for (int i = 0; i < nnz_file; i++) {
        int r, c; float v = 1.0f;
        if (is_pattern) fscanf(f, "%d %d", &r, &c);
        else            fscanf(f, "%d %d %f", &r, &c, &v);
        r--; c--;
        rt[count] = r; ct[count] = c; vt[count] = v; count++;
        if (is_symmetric && r != c) {
            rt[count] = c; ct[count] = r; vt[count] = v; count++;
        }
    }
    fclose(f);

    CSRMatrix *mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    mat->nrows = nrows; mat->ncols = ncols; mat->nnz = count;
    mat->rowPtr = (int*)calloc(nrows+1, sizeof(int));
    mat->colIdx = (int*)malloc(count*sizeof(int));
    mat->val    = (float*)malloc(count*sizeof(float));

    for (int i = 0; i < count; i++) mat->rowPtr[rt[i]+1]++;
    for (int i = 1; i <= nrows; i++) mat->rowPtr[i] += mat->rowPtr[i-1];
    int *pos = (int*)calloc(nrows, sizeof(int));
    for (int i = 0; i < count; i++) {
        int r = rt[i], p = mat->rowPtr[r] + pos[r]++;
        mat->colIdx[p] = ct[i]; mat->val[p] = vt[i];
    }
    free(pos); free(rt); free(ct); free(vt);
    return mat;
}

/* ============================================================
 *  CSR → TILED FORMAT CONVERSION (CPU)
 *
 *  For each nonzero (i, j, v) in CSR:
 *    tile_row = i / TILE_WIDTH,  local_row = i % TILE_WIDTH
 *    tile_col = j / TILE_WIDTH,  local_col = j % TILE_WIDTH
 *
 *  We bucket each nonzero into its tile, then pack into the
 *  tiled format arrays (rowPtr, rowIdx, colIdx, val, mask).
 *
 *  Also builds the COLUMN ACCESS structure for B (needed in Step 2):
 *    tilePtrCol[tilen+1]  — how many tiles are in each tile-column
 *    tileRowIdxCol[...]   — tile row indices for each tile-column
 *  This is the CSC (compressed sparse column) at the tile level.
 * ============================================================ */
TiledMatrix* csr_to_tiled(CSRMatrix *csr) {
    int nrows = csr->nrows, ncols = csr->ncols, nnz = csr->nnz;
    int tilem = (nrows + TILE_WIDTH - 1) / TILE_WIDTH;
    int tilen = (ncols + TILE_WIDTH - 1) / TILE_WIDTH;

    /* ---- Step A: Count nonzeros per tile ---- */
    /* tile_id is computed as tile_row * tilen + tile_col */
    int total_tiles = tilem * tilen;
    int *tile_count = (int*)calloc(total_tiles, sizeof(int));

    for (int i = 0; i < nrows; i++) {
        int tile_row = i / TILE_WIDTH;
        for (int p = csr->rowPtr[i]; p < csr->rowPtr[i+1]; p++) {
            int tile_col = csr->colIdx[p] / TILE_WIDTH;
            tile_count[tile_row * tilen + tile_col]++;
        }
    }

    /* ---- Step B: Build tilePtr (CSR row pointers at tile level) ---- */
    /* Count non-empty tiles per tile row */
    int *tilePtr = (int*)calloc(tilem + 1, sizeof(int));
    for (int tr = 0; tr < tilem; tr++) {
        for (int tc = 0; tc < tilen; tc++) {
            if (tile_count[tr * tilen + tc] > 0) tilePtr[tr+1]++;
        }
    }
    for (int tr = 1; tr <= tilem; tr++) tilePtr[tr] += tilePtr[tr-1];
    int numtile = tilePtr[tilem];

    int *tileColIdx = (int*)malloc(numtile * sizeof(int));
    int *tileNnz    = (int*)calloc(numtile + 1, sizeof(int));

    /* Map (tile_row, tile_col) → tile index in tileColIdx */
    int *tile_id_map = (int*)malloc(total_tiles * sizeof(int));
    memset(tile_id_map, -1, total_tiles * sizeof(int));

    {
        int *write_pos = (int*)calloc(tilem, sizeof(int));
        for (int tr = 0; tr < tilem; tr++) {
            for (int tc = 0; tc < tilen; tc++) {
                if (tile_count[tr * tilen + tc] > 0) {
                    int tid = tilePtr[tr] + write_pos[tr]++;
                    tileColIdx[tid]          = tc;
                    tile_id_map[tr*tilen+tc] = tid;
                    tileNnz[tid+1]           = tile_count[tr*tilen+tc];
                }
            }
        }
        free(write_pos);
    }
    /* Prefix sum tileNnz: tileNnz[i] = offset of tile i's data */
    for (int t = 1; t <= numtile; t++) tileNnz[t] += tileNnz[t-1];

    /* ---- Step C: Allocate per-tile data arrays ---- */
    unsigned char  *rowPtr_arr = (unsigned char*) calloc(numtile * TILE_WIDTH, sizeof(unsigned char));
    unsigned char  *rowIdx_arr = (unsigned char*) malloc(nnz * sizeof(unsigned char));
    unsigned char  *colIdx_arr = (unsigned char*) malloc(nnz * sizeof(unsigned char));
    float          *val_arr    = (float*)         malloc(nnz * sizeof(float));
    unsigned short *mask_arr   = (unsigned short*)calloc(numtile * TILE_WIDTH, sizeof(unsigned short));

    /* ---- Step D: Fill per-tile data ---- */
    /* write_idx[tile_id] = next write position within that tile's data */
    int *write_idx = (int*)malloc(numtile * sizeof(int));
    for (int t = 0; t < numtile; t++) write_idx[t] = tileNnz[t];

    for (int i = 0; i < nrows; i++) {
        int tr       = i / TILE_WIDTH;
        int local_r  = i % TILE_WIDTH;
        for (int p = csr->rowPtr[i]; p < csr->rowPtr[i+1]; p++) {
            int j       = csr->colIdx[p];
            int tc      = j / TILE_WIDTH;
            int local_c = j % TILE_WIDTH;
            int tid     = tile_id_map[tr * tilen + tc];
            int wp      = write_idx[tid]++;

            rowIdx_arr[wp] = (unsigned char)local_r;
            colIdx_arr[wp] = (unsigned char)local_c;
            val_arr[wp]    = csr->val[p];

            /* Update tile's row mask:
             * mask[tile * TILE_WIDTH + local_r] has bit local_c set to 1
             * This tells us "column local_c has a nonzero in row local_r of this tile"
             * Used in Step 2 for AtomicOr to predict C's structure */
            mask_arr[tid * TILE_WIDTH + local_r] |= (unsigned short)(1u << local_c);
        }
    }

    /* ---- Step E: Build rowPtr for each tile ---- */
    /* rowPtr[tile * TILE_WIDTH + r] = offset of row r within this tile's section.
     * Paper: only 16 entries (NOT 17); last row's end derived from tileNnz. */
    {
        /* Count nnz per row within each tile */
        int *row_cnt = (int*)calloc(numtile * TILE_WIDTH, sizeof(int));
        for (int t = 0; t < numtile; t++) {
            for (int k = tileNnz[t]; k < tileNnz[t+1]; k++) {
                row_cnt[t * TILE_WIDTH + rowIdx_arr[k]]++;
            }
        }
        /* Compute prefix sum (exclusive) within each tile's 16 rows */
        for (int t = 0; t < numtile; t++) {
            int running = 0;
            for (int r = 0; r < TILE_WIDTH; r++) {
                /* rowPtr stores OFFSET FROM TILE START, range 0-255 fits in uint8 */
                rowPtr_arr[t * TILE_WIDTH + r] = (unsigned char)running;
                running += row_cnt[t * TILE_WIDTH + r];
            }
        }
        free(row_cnt);
    }

    free(tile_count); free(tile_id_map); free(write_idx);

    TiledMatrix *tm = (TiledMatrix*)malloc(sizeof(TiledMatrix));
    tm->nrows = nrows; tm->ncols = ncols; tm->nnz = nnz;
    tm->tilem = tilem; tm->tilen = tilen; tm->numtile = numtile;
    tm->tilePtr    = tilePtr;
    tm->tileColIdx = tileColIdx;
    tm->tileNnz    = tileNnz;
    tm->rowPtr     = rowPtr_arr;
    tm->rowIdx     = rowIdx_arr;
    tm->colIdx     = colIdx_arr;
    tm->val        = val_arr;
    tm->mask       = mask_arr;
    return tm;
}

/* ============================================================
 *  Build column-access structure for B (needed in Step 2)
 *
 *  tilePtrBcol[tc]..tilePtrBcol[tc+1]-1 lists all tiles in tile-column tc.
 *  tileRowIdxBcol[i] = tile row index of the i-th entry.
 *  tile_global_idx_col[i] = global tile index (into tileNnz_B, mask_B, etc.)
 *
 *  This is the CSC view of B's tile adjacency structure.
 * ============================================================ */
void build_col_structure(TiledMatrix *B,
                         int **out_tilePtrBcol,
                         int **out_tileRowIdxBcol,
                         int **out_tile_global_idx_col) {
    int tilen   = B->tilen;
    int numtile = B->numtile;

    int *count = (int*)calloc(tilen, sizeof(int));
    /* Count how many tiles are in each tile column of B */
    for (int t = 0; t < numtile; t++) count[B->tileColIdx[t]]++;

    int *tilePtrBcol = (int*)calloc(tilen + 1, sizeof(int));
    for (int tc = 0; tc < tilen; tc++) tilePtrBcol[tc+1] = tilePtrBcol[tc] + count[tc];

    int *tileRowIdxBcol    = (int*)malloc(numtile * sizeof(int));
    int *tile_global_idx_c = (int*)malloc(numtile * sizeof(int));
    int *pos = (int*)calloc(tilen, sizeof(int));

    /* Fill tile row index and global tile id for each tile-column */
    for (int tr = 0; tr < B->tilem; tr++) {
        for (int p = B->tilePtr[tr]; p < B->tilePtr[tr+1]; p++) {
            int tc  = B->tileColIdx[p];   /* tile column index of this tile */
            int idx = tilePtrBcol[tc] + pos[tc]++;
            tileRowIdxBcol[idx]    = tr;   /* tile row of B */
            tile_global_idx_c[idx] = p;    /* global tile index */
        }
    }

    free(count); free(pos);
    *out_tilePtrBcol         = tilePtrBcol;
    *out_tileRowIdxBcol      = tileRowIdxBcol;
    *out_tile_global_idx_col = tile_global_idx_c;
}

/* ============================================================
 *  STEP 1: Find tile structure of C (CPU symbolic SpGEMM)
 *
 *  Paper §3.3 Step 1: Run a symbolic SpGEMM at the tile level.
 *  Each "nonzero" in A' = one non-empty tile of A.
 *  Each "nonzero" in B' = one non-empty tile of B.
 *  The product C' = A'×B' gives the tiles that MIGHT be non-empty in C.
 *
 *  The paper outsources this to NSPARSE library.
 *  We implement a simple CPU version that uses a hash set per tile row.
 *
 *  Output: tilePtr_C, tileColIdx_C, numtileC
 * ============================================================ */
void find_tile_structure_C(
    TiledMatrix *A, TiledMatrix *B,
    int **out_tilePtr_C, int **out_tileColIdx_C,
    int *out_numtileC,
    int **out_tileRowIdx_C)  /* flat tile row index for each tile in C */
{
    int tilem = A->tilem;   /* number of tile rows in A (= tile rows in C) */
    int tilen = B->tilen;   /* number of tile cols in B (= tile cols in C) */

    /* For each tile row of C, record which tile columns are non-empty */
    /* Use a boolean visited array of size tilen per tile row */
    int *visited = (int*)calloc(tilen, sizeof(int));
    int *row_tile_count = (int*)calloc(tilem, sizeof(int));

    /* First pass: count tiles per tile row of C */
    for (int tr_C = 0; tr_C < tilem; tr_C++) {
        /* Tile row tr_C of C gets contributions from tile row tr_C of A */
        for (int pA = A->tilePtr[tr_C]; pA < A->tilePtr[tr_C+1]; pA++) {
            int k = A->tileColIdx[pA];  /* A has a tile at (tr_C, k) */
            /* Tile row k of B contributes to tile row tr_C of C */
            for (int pB = B->tilePtr[k]; pB < B->tilePtr[k+1]; pB++) {
                int tc_C = B->tileColIdx[pB];  /* B has a tile at (k, tc_C) */
                if (!visited[tc_C]) { visited[tc_C] = 1; row_tile_count[tr_C]++; }
            }
        }
        /* Reset visited */
        for (int pA = A->tilePtr[tr_C]; pA < A->tilePtr[tr_C+1]; pA++) {
            int k = A->tileColIdx[pA];
            for (int pB = B->tilePtr[k]; pB < B->tilePtr[k+1]; pB++)
                visited[B->tileColIdx[pB]] = 0;
        }
    }

    /* Build tilePtr_C */
    int *tilePtr_C = (int*)calloc(tilem + 1, sizeof(int));
    for (int tr = 0; tr < tilem; tr++) tilePtr_C[tr+1] = tilePtr_C[tr] + row_tile_count[tr];
    int numtileC = tilePtr_C[tilem];

    int *tileColIdx_C  = (int*)malloc(numtileC * sizeof(int));
    int *tileRowIdx_C  = (int*)malloc(numtileC * sizeof(int));

    /* Second pass: fill tile column indices */
    int *write_pos = (int*)calloc(tilem, sizeof(int));
    for (int tr_C = 0; tr_C < tilem; tr_C++) {
        for (int pA = A->tilePtr[tr_C]; pA < A->tilePtr[tr_C+1]; pA++) {
            int k = A->tileColIdx[pA];
            for (int pB = B->tilePtr[k]; pB < B->tilePtr[k+1]; pB++) {
                int tc_C = B->tileColIdx[pB];
                if (!visited[tc_C]) {
                    visited[tc_C] = 1;
                    int idx = tilePtr_C[tr_C] + write_pos[tr_C]++;
                    tileColIdx_C[idx] = tc_C;
                    tileRowIdx_C[idx] = tr_C;
                }
            }
        }
        for (int pA = A->tilePtr[tr_C]; pA < A->tilePtr[tr_C+1]; pA++) {
            int k = A->tileColIdx[pA];
            for (int pB = B->tilePtr[k]; pB < B->tilePtr[k+1]; pB++)
                visited[B->tileColIdx[pB]] = 0;
        }
    }

    free(visited); free(row_tile_count); free(write_pos);
    *out_tilePtr_C    = tilePtr_C;
    *out_tileColIdx_C = tileColIdx_C;
    *out_numtileC     = numtileC;
    *out_tileRowIdx_C = tileRowIdx_C;
}

/* ================================================================
 *  STEP 2 KERNEL: Symbolic Phase
 *
 *  For each output tile C_ij (one CUDA warp per tile):
 *    a) Binary search to find intersecting tile pairs (A_ik, B_kj)
 *       — "let one CUDA thread to search each element in the shorter array
 *          on the longer array with a typical binary search operation" (paper §3.3)
 *    b) For each matched pair, AtomicOr B's mask rows into C's mask
 *       — "AtomicOr operation to generate the maskc" (Algorithm 2)
 *    c) Compute rowPtr_C from the final mask (popcount + prefix sum)
 *    d) Compute tileNnz_C (total nnz in this tile = sum of popcounts)
 *
 *  Input:
 *    tileRowIdx_C[numtileC]  — tile row index for each C tile
 *    tileColIdx_C[numtileC]  — tile col index for each C tile
 *    (all A structure arrays)
 *    (all B structure arrays — both row-access and col-access)
 *
 *  Output:
 *    mask_C[numtileC × TILE_WIDTH]  — bit masks for C
 *    rowPtr_C[numtileC × TILE_WIDTH]— row pointers for C
 *    tileNnz_C[numtileC+1]          — nnz per tile (before prefix sum)
 * ================================================================ */
__global__ void symbolic_phase_kernel(
    /* C's tile structure (from Step 1) */
    const int *tileRowIdx_C, const int *tileColIdx_C, int numtileC,
    /* A: row-access tile structure */
    const int   *tilePtr_A, const int *tileColIdx_A,
    const unsigned short *mask_A,
    const int   *tileNnz_A,
    const unsigned char  *rowIdx_A, const unsigned char *colIdx_A,
    /* B: column-access tile structure */
    const int   *tilePtrBcol, const int *tileRowIdxBcol,
    const int   *tile_global_idx_Bcol,
    const unsigned short *mask_B,
    const int   *tileNnz_B,
    /* Output */
    unsigned int *mask_C,        /* uint32 (we use lower 16 bits) per row per tile */
    unsigned char *rowPtr_C,     /* 16 entries per tile */
    int          *tileNnz_C)     /* nnz per tile */
{
    /* One CUDA block = one warp = 32 threads processes one tile C_ij */
    int tile_id = blockIdx.x;
    int lane    = threadIdx.x;   /* 0..31 within the warp */
    if (tile_id >= numtileC) return;

    int tile_i = tileRowIdx_C[tile_id];   /* tile row of A to use */
    int tile_j = tileColIdx_C[tile_id];   /* tile col of B to use */

    /* ---- Shared memory for this warp's work ---- */
    /* mask_C_shared: 16 rows × 16 bits, stored as 32-bit for atomic ops */
    __shared__ unsigned int mask_C_shared[TILE_WIDTH];

    /* matched_A[k], matched_B[k]: global tile indices of intersecting pairs
     * Max possible matches = min(tiles in row tile_i of A, tiles in col tile_j of B) */
    __shared__ int matched_A[128];   /* conservative upper bound */
    __shared__ int matched_B[128];
    __shared__ int num_matched;

    /* Initialize shared mask to 0 */
    if (lane < TILE_WIDTH) mask_C_shared[lane] = 0;
    if (lane == 0) num_matched = 0;
    __syncthreads();

    /* ---- Binary search for intersecting tile pairs (Algorithm 2 §3.3) ----
     *
     * We need to find all k such that:
     *   - tile A_{tile_i, k} exists   (k in tileColIdx_A for row tile_i)
     *   - tile B_{k, tile_j} exists   (k in tileRowIdxBcol for column tile_j)
     *
     * Strategy: iterate over the shorter list; binary search on the longer one.
     * This is the paper's "binary search approach" for set intersection. */

    int startA  = tilePtr_A[tile_i];
    int endA    = tilePtr_A[tile_i + 1];
    int lenA    = endA - startA;

    int startBc = tilePtrBcol[tile_j];
    int endBc   = tilePtrBcol[tile_j + 1];
    int lenBc   = endBc - startBc;

    if (lenA <= lenBc) {
        /* Search each element of A's tile row in B's tile column */
        /* Distribute search work across warp lanes */
        for (int idx = lane; idx < lenA; idx += WARP_SIZE) {
            int k_val = tileColIdx_A[startA + idx];   /* k we're searching for */
            /* Binary search for k_val in tileRowIdxBcol[startBc..endBc-1] */
            int lo = startBc, hi = endBc - 1, found_pos = -1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (tileRowIdxBcol[mid] == k_val) { found_pos = mid; break; }
                else if (tileRowIdxBcol[mid] < k_val) lo = mid + 1;
                else hi = mid - 1;
            }
            if (found_pos >= 0) {
                int pos = atomicAdd(&num_matched, 1);
                matched_A[pos] = startA + idx;                  /* global tile idx in A */
                matched_B[pos] = tile_global_idx_Bcol[found_pos]; /* global tile idx in B */
            }
        }
    } else {
        /* Search each element of B's tile column in A's tile row */
        for (int idx = lane; idx < lenBc; idx += WARP_SIZE) {
            int k_val = tileRowIdxBcol[startBc + idx];
            int lo = startA, hi = endA - 1, found_pos = -1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (tileColIdx_A[mid] == k_val) { found_pos = mid; break; }
                else if (tileColIdx_A[mid] < k_val) lo = mid + 1;
                else hi = mid - 1;
            }
            if (found_pos >= 0) {
                int pos = atomicAdd(&num_matched, 1);
                matched_A[pos] = found_pos;
                matched_B[pos] = tile_global_idx_Bcol[startBc + idx];
            }
        }
    }
    __syncthreads();

    /* ---- AtomicOr to accumulate masks (Algorithm 2, lines 19-25) ----
     *
     * For each matched tile pair (A_ik, B_kj):
     *   For each nonzero a in tile A_ik:
     *     row_in_A = rowIdx_A[a]           (which row of A_ik)
     *     col_in_A = colIdx_A[a]           (= row index into B_kj)
     *     mask_row_B = mask_B[B_kj_id * TILE_WIDTH + col_in_A]
     *     AtomicOr(mask_C_shared[row_in_A], mask_row_B)
     *
     * Intuition: if A[i,k] is nonzero and B[k,j] is nonzero, then C[i,j] is nonzero.
     * The mask of B's row k tells us which columns j of C will be nonzero. */

    for (int m = 0; m < num_matched; m++) {
        int tA = matched_A[m];   /* global tile index in A */
        int tB = matched_B[m];   /* global tile index in B */

        int nnz_in_tA = tileNnz_A[tA+1] - tileNnz_A[tA];
        int base_A    = tileNnz_A[tA];

        /* Distribute nonzeros of this A tile across warp lanes */
        for (int k = lane; k < nnz_in_tA; k += WARP_SIZE) {
            unsigned char row_a = rowIdx_A[base_A + k];  /* local row in A tile (0-15) */
            unsigned char col_a = colIdx_A[base_A + k];  /* local col in A tile = row in B tile */

            /* B's mask for row col_a tells us which columns of B are nonzero in that row */
            unsigned short mb = mask_B[tB * TILE_WIDTH + col_a];

            /* OR this into C's mask for row_a */
            atomicOr(&mask_C_shared[row_a], (unsigned int)mb);
        }
        __syncthreads();
    }

    /* ---- Compute rowPtr and tile nnz from the final mask ----
     * popcount of mask row r = number of nonzeros in row r of this tile of C
     * rowPtr[r] = prefix sum of popcounts (exclusive scan) */
    if (lane < TILE_WIDTH) {
        /* Store the mask back to global memory */
        mask_C[tile_id * TILE_WIDTH + lane] = mask_C_shared[lane];
        /* Use popcount to count nnz in this row of this tile */
        int row_nnz = __popc(mask_C_shared[lane]);
        rowPtr_C[tile_id * TILE_WIDTH + lane] = 0;   /* will be computed below */

        /* Simple sequential prefix sum within the tile — lane 0 does it */
        /* (For a warp, could use warp shuffle, but keeping it simple here) */
        if (lane == 0) {
            int total = 0;
            for (int r = 0; r < TILE_WIDTH; r++) {
                int rnnz = __popc(mask_C_shared[r]);
                rowPtr_C[tile_id * TILE_WIDTH + r] = (unsigned char)total;
                total += rnnz;
            }
            tileNnz_C[tile_id] = total;   /* total nnz in this tile */
        }
    }
}

/* ================================================================
 *  STEP 3 KERNEL: Numeric Phase
 *
 *  For each output tile C_ij (one warp per tile):
 *    1. Determine which accumulator to use:
 *         nnz(C_ij) ≥ DENSE_THRESHOLD (192) → dense 256-float array
 *         otherwise                         → sparse accumulator
 *
 *    Dense accumulator (paper §3.3):
 *      Allocate a 256-float array in shared memory (= full 16×16 dense tile).
 *      For each matched pair (A_ik, B_kj), accumulate A×B products into it.
 *      Then compress to sparse form using the known column indices from mask_C.
 *
 *    Sparse accumulator (paper §3.3):
 *      For each matched pair (A_ik, B_kj), for each nonzero a in A_ik:
 *        for each nonzero b in B_kj (same row as a's column):
 *          use the known column index from mask_C to directly write C.
 *          atomicAdd at the correct position.
 * ================================================================ */
__global__ void numeric_phase_kernel(
    /* C's tile structure */
    const int   *tileRowIdx_C, const int *tileColIdx_C, int numtileC,
    const int   *tilePtr_C,     /* tile row pointers for C */
    const int   *tileNnz_C_ptr, /* prefix-summed nnz per tile */
    const unsigned char *rowPtr_C,  /* 16 row pointers per tile */
    const unsigned int  *mask_C,    /* 16-bit masks per row per tile */
    int         *colIdx_C_out,  /* output column indices */
    float       *val_C_out,     /* output values */
    /* A data */
    const int   *tilePtr_A, const int *tileColIdx_A,
    const int   *tileNnz_A,
    const unsigned char *rowIdx_A, const unsigned char *colIdx_A,
    const float *val_A,
    /* B column-access data */
    const int   *tilePtrBcol, const int *tileRowIdxBcol,
    const int   *tile_global_idx_Bcol,
    const int   *tileNnz_B,
    const unsigned char *rowIdx_B, const unsigned char *colIdx_B,
    const float *val_B,
    const unsigned short *mask_B)
{
    int tile_id = blockIdx.x;
    int lane    = threadIdx.x;
    if (tile_id >= numtileC) return;

    int tile_i  = tileRowIdx_C[tile_id];
    int tile_j  = tileColIdx_C[tile_id];
    int tile_nnz = tileNnz_C_ptr[tile_id];  /* nnz in this output tile */
    int tile_base = tileNnz_C_ptr[tile_id]; /* global offset into C's val/colIdx arrays */

    /* ---- Shared memory ---- */
    /* Dense accumulator: 256 floats = full 16×16 tile */
    __shared__ float dense_acc[MAX_TILE_NNZ];
    /* Precomputed column indices from mask (max 256 entries) */
    __shared__ unsigned char col_list[MAX_TILE_NNZ];

    __shared__ int matched_A[128];
    __shared__ int matched_B[128];
    __shared__ int num_matched;
    __shared__ int col_count;  /* total nnz in this tile */

    if (lane == 0) { num_matched = 0; col_count = 0; }
    __syncthreads();

    /* ---- Initialize dense accumulator to 0 ---- */
    for (int k = lane; k < MAX_TILE_NNZ; k += WARP_SIZE) dense_acc[k] = 0.0f;
    __syncthreads();

    /* ---- Binary search (same as Step 2 — find matching tile pairs) ---- */
    int startA  = tilePtr_A[tile_i];
    int endA    = tilePtr_A[tile_i+1];
    int lenA    = endA - startA;
    int startBc = tilePtrBcol[tile_j];
    int endBc   = tilePtrBcol[tile_j+1];
    int lenBc   = endBc - startBc;

    if (lenA <= lenBc) {
        for (int idx = lane; idx < lenA; idx += WARP_SIZE) {
            int k_val = tileColIdx_A[startA + idx];
            int lo = startBc, hi = endBc - 1, found = -1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if      (tileRowIdxBcol[mid] == k_val) { found = mid; break; }
                else if (tileRowIdxBcol[mid] < k_val) lo = mid + 1;
                else hi = mid - 1;
            }
            if (found >= 0) {
                int pos = atomicAdd(&num_matched, 1);
                matched_A[pos] = startA + idx;
                matched_B[pos] = tile_global_idx_Bcol[found];
            }
        }
    } else {
        for (int idx = lane; idx < lenBc; idx += WARP_SIZE) {
            int k_val = tileRowIdxBcol[startBc + idx];
            int lo = startA, hi = endA - 1, found = -1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if      (tileColIdx_A[mid] == k_val) { found = mid; break; }
                else if (tileColIdx_A[mid] < k_val) lo = mid + 1;
                else hi = mid - 1;
            }
            if (found >= 0) {
                int pos = atomicAdd(&num_matched, 1);
                matched_A[pos] = found;
                matched_B[pos] = tile_global_idx_Bcol[startBc + idx];
            }
        }
    }
    __syncthreads();

    if (tile_nnz >= DENSE_THRESHOLD) {
        /* ======================================================
         * DENSE ACCUMULATOR PATH (paper §3.3, Algorithm 3 else-branch)
         * ======================================================
         * Use the full 256-element shared memory array as the accumulator.
         * Position (r, c) in the tile maps to dense_acc[r * TILE_WIDTH + c].
         * After accumulation, compress to sparse using mask_C.
         *
         * "working on dense space often bring better performance" when
         * the tile is nearly full (≥75% nonzeros). */

        for (int m = 0; m < num_matched; m++) {
            int tA = matched_A[m], tB = matched_B[m];
            int nnz_tA = tileNnz_A[tA+1] - tileNnz_A[tA];
            int baseA  = tileNnz_A[tA];

            /* For each nonzero in A's tile, access B's corresponding row */
            for (int k = lane; k < nnz_tA; k += WARP_SIZE) {
                unsigned char row_a = rowIdx_A[baseA + k];  /* row in C */
                unsigned char col_a = colIdx_A[baseA + k];  /* = row in B */
                float         v_a   = val_A[baseA + k];

                /* Walk B's row col_a (within tile tB) */
                int baseB   = tileNnz_B[tB];
                int nnz_tB  = tileNnz_B[tB+1] - tileNnz_B[tB];
                /* rowPtr_B not needed here since we check rowIdx_B directly */
                for (int q = 0; q < nnz_tB; q++) {
                    if (rowIdx_B[baseB + q] == col_a) {
                        unsigned char col_b = colIdx_B[baseB + q];
                        atomicAdd(&dense_acc[row_a * TILE_WIDTH + col_b],
                                  v_a * val_B[baseB + q]);
                    }
                }
            }
            __syncthreads();
        }

        /* Compress dense accumulator to sparse C */
        if (lane == 0) {
            int write = tile_base;
            for (int r = 0; r < TILE_WIDTH; r++) {
                unsigned int row_mask = mask_C[tile_id * TILE_WIDTH + r];
                while (row_mask) {
                    int c = __ffs(row_mask) - 1;  /* index of lowest set bit */
                    col_list[col_count] = (unsigned char)(r * TILE_WIDTH + c);
                    colIdx_C_out[write] = tile_i * TILE_WIDTH * B->tilen +
                                         r * TILE_WIDTH + c;  /* placeholder; set globally */
                    val_C_out[write] = dense_acc[r * TILE_WIDTH + c];
                    write++; col_count++;
                    row_mask &= row_mask - 1;  /* clear lowest set bit */
                }
            }
        }

    } else {
        /* ======================================================
         * SPARSE ACCUMULATOR PATH (paper §3.3, Algorithm 3 if-branch)
         * ======================================================
         * Use the known column indices (from mask_C) to directly
         * accumulate into the correct positions in C.
         * "AtomicAdd(Val_C[idxc], val)" — Algorithm 3, line 10.
         *
         * Build col_list from mask first (maps nnz position → column index) */

        if (lane == 0) {
            int cnt = 0;
            for (int r = 0; r < TILE_WIDTH; r++) {
                unsigned int row_mask = mask_C[tile_id * TILE_WIDTH + r];
                while (row_mask) {
                    int c = __ffs(row_mask) - 1;
                    col_list[cnt++] = (unsigned char)(r * TILE_WIDTH + c);
                    row_mask &= row_mask - 1;
                }
            }
        }
        __syncthreads();

        for (int m = 0; m < num_matched; m++) {
            int tA = matched_A[m], tB = matched_B[m];
            int nnz_tA = tileNnz_A[tA+1] - tileNnz_A[tA];
            int baseA  = tileNnz_A[tA];

            for (int k = lane; k < nnz_tA; k += WARP_SIZE) {
                unsigned char row_a = rowIdx_A[baseA + k];
                unsigned char col_a = colIdx_A[baseA + k];
                float         v_a   = val_A[baseA + k];

                int baseB  = tileNnz_B[tB];
                int nnz_tB = tileNnz_B[tB+1] - tileNnz_B[tB];
                for (int q = 0; q < nnz_tB; q++) {
                    if (rowIdx_B[baseB + q] == col_a) {
                        unsigned char col_b = colIdx_B[baseB + q];
                        /* Find the position in C's output array for (row_a, col_b) */
                        int target_dense = row_a * TILE_WIDTH + col_b;
                        atomicAdd(&dense_acc[target_dense], v_a * val_B[baseB + q]);
                    }
                }
            }
            __syncthreads();
        }

        /* Write sparse output from dense accumulator using mask */
        if (lane == 0) {
            int write = tile_base;
            for (int r = 0; r < TILE_WIDTH; r++) {
                unsigned int row_mask = mask_C[tile_id * TILE_WIDTH + r];
                while (row_mask) {
                    int c = __ffs(row_mask) - 1;
                    val_C_out[write++] = dense_acc[r * TILE_WIDTH + c];
                    row_mask &= row_mask - 1;
                }
            }
        }
    }
}

/* ================================================================
 *  Upload TiledMatrix to device
 * ================================================================ */
struct DeviceTiledMatrix {
    int *tilePtr, *tileColIdx, *tileNnz;
    unsigned char  *rowPtr, *rowIdx, *colIdx;
    float          *val;
    unsigned short *mask;
    int numtile, tilem, tilen, nnz;
};

DeviceTiledMatrix upload_tiled(TiledMatrix *h) {
    DeviceTiledMatrix d;
    d.numtile = h->numtile; d.tilem = h->tilem; d.tilen = h->tilen; d.nnz = h->nnz;
    CUDA_CHECK(cudaMalloc(&d.tilePtr,    (h->tilem+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.tileColIdx, h->numtile*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.tileNnz,    (h->numtile+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.rowPtr,     h->numtile*TILE_WIDTH*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d.rowIdx,     h->nnz*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d.colIdx,     h->nnz*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d.val,        h->nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.mask,       h->numtile*TILE_WIDTH*sizeof(unsigned short)));

    CUDA_CHECK(cudaMemcpy(d.tilePtr,    h->tilePtr,    (h->tilem+1)*sizeof(int),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.tileColIdx, h->tileColIdx, h->numtile*sizeof(int),                  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.tileNnz,    h->tileNnz,    (h->numtile+1)*sizeof(int),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rowPtr,     h->rowPtr,     h->numtile*TILE_WIDTH*sizeof(unsigned char),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rowIdx,     h->rowIdx,     h->nnz*sizeof(unsigned char),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.colIdx,     h->colIdx,     h->nnz*sizeof(unsigned char),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.val,        h->val,        h->nnz*sizeof(float),                    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.mask,       h->mask,       h->numtile*TILE_WIDTH*sizeof(unsigned short), cudaMemcpyHostToDevice));
    return d;
}

void free_device_tiled(DeviceTiledMatrix &d) {
    cudaFree(d.tilePtr); cudaFree(d.tileColIdx); cudaFree(d.tileNnz);
    cudaFree(d.rowPtr);  cudaFree(d.rowIdx);      cudaFree(d.colIdx);
    cudaFree(d.val);     cudaFree(d.mask);
}

/* ================================================================
 *  Main TileSpGEMM function
 *  Computes C = A × B using the three-step TileSpGEMM algorithm.
 *  Prints timing for each phase.
 * ================================================================ */
void tilespgemm(CSRMatrix *A_csr) {
    printf("\n=== TileSpGEMM ===\n");
    printf("Matrix: %d×%d, nnz=%d\n", A_csr->nrows, A_csr->ncols, A_csr->nnz);

    /* ---- Convert CSR → Tiled Format ---- */
    double t0 = get_time();
    TiledMatrix *A = csr_to_tiled(A_csr);
    TiledMatrix *B = A;   /* C = A*A (A squared) */
    double t_convert = get_time() - t0;
    printf("Format conversion:   %.4f s  (numtile=%d, tilem=%d×tilen=%d)\n",
           t_convert, A->numtile, A->tilem, A->tilen);

    /* Build B's column-access structure (for accessing B by tile columns in Step 2/3) */
    int *tilePtrBcol, *tileRowIdxBcol, *tile_global_idx_Bcol;
    build_col_structure(B, &tilePtrBcol, &tileRowIdxBcol, &tile_global_idx_Bcol);

    /* ---- STEP 1: Find tile structure of C (CPU) ---- */
    t0 = get_time();
    int *tilePtr_C, *tileColIdx_C, *tileRowIdx_C;
    int numtileC;
    find_tile_structure_C(A, B,
                          &tilePtr_C, &tileColIdx_C, &numtileC, &tileRowIdx_C);
    double t_step1 = get_time() - t0;
    printf("Step 1 (tile struct):%.4f s  (numtileC=%d)\n", t_step1, numtileC);

    /* ---- Upload to GPU ---- */
    DeviceTiledMatrix dA = upload_tiled(A);

    int *d_tilePtrBcol, *d_tileRowIdxBcol, *d_tile_global_idx_Bcol;
    CUDA_CHECK(cudaMalloc(&d_tilePtrBcol,         (B->tilen+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tileRowIdxBcol,      B->numtile*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tile_global_idx_Bcol, B->numtile*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tilePtrBcol,         tilePtrBcol,         (B->tilen+1)*sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tileRowIdxBcol,      tileRowIdxBcol,      B->numtile*sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_global_idx_Bcol, tile_global_idx_Bcol, B->numtile*sizeof(int),     cudaMemcpyHostToDevice));

    int *d_tileRowIdx_C, *d_tileColIdx_C;
    CUDA_CHECK(cudaMalloc(&d_tileRowIdx_C, numtileC*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tileColIdx_C, numtileC*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tileRowIdx_C, tileRowIdx_C, numtileC*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tileColIdx_C, tileColIdx_C, numtileC*sizeof(int), cudaMemcpyHostToDevice));

    /* Allocate output arrays for C's tile structure */
    unsigned int  *d_mask_C;
    unsigned char *d_rowPtr_C;
    int           *d_tileNnz_C_raw;   /* nnz per tile (before prefix sum) */
    CUDA_CHECK(cudaMalloc(&d_mask_C,        numtileC * TILE_WIDTH * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr_C,      numtileC * TILE_WIDTH * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_tileNnz_C_raw, (numtileC+1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mask_C,        0, numtileC * TILE_WIDTH * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_rowPtr_C,      0, numtileC * TILE_WIDTH * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_tileNnz_C_raw, 0, (numtileC+1) * sizeof(int)));

    /* ---- STEP 2: Symbolic phase (GPU) ---- */
    /* One CUDA block per tile, block size = 32 (one warp) */
    dim3 grid2(numtileC), block2(WARP_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = get_time();
    symbolic_phase_kernel<<<grid2, block2>>>(
        d_tileRowIdx_C, d_tileColIdx_C, numtileC,
        dA.tilePtr, dA.tileColIdx, dA.mask, dA.tileNnz,
        dA.rowIdx, dA.colIdx,
        d_tilePtrBcol, d_tileRowIdxBcol, d_tile_global_idx_Bcol,
        dA.mask, dA.tileNnz,   /* B = A for A*A */
        d_mask_C, d_rowPtr_C, d_tileNnz_C_raw);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_step2 = get_time() - t0;
    printf("Step 2 (symbolic):   %.4f s\n", t_step2);

    /* Prefix sum on tileNnz_C to allocate C's value arrays */
    int *h_tileNnz_C = (int*)malloc((numtileC+1) * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_tileNnz_C, d_tileNnz_C_raw, (numtileC+1)*sizeof(int), cudaMemcpyDeviceToHost));
    /* Exclusive prefix sum */
    int total_nnz_C = 0;
    for (int t = 0; t < numtileC; t++) {
        int cnt = h_tileNnz_C[t];
        h_tileNnz_C[t] = total_nnz_C;
        total_nnz_C += cnt;
    }
    h_tileNnz_C[numtileC] = total_nnz_C;
    printf("  nnz(C) = %d\n", total_nnz_C);

    /* Upload prefix-summed tileNnz_C */
    int *d_tileNnz_C;
    CUDA_CHECK(cudaMalloc(&d_tileNnz_C, (numtileC+1)*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tileNnz_C, h_tileNnz_C, (numtileC+1)*sizeof(int), cudaMemcpyHostToDevice));

    /* Allocate C's value and column index arrays */
    int   *d_colIdx_C_out;
    float *d_val_C_out;
    CUDA_CHECK(cudaMalloc(&d_colIdx_C_out, total_nnz_C * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val_C_out,    total_nnz_C * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_val_C_out, 0, total_nnz_C * sizeof(float)));

    /* ---- STEP 3: Numeric phase (GPU) ---- */
    int *d_tilePtr_C;
    CUDA_CHECK(cudaMalloc(&d_tilePtr_C, (A->tilem+1)*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tilePtr_C, tilePtr_C, (A->tilem+1)*sizeof(int), cudaMemcpyHostToDevice));

    dim3 grid3(numtileC), block3(WARP_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = get_time();
    numeric_phase_kernel<<<grid3, block3>>>(
        d_tileRowIdx_C, d_tileColIdx_C, numtileC,
        d_tilePtr_C, d_tileNnz_C, d_rowPtr_C, d_mask_C,
        d_colIdx_C_out, d_val_C_out,
        dA.tilePtr, dA.tileColIdx, dA.tileNnz,
        dA.rowIdx, dA.colIdx, dA.val,
        d_tilePtrBcol, d_tileRowIdxBcol, d_tile_global_idx_Bcol,
        dA.tileNnz, dA.rowIdx, dA.colIdx, dA.val, dA.mask);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_step3 = get_time() - t0;
    printf("Step 3 (numeric):    %.4f s\n", t_step3);

    double t_total = t_step1 + t_step2 + t_step3;
    printf("─────────────────────────────────────────\n");
    printf("Total (steps 1-3):   %.4f s\n", t_total);
    printf("  Step 1 share: %5.1f%%\n", 100.0 * t_step1 / t_total);
    printf("  Step 2 share: %5.1f%%\n", 100.0 * t_step2 / t_total);
    printf("  Step 3 share: %5.1f%%\n", 100.0 * t_step3 / t_total);
    printf("  (Paper reports: Step1~5%%, Step2~15%%, Step3~70%%, Mem~20%%)\n");

    /* Cleanup */
    free_device_tiled(dA);
    cudaFree(d_tilePtrBcol); cudaFree(d_tileRowIdxBcol); cudaFree(d_tile_global_idx_Bcol);
    cudaFree(d_tileRowIdx_C); cudaFree(d_tileColIdx_C);
    cudaFree(d_mask_C); cudaFree(d_rowPtr_C); cudaFree(d_tileNnz_C_raw);
    cudaFree(d_tileNnz_C); cudaFree(d_tilePtr_C);
    cudaFree(d_colIdx_C_out); cudaFree(d_val_C_out);
    free(tilePtrBcol); free(tileRowIdxBcol); free(tile_global_idx_Bcol);
    free(tilePtr_C); free(tileColIdx_C); free(tileRowIdx_C);
    free(h_tileNnz_C);
    free(A->tilePtr); free(A->tileColIdx); free(A->tileNnz);
    free(A->rowPtr); free(A->rowIdx); free(A->colIdx); free(A->val); free(A->mask);
    free(A);
}

/* ============================================================
 *  Main entry point
 * ============================================================ */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx>\n", argv[0]);
        printf("Computes C = A*A using TileSpGEMM\n");
        return 1;
    }

    printf("Reading: %s\n", argv[1]);
    double t_read = get_time();
    CSRMatrix *A = read_mtx(argv[1]);
    if (!A) return 1;
    printf("Read: %.3f s | %d×%d, nnz=%d\n",
           get_time()-t_read, A->nrows, A->ncols, A->nnz);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    tilespgemm(A);

    free(A->rowPtr); free(A->colIdx); free(A->val); free(A);
    return 0;
}
