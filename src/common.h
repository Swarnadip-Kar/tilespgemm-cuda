#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <sys/time.h>

/* ─── Error checking macros ─────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                        \
        cusparseStatus_t st = (call);                                           \
        if (st != CUSPARSE_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[CUSPARSE ERROR] %s:%d  %d\n",                    \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ─── Timing helper ─────────────────────────────────────────────────────── */
static inline double wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ─── CSR matrix ────────────────────────────────────────────────────────── */
typedef struct {
    int    rows, cols, nnz;
    int   *rowPtr;   /* host */
    int   *colIdx;   /* host */
    double *val;     /* host */
    /* device mirrors */
    int   *d_rowPtr;
    int   *d_colIdx;
    double *d_val;
} CsrMatrix;

void csr_alloc_device(CsrMatrix *A) {
    CUDA_CHECK(cudaMalloc(&A->d_rowPtr, (A->rows+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A->d_colIdx, A->nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A->d_val,    A->nnz*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A->d_rowPtr, A->rowPtr, (A->rows+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A->d_colIdx, A->colIdx, A->nnz*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A->d_val,    A->val,    A->nnz*sizeof(double),     cudaMemcpyHostToDevice));
}

void csr_free(CsrMatrix *A) {
    free(A->rowPtr); free(A->colIdx); free(A->val);
    if (A->d_rowPtr) cudaFree(A->d_rowPtr);
    if (A->d_colIdx) cudaFree(A->d_colIdx);
    if (A->d_val)    cudaFree(A->d_val);
}

/* ─── Load binary CSR written by convert_mtx.py ─────────────────────────── */
/* Format: rows(int) cols(int) nnz(int) rowPtr[rows+1] colIdx[nnz] val[nnz] */
int csr_load_binary(const char *path, CsrMatrix *A) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fread(&A->rows, sizeof(int), 1, fp);
    fread(&A->cols, sizeof(int), 1, fp);
    fread(&A->nnz,  sizeof(int), 1, fp);
    A->rowPtr = (int*)   malloc((A->rows+1)*sizeof(int));
    A->colIdx = (int*)   malloc(A->nnz*sizeof(int));
    A->val    = (double*)malloc(A->nnz*sizeof(double));
    fread(A->rowPtr, sizeof(int),    A->rows+1, fp);
    fread(A->colIdx, sizeof(int),    A->nnz,    fp);
    fread(A->val,    sizeof(double), A->nnz,    fp);
    fclose(fp);
    A->d_rowPtr = NULL; A->d_colIdx = NULL; A->d_val = NULL;
    return 0;
}
