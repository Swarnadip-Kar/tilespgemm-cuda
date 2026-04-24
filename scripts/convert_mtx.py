#!/usr/bin/env python3
"""
convert_mtx.py
──────────────────────────────────────────────────────────────────────────────
Converts a Matrix Market (.mtx) file to a binary CSR file consumed by
row_spgemm and tile_spgemm.

Binary format (little-endian):
  int32  rows
  int32  cols
  int32  nnz
  int32[rows+1]  rowPtr   (0-based)
  int32[nnz]     colIdx   (0-based)
  float64[nnz]   val

Usage:
  python3 convert_mtx.py <input.mtx> <output.csr>

The script handles:
  - coordinate (COO) format
  - 'symmetric' and 'general' matrices
  - 1-indexed → 0-indexed conversion
  - Self-loops and explicit zeros
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import struct
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

def convert(mtx_path: str, out_path: str) -> None:
    name = os.path.basename(mtx_path)
    print(f"  [convert] Reading  {name} ...", flush=True)
    try:
        mat = mmread(mtx_path)
    except Exception as e:
        print(f"  [convert] ERROR reading {mtx_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert to CSR (handles symmetric, pattern, etc.)
    csr = csr_matrix(mat, dtype=np.float64)
    # Make sure it is square (needed for SpGEMM C = A^2)
    n = max(csr.shape)
    if csr.shape[0] != csr.shape[1]:
        print(f"  [convert] WARNING: {name} is not square ({csr.shape}), padding.", flush=True)
        from scipy.sparse import coo_matrix
        data  = csr.data
        row   = csr.nonzero()[0]
        col   = csr.nonzero()[1]
        csr   = coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float64).tocsr()

    csr.sort_indices()

    rows   = int(csr.shape[0])
    cols   = int(csr.shape[1])
    nnz    = int(csr.nnz)
    rowPtr = csr.indptr.astype(np.int32)
    colIdx = csr.indices.astype(np.int32)
    val    = csr.data.astype(np.float64)

    print(f"  [convert] {name}: {rows}x{cols}, nnz={nnz}", flush=True)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<iii', rows, cols, nnz))
        f.write(rowPtr.tobytes())
        f.write(colIdx.tobytes())
        f.write(val.tobytes())
    print(f"  [convert] Written  {out_path} ({os.path.getsize(out_path)//1024} KB)", flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 convert_mtx.py <input.mtx> <output.csr>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
