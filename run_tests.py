#!/usr/bin/env python3
"""
run_tests.py

Test script for TileSpGEMM and baseline SpGEMM.

Workflow:
  1. Check if compiled binaries exist; compile if not
  2. For each test matrix:
       a. Check if already downloaded locally
       b. If not, download from SuiteSparse Matrix Collection
       c. Extract .tar.gz
       d. Run both algorithms, capture timing per phase
  3. Print comparison table

Matrices used: subset from Table 2 of the TileSpGEMM paper (PPoPP '22)
SuiteSparse download: https://sparse.tamu.edu/MM/{group}/{name}.tar.gz

Usage:
  python3 run_tests.py               # download and test all matrices
  python3 run_tests.py --generate    # use generated random matrices (no download needed)
  python3 run_tests.py --skip-build  # skip recompilation
"""

import os
import sys
import subprocess
import tarfile
import urllib.request
import urllib.error
import time
import re
import argparse
import scipy.sparse as sp
import numpy as np


# ──────────────────────────────────────────────────────────────
#  Test matrix definitions
#  Format: (group, name, description from paper Table 2)
#  Source: https://sparse.tamu.edu
# ──────────────────────────────────────────────────────────────
PAPER_MATRICES = [
    # (group,         name,                 description)
    ("Williams",      "cant",               "62K×62K, 4.0M nnz — FEM, structural"),
    ("Williams",      "pdb1HYS",            "36K×36K, 4.3M nnz — protein structure"),
    ("Williams",      "mac_econ_fwd500",    "206K×206K, 1.3M nnz — economic model"),
    ("Hamm",          "mc2depi",            "525K×525K, 2.1M nnz — epidemiology"),
    ("Rothberg",      "pwtk",               "218K×218K, 11.6M nnz — airplane wing"),
    ("Williams",      "cop20k_A",           "121K×121K, 2.6M nnz — CFD"),
]

# Smaller matrices for quick testing (download faster)
QUICK_MATRICES = [
    ("HB",            "bcsstk30",           "28K×28K, 1.0M nnz — structural"),
    ("Williams",      "mac_econ_fwd500",    "206K×206K, 1.3M nnz — economic"),
    ("Hamm",          "mc2depi",            "525K×525K, 2.1M nnz — epidemiology"),
]

DATASET_DIR = "datasets"
SUITESPARSE_URL = "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"
SUITESPARSE_URL2 = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


# ──────────────────────────────────────────────────────────────
#  Compilation
# ──────────────────────────────────────────────────────────────
def compile_programs():
    """Compile both CUDA programs using make."""
    print("Compiling CUDA programs...")
    result = subprocess.run(["make", "all"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Build FAILED:")
        print(result.stderr)
        # Try to get helpful error info
        if "nvcc" in result.stderr.lower() and "not found" in result.stderr.lower():
            print("\n[HINT] CUDA toolkit not found. Install it or load the module:")
            print("  module load cuda  (on HPC clusters)")
            print("  https://developer.nvidia.com/cuda-downloads")
        sys.exit(1)
    print("Build OK.\n")


def check_binaries():
    """Return True if both binaries exist."""
    return os.path.exists("spgemm_baseline") and os.path.exists("tilespgemm")


# ──────────────────────────────────────────────────────────────
#  Dataset management
# ──────────────────────────────────────────────────────────────
def find_mtx(group, name):
    """Search for the .mtx file in the dataset directory."""
    # Common path patterns after extraction
    candidates = [
        os.path.join(DATASET_DIR, name, f"{name}.mtx"),
        os.path.join(DATASET_DIR, f"{name}.mtx"),
        os.path.join(DATASET_DIR, group, name, f"{name}.mtx"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def download_matrix(group, name):
    """
    Download a matrix from SuiteSparse.
    Returns path to the .mtx file, or None on failure.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Check if already downloaded
    mtx_path = find_mtx(group, name)
    if mtx_path:
        print(f"  [CACHED] {name}: {mtx_path}")
        return mtx_path

    # Try both download URLs
    tgz_path = os.path.join(DATASET_DIR, f"{name}.tar.gz")
    downloaded = False

    for url_template in [SUITESPARSE_URL, SUITESPARSE_URL2]:
        url = url_template.format(group=group, name=name)
        print(f"  Downloading {name} from {url}")
        try:
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 / total_size)
                    print(f"\r    {pct:.0f}%  ({block_num*block_size/1e6:.1f} / {total_size/1e6:.1f} MB)", end="")

            urllib.request.urlretrieve(url, tgz_path, reporthook=progress_hook)
            print()  # newline after progress
            downloaded = True
            break
        except urllib.error.URLError as e:
            print(f"\n    Failed ({e}), trying alternate URL...")
        except Exception as e:
            print(f"\n    Error: {e}")

    if not downloaded:
        print(f"  [SKIP] Could not download {name}")
        return None

    # Extract the tarball
    print(f"  Extracting {name}.tar.gz ...")
    try:
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(DATASET_DIR)
        os.remove(tgz_path)
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return None

    mtx_path = find_mtx(group, name)
    if mtx_path:
        print(f"  Extracted to: {mtx_path}")
    return mtx_path


# ──────────────────────────────────────────────────────────────
#  Random matrix generator (fallback, no download needed)
# ──────────────────────────────────────────────────────────────
def generate_random_mtx(n, density, name):
    """
    Generate a random sparse symmetric matrix and save as MTX.
    Useful for testing without downloading datasets.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    path = os.path.join(DATASET_DIR, f"{name}.mtx")
    if os.path.exists(path):
        return path

    print(f"  Generating random {n}×{n} matrix (density={density:.4f}) ...")
    nnz_per_row = max(1, int(n * density))
    rows, cols, vals = [], [], []
    for i in range(n):
        # Random nonzeros per row
        js = np.random.choice(n, size=min(nnz_per_row, n), replace=False)
        for j in js:
            rows.append(i+1); cols.append(j+1)  # 1-indexed for MTX
            vals.append(np.random.uniform(0.1, 2.0))
            if i != j:  # symmetric
                rows.append(j+1); cols.append(i+1)
                vals.append(vals[-1])

    nnz = len(rows)
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"% Randomly generated {n}x{n} sparse symmetric matrix\n")
        f.write(f"{n} {n} {nnz//2}\n")  # MTX symmetric: only lower triangle
        for i in range(0, nnz, 2):  # write only one of each symmetric pair
            if rows[i] >= cols[i]:
                f.write(f"{rows[i]} {cols[i]} {vals[i]:.6f}\n")

    print(f"  Saved to {path} (nnz≈{nnz})")
    return path


# ──────────────────────────────────────────────────────────────
#  Running and parsing output
# ──────────────────────────────────────────────────────────────
def run_program(binary, mtx_path, timeout=300):
    """
    Run a binary with the given MTX file.
    Returns (stdout, elapsed_wall_time) or (None, None) on failure.
    """
    if not os.path.exists(binary):
        return None, None
    t_start = time.time()
    try:
        result = subprocess.run(
            [f"./{binary}", mtx_path],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t_start
        if result.returncode != 0:
            print(f"    [ERROR] {binary}:")
            print(result.stderr[:500])
            return None, elapsed
        return result.stdout, elapsed
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {binary} exceeded {timeout}s")
        return None, None


def parse_baseline_output(output):
    """Extract timing from baseline's stdout."""
    if not output: return {}
    res = {}
    m = re.search(r"Pass 1.*?:\s+([\d.]+) second", output)
    if m: res["pass1_symbolic"] = float(m.group(1))
    m = re.search(r"Pass 2.*?:\s+([\d.]+) second", output)
    if m: res["pass2_numeric"] = float(m.group(1))
    m = re.search(r"Total GPU.*?:\s+([\d.]+) second", output)
    if m: res["total"] = float(m.group(1))
    m = re.search(r"nnz\(C\)\s*=\s*(\d+)", output)
    if m: res["nnz_C"] = int(m.group(1))
    return res


def parse_tilespgemm_output(output):
    """Extract per-step timing from TileSpGEMM's stdout."""
    if not output: return {}
    res = {}
    m = re.search(r"Format conversion.*?([\d.]+) s", output)
    if m: res["format_conv"] = float(m.group(1))
    m = re.search(r"Step 1.*?([\d.]+) s", output)
    if m: res["step1"] = float(m.group(1))
    m = re.search(r"Step 2.*?([\d.]+) s", output)
    if m: res["step2"] = float(m.group(1))
    m = re.search(r"Step 3.*?([\d.]+) s", output)
    if m: res["step3"] = float(m.group(1))
    m = re.search(r"Total.*steps.*?([\d.]+) s", output)
    if m: res["total"] = float(m.group(1))
    m = re.search(r"nnz\(C\)\s*=\s*(\d+)", output)
    if m: res["nnz_C"] = int(m.group(1))
    return res


# ──────────────────────────────────────────────────────────────
#  Print a formatted results table
# ──────────────────────────────────────────────────────────────
def print_results_table(all_results):
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    hdr = (f"{'Matrix':<22} {'Baseline':<10} {'Base-Sym':<10} {'Base-Num':<10} "
           f"{'TileStep1':<10} {'TileStep2':<10} {'TileStep3':<10} "
           f"{'TileTotal':<10} {'Speedup':<8} {'nnz(C)'}")
    print(hdr)
    print("-"*100)

    for name, b, t in all_results:
        # Baseline timing
        b_total  = b.get("total", float("nan"))
        b_sym    = b.get("pass1_symbolic", float("nan"))
        b_num    = b.get("pass2_numeric", float("nan"))
        # TileSpGEMM timing
        t_total  = t.get("total", float("nan"))
        t_s1     = t.get("step1", float("nan"))
        t_s2     = t.get("step2", float("nan"))
        t_s3     = t.get("step3", float("nan"))
        # Speedup
        speedup = b_total / t_total if (b_total > 0 and t_total > 0) else float("nan")
        nnz_C   = t.get("nnz_C", b.get("nnz_C", "?"))

        def fmt(v):
            return f"{v:.4f}" if not (isinstance(v, float) and (v != v)) else "N/A"

        speedup_str = f"{speedup:.2f}x" if speedup == speedup else "N/A"
        print(f"{name:<22} {fmt(b_total):<10} {fmt(b_sym):<10} {fmt(b_num):<10} "
              f"{fmt(t_s1):<10} {fmt(t_s2):<10} {fmt(t_s3):<10} "
              f"{fmt(t_total):<10} {speedup_str:<8} {nnz_C}")

    print("="*100)
    print("\nNotes:")
    print("  All times in seconds.  Speedup = Baseline_total / TileSpGEMM_total.")
    print("  Step1 = tile structure finding (CPU)")
    print("  Step2 = symbolic phase (GPU: binary search + AtomicOr masks)")
    print("  Step3 = numeric phase (GPU: adaptive sparse/dense accumulator)")
    print("  Paper's expected Step shares: Step1≈5%, Step2≈15%, Step3≈70%, Mem≈20%")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test TileSpGEMM vs baseline")
    parser.add_argument("--generate",  action="store_true",
                        help="Use randomly generated matrices (no download)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip recompilation (assume binaries exist)")
    parser.add_argument("--quick",     action="store_true",
                        help="Use only small quick-test matrices")
    parser.add_argument("--matrix",    type=str, default=None,
                        help="Run on a single local .mtx file")
    args = parser.parse_args()

    # ---- Compilation ----
    if not args.skip_build:
        compile_programs()
    elif not check_binaries():
        print("Binaries not found! Run without --skip-build first.")
        sys.exit(1)

    all_results = []

    # ---- Single file mode ----
    if args.matrix:
        matrices = [("custom", os.path.basename(args.matrix).replace(".mtx",""),
                     "custom", args.matrix)]
    elif args.generate:
        # Generate random matrices of increasing sizes (no internet required)
        gen_specs = [
            (5000,  0.002,  "random_5k"),
            (20000, 0.0005, "random_20k"),
            (50000, 0.0002, "random_50k"),
        ]
        matrices = []
        for n, d, name in gen_specs:
            path = generate_random_mtx(n, d, name)
            matrices.append(("generated", name, f"{n}×{n}", path))
    else:
        matrix_list = QUICK_MATRICES if args.quick else PAPER_MATRICES
        matrices = []
        for group, name, desc in matrix_list:
            print(f"\nPreparing: {name}  ({desc})")
            path = download_matrix(group, name)
            if path:
                matrices.append((group, name, desc, path))

    if not matrices:
        print("\nNo matrices available. Use --generate for offline testing.")
        sys.exit(1)

    # ---- Run tests ----
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    for _, name, desc, mtx_path in matrices:
        print(f"\n{'─'*60}")
        print(f"Matrix: {name}  ({desc})")
        print(f"File:   {mtx_path}")
        print(f"{'─'*60}")

        # Run baseline
        print("\n[1/2] Baseline SpGEMM:")
        b_out, _ = run_program("spgemm_baseline", mtx_path)
        if b_out: print(b_out.strip())
        b_stats = parse_baseline_output(b_out) if b_out else {}

        # Run TileSpGEMM
        print("\n[2/2] TileSpGEMM:")
        t_out, _ = run_program("tilespgemm", mtx_path)
        if t_out: print(t_out.strip())
        t_stats = parse_tilespgemm_output(t_out) if t_out else {}

        all_results.append((name, b_stats, t_stats))

        # Quick sanity check: nnz(C) should match between both methods
        nnz_b = b_stats.get("nnz_C")
        nnz_t = t_stats.get("nnz_C")
        if nnz_b and nnz_t:
            if nnz_b == nnz_t:
                print(f"\n  ✓ nnz(C) match: {nnz_b}")
            else:
                print(f"\n  ✗ nnz(C) MISMATCH: baseline={nnz_b}, tilespgemm={nnz_t}")

    # ---- Print summary table ----
    print_results_table(all_results)

    # ---- Save results to file ----
    results_file = "results.txt"
    with open(results_file, "w") as f:
        f.write("TileSpGEMM vs Baseline SpGEMM Results\n")
        f.write("="*60 + "\n\n")
        for name, b, t in all_results:
            f.write(f"Matrix: {name}\n")
            f.write(f"  Baseline:     step1={b.get('pass1_symbolic','?')}s, "
                    f"step2={b.get('pass2_numeric','?')}s, "
                    f"total={b.get('total','?')}s\n")
            f.write(f"  TileSpGEMM:   step1={t.get('step1','?')}s, "
                    f"step2={t.get('step2','?')}s, "
                    f"step3={t.get('step3','?')}s, "
                    f"total={t.get('total','?')}s\n")
            f.write(f"  nnz(C): baseline={b.get('nnz_C','?')}, "
                    f"tilespgemm={t.get('nnz_C','?')}\n\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
