#!/usr/bin/env python3
"""
make_all_graphs.py — Calls all five plotting scripts in sequence.
"""
import sys, os, subprocess

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(SCRIPT_DIR), 'results', 'results.json')
GRAPHS_DIR   = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(SCRIPT_DIR), 'graphs')
PYTHON       = sys.executable

PLOTS = [
    ('plot_gflops.py',            'Figure 1 — GFlops comparison'),
    ('plot_peak_space.py',        'Figure 2 — Peak space cost'),
    ('plot_runtime_breakdown.py', 'Figure 3 — Runtime breakdown'),
    ('plot_conversion_time.py',   'Figure 4 — Conversion time vs SpGEMM time'),
    ('plot_space_comparison.py',  'Figure 5 — CSR vs Tiled space'),
]

print(f"\n{'='*60}", flush=True)
print("  Generating all graphs ...", flush=True)
print(f"{'='*60}", flush=True)

for script, desc in PLOTS:
    path = os.path.join(SCRIPT_DIR, script)
    print(f"\n[Graph] {desc}", flush=True)
    r = subprocess.run([PYTHON, path, RESULTS_FILE, GRAPHS_DIR],
                       capture_output=False, text=True)
    if r.returncode != 0:
        print(f"  [WARN] {script} exited with code {r.returncode}", flush=True)

print(f"\n[Done] All graphs saved to: {GRAPHS_DIR}", flush=True)
print(f"       Files: {sorted(os.listdir(GRAPHS_DIR))}", flush=True)
