#!/usr/bin/env python3
"""
plot_space_comparison.py  ─  Figure 11 equivalent (CSR vs Tiled only)
Space cost comparison: standard CSR vs our tiled sparse format.
One scatter/bar per matrix.
"""

import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else 'results/results.json'
OUT_DIR      = sys.argv[2] if len(sys.argv) > 2 else 'graphs'
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

matrices    = []
csr_mb      = []
tiled_mb    = []

for entry in data:
    td = entry.get('tile', {})
    if not td:
        continue
    cb = td.get('csr_bytes', 0) / 1e6
    tb = td.get('tiled_bytes', 0) / 1e6
    if cb == 0 and tb == 0:
        continue
    matrices.append(entry['matrix'])
    csr_mb.append(cb)
    tiled_mb.append(tb)

n = len(matrices)
x = np.arange(n)
w = 0.35

fig, ax = plt.subplots(figsize=(max(10, n*1.8), 6))

bars_csr   = ax.bar(x - w/2, csr_mb,   w, label='Standard CSR format',
                    color='steelblue', edgecolor='black', linewidth=0.5)
bars_tiled = ax.bar(x + w/2, tiled_mb, w, label='Tiled sparse format (TileSpGEMM)',
                    color='darkorange', edgecolor='black', linewidth=0.5)

# Value labels
for bar in bars_csr:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.2, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7, rotation=45)
for bar in bars_tiled:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.2, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7, rotation=45)

# Difference annotation
for i, (cb, tb) in enumerate(zip(csr_mb, tiled_mb)):
    diff = cb - tb
    if abs(diff) > 0.1:
        sign = '−' if diff > 0 else '+'
        ax.annotate(f'{sign}{abs(diff):.1f} MB',
                    xy=(x[i], max(cb,tb)+1.5),
                    ha='center', fontsize=7, color='dimgray',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    xytext=(x[i], max(cb,tb)+2.5))

ax.set_xlabel('Matrix Name', fontsize=12, fontweight='bold')
ax.set_ylabel('Storage Size (MB)', fontsize=12, fontweight='bold')
ax.set_title('Space Cost Comparison: Standard CSR vs Tiled Sparse Format\n'
             '(Tiled format includes rowPtr[16] + bitmask[16] per tile; '
             'often less than CSR due to compact local indices)',
             fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=30, ha='right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=10)

# Add note about why tiled can be smaller
ax.text(0.01, 0.97,
        'Note: Tiled uses 8-bit local indices vs 32-bit global indices in CSR.\n'
        'For dense tiles this saves 3 bytes/nonzero.',
        transform=ax.transAxes, fontsize=8, va='top', color='dimgray',
        bbox=dict(facecolor='lightyellow', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig5_space_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[Plot] Saved: {out}", flush=True)
plt.close()
