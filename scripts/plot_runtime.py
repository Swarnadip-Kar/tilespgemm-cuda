#!/usr/bin/env python3
"""
plot_runtime.py  ─  Runtime comparison
Execution time (ms) for each matrix — TileSpGEMM vs RowSpGEMM.
Grouped bar chart, one group per matrix.
"""

import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else 'results/results.json'
OUT_DIR      = sys.argv[2] if len(sys.argv) > 2 else 'graphs'
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

matrices     = []
row_times    = []
tile_times   = []

for entry in data:
    name = entry['matrix']
    rt = entry.get('row',  {}).get('time_ms', 0.0)
    tt = entry.get('tile', {}).get('time_ms', 0.0)
    matrices.append(name)
    row_times.append(rt)
    tile_times.append(tt)

n = len(matrices)
x = np.arange(n)
w = 0.35

fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 6))

bars1 = ax.bar(x - w/2, row_times,  w, label='RowSpGEMM (cuSPARSE)',
               color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, tile_times, w, label='TileSpGEMM (this work)',
               color='darkorange', edgecolor='black', linewidth=0.5)

def autolabel(bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, rotation=45)

autolabel(bars1)
autolabel(bars2)

ax.set_xlabel('Matrix Name', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('SpGEMM Runtime Comparison: RowSpGEMM vs TileSpGEMM\n'
             r'(Computing $C = A^2$, double precision)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=30, ha='right', fontsize=9)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=10, loc='upper right')

# Annotate failures
for i, (rt, tt) in enumerate(zip(row_times, tile_times)):
    if rt == 0:
        ax.text(x[i] - w/2, 0.01, 'N/A', ha='center', va='bottom',
                fontsize=7, color='red', rotation=90)
    if tt == 0:
        ax.text(x[i] + w/2, 0.01, 'N/A', ha='center', va='bottom',
                fontsize=7, color='red', rotation=90)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_runtime_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[Plot] Saved: {out}", flush=True)
plt.close()
