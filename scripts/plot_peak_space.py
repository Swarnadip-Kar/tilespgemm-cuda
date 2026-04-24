#!/usr/bin/env python3
"""
plot_peak_space.py  ─  Figure 9 equivalent
Runtime peak memory cost (MB) for each matrix — TileSpGEMM vs RowSpGEMM.
Side-by-side bar chart.
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

matrices  = []
row_mb    = []
tile_mb   = []
row_ms    = []
tile_ms   = []

for entry in data:
    matrices.append(entry['matrix'])
    row_mb.append(entry.get('row',  {}).get('mem_bytes', 0) / 1e6)
    tile_mb.append(entry.get('tile', {}).get('mem_bytes', 0) / 1e6)
    row_ms.append(entry.get('row',  {}).get('time_ms', 0))
    tile_ms.append(entry.get('tile', {}).get('time_ms', 0))

n  = len(matrices)
x  = np.arange(n)
w  = 0.35
colors_row  = ['steelblue'] * n
colors_tile = ['darkorange'] * n

fig, ax = plt.subplots(figsize=(max(10, n*1.8), 6))

ax.bar(x - w/2, row_mb,  w, label='RowSpGEMM (cuSPARSE)',
       color=colors_row,  edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, tile_mb, w, label='TileSpGEMM (this work)',
       color=colors_tile, edgecolor='black', linewidth=0.5)

# Annotate bars with MB value
for i, (rm, tm) in enumerate(zip(row_mb, tile_mb)):
    if rm > 0:
        ax.text(x[i]-w/2, rm+0.5, f'{rm:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)
    if tm > 0:
        ax.text(x[i]+w/2, tm+0.5, f'{tm:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Matrix Name', fontsize=12, fontweight='bold')
ax.set_ylabel('Peak Memory Cost (MB)', fontsize=12, fontweight='bold')
ax.set_title('Runtime Peak Space Cost\n'
             r'(Computing $C = A^2$, includes all device allocations)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=30, ha='right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=10)

# Secondary x-axis: completion time annotation
for i, (rms, tms) in enumerate(zip(row_ms, tile_ms)):
    ax.annotate(f'Row:{rms:.0f}ms\nTile:{tms:.0f}ms',
                xy=(x[i], 0), xytext=(x[i], -max(row_mb+tile_mb)*0.15 if max(row_mb+tile_mb)>0 else -1),
                fontsize=6, ha='center', va='top', color='dimgray')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig2_peak_space_cost.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[Plot] Saved: {out}", flush=True)
plt.close()
