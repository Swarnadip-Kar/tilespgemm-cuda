#!/usr/bin/env python3
"""
plot_runtime_breakdown.py  ─  Figure 10 equivalent
Runtime breakdown of TileSpGEMM per matrix.
Stacked bar: Step1 | Step2 | Step3 (+ conversion shown separately).
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
step1_ms  = []
step2_ms  = []
step3_ms  = []
conv_ms   = []

for entry in data:
    td = entry.get('tile', {})
    if not td:
        continue
    matrices.append(entry['matrix'])
    step1_ms.append(td.get('step1_ms', 0))
    step2_ms.append(td.get('step2_ms', 0))
    step3_ms.append(td.get('step3_ms', 0))
    conv_ms.append(td.get('conversion_ms', 0))

n = len(matrices)
x = np.arange(n)
w = 0.5

fig, ax = plt.subplots(figsize=(max(10, n*1.5), 6))

b1 = ax.bar(x, step1_ms, w, label='Step 1: Tile structure (symbolic SpGEMM on A′B′)',
            color='tomato', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x, step2_ms, w, bottom=step1_ms,
            label='Step 2: Symbolic phase (bitmask + rowPtr per tile)',
            color='mediumseagreen', edgecolor='black', linewidth=0.5)
b3 = ax.bar(x, step3_ms, w,
            bottom=[step1_ms[i]+step2_ms[i] for i in range(n)],
            label='Step 3: Numeric phase (adaptive accumulator)',
            color='gold', edgecolor='black', linewidth=0.5)

# Total time annotation on top of each bar
for i in range(n):
    total = step1_ms[i]+step2_ms[i]+step3_ms[i]
    if total > 0:
        ax.text(x[i], total+0.5, f'{total:.1f}ms', ha='center', va='bottom',
                fontsize=7, fontweight='bold')

ax.set_xlabel('Matrix Name', fontsize=12, fontweight='bold')
ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
ax.set_title('TileSpGEMM Runtime Breakdown per Phase\n'
             r'(Computing $C = A^2$, Step 1 < 5% of total)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=30, ha='right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=9, loc='upper right')

# Percentage annotations inside bars
for i in range(n):
    total = step1_ms[i]+step2_ms[i]+step3_ms[i]
    if total <= 0:
        continue
    def pct(v): return f'{100*v/total:.0f}%'
    # Only annotate if bar is tall enough
    if step1_ms[i]/total > 0.05:
        ax.text(x[i], step1_ms[i]/2, pct(step1_ms[i]),
                ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    mid2 = step1_ms[i] + step2_ms[i]/2
    if step2_ms[i]/total > 0.05:
        ax.text(x[i], mid2, pct(step2_ms[i]),
                ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    mid3 = step1_ms[i]+step2_ms[i]+step3_ms[i]/2
    if step3_ms[i]/total > 0.05:
        ax.text(x[i], mid3, pct(step3_ms[i]),
                ha='center', va='center', fontsize=7, color='black', fontweight='bold')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig3_runtime_breakdown.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[Plot] Saved: {out}", flush=True)
plt.close()

# ── Also plot as percentage stacked bars (100%) ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(max(10, n*1.5), 5))
pct1 = [100*step1_ms[i]/(step1_ms[i]+step2_ms[i]+step3_ms[i]) if (step1_ms[i]+step2_ms[i]+step3_ms[i])>0 else 0 for i in range(n)]
pct2 = [100*step2_ms[i]/(step1_ms[i]+step2_ms[i]+step3_ms[i]) if (step1_ms[i]+step2_ms[i]+step3_ms[i])>0 else 0 for i in range(n)]
pct3 = [100*step3_ms[i]/(step1_ms[i]+step2_ms[i]+step3_ms[i]) if (step1_ms[i]+step2_ms[i]+step3_ms[i])>0 else 0 for i in range(n)]

ax2.bar(x, pct1, w, label='Step 1', color='tomato',       edgecolor='black', linewidth=0.5)
ax2.bar(x, pct2, w, bottom=pct1, label='Step 2',         color='mediumseagreen', edgecolor='black', linewidth=0.5)
ax2.bar(x, pct3, w, bottom=[pct1[i]+pct2[i] for i in range(n)], label='Step 3',
        color='gold', edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Matrix Name', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage of Total Runtime (%)', fontsize=12, fontweight='bold')
ax2.set_title('TileSpGEMM Runtime Breakdown (% of total)\n'
              r'(Computing $C = A^2$)', fontsize=13, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(matrices, rotation=30, ha='right', fontsize=9)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(fontsize=9)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'fig3b_runtime_breakdown_pct.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"[Plot] Saved: {out2}", flush=True)
plt.close()
