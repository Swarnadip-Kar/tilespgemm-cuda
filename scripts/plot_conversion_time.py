#!/usr/bin/env python3
"""
plot_conversion_time.py  ─  Figure 12 equivalent
Scatter plot: CSR→Tiled format conversion time vs single TileSpGEMM runtime.
X-axis: #flops of C = A^2 (log10 scale)
Y-axis: runtime (ms, log10 scale)
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

names    = []
flops    = []
conv_ms  = []
total_ms = []

for entry in data:
    td = entry.get('tile', {})
    if not td or td.get('flops', 0) == 0:
        continue
    names.append(entry['matrix'])
    flops.append(td['flops'])
    conv_ms.append(td.get('conversion_ms', 0))
    total_ms.append(td.get('time_ms', 0))

flops    = np.array(flops,    dtype=float)
conv_ms  = np.array(conv_ms,  dtype=float)
total_ms = np.array(total_ms, dtype=float)

log_flops   = np.log10(flops + 1e-9)
log_conv    = np.log10(np.clip(conv_ms,  1e-3, None))
log_total   = np.log10(np.clip(total_ms, 1e-3, None))

fig, ax = plt.subplots(figsize=(8, 5))

sc1 = ax.scatter(log_flops, log_conv,  s=80, c='steelblue',
                 marker='o', label='CSR → Tiled format conversion time', zorder=5, alpha=0.85)
sc2 = ax.scatter(log_flops, log_total, s=80, c='darkorange',
                 marker='^', label='Single TileSpGEMM runtime', zorder=5, alpha=0.85)

# Label each point
for i, name in enumerate(names):
    ax.annotate(name, (log_flops[i], log_total[i]),
                textcoords='offset points', xytext=(4, 2), fontsize=7, alpha=0.8)

# Trend lines
if len(log_flops) > 1:
    for yvals, color, ls in [(log_conv,'steelblue','--'),(log_total,'darkorange','-.')]:
        z = np.polyfit(log_flops, yvals, 1)
        p = np.poly1d(z)
        xfit = np.linspace(log_flops.min(), log_flops.max(), 100)
        ax.plot(xfit, p(xfit), linestyle=ls, color=color, alpha=0.5, linewidth=1.5)

ax.set_xlabel(r'$\#$flops of $C = A^2$ (log$_{10}$ scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Runtime (ms, log$_{10}$ scale)', fontsize=12, fontweight='bold')
ax.set_title('CSR→Tiled Format Conversion Time vs TileSpGEMM Runtime\n'
             '(Conversion cost amortized over multiple SpGEMM calls)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig4_conversion_vs_spgemm.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[Plot] Saved: {out}", flush=True)
plt.close()
