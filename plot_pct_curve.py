"""
Figure 4.3 — Effect of GEN Pseudo-label Percentile (PCT) on CUB-200-2011
Ablation data từ eval_cub_gesnc.py với checkpoint epoch 60.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Ablation data ────────────────────────────────────────────────────────────
pct_vals   = [0,     5,     10,    15,    20]
all_acc    = [62.41, 63.17, 64.01, 64.20, 64.19]
old_acc    = [69.62, 70.18, 71.19, 71.86, 73.57]
new_acc    = [55.36, 56.31, 57.00, 56.72, 55.02]
h_score    = [61.68, 62.49, 63.31, 63.40, 62.95]

# CiPR baseline (SNC gốc, epoch 60, không có GEN)
baseline_all = 62.22
baseline_old = 63.58
baseline_new = 61.54

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

fig, ax = plt.subplots(figsize=(8, 5.2))

# ── Line plots ───────────────────────────────────────────────────────────────
ax.plot(pct_vals, all_acc, 'o-', color='#2563EB', linewidth=2.2,
        markersize=8, label='All ACC', zorder=3)
ax.plot(pct_vals, old_acc, 's--', color='#16A34A', linewidth=2.0,
        markersize=7, label='Old ACC', zorder=3)
ax.plot(pct_vals, new_acc, '^--', color='#DC2626', linewidth=2.0,
        markersize=7, label='New ACC', zorder=3)
ax.plot(pct_vals, h_score, 'D-', color='#7C3AED', linewidth=2.2,
        markersize=7, label='H-score', zorder=3)

# ── Annotate best points ──────────────────────────────────────────────────────
best_all_idx = all_acc.index(max(all_acc))  # PCT=15
best_h_idx   = h_score.index(max(h_score))  # PCT=15

ax.annotate(f'All={max(all_acc):.2f}%',
            xy=(pct_vals[best_all_idx], max(all_acc)),
            xytext=(pct_vals[best_all_idx] - 2.5, max(all_acc) + 0.6),
            fontsize=10, color='#2563EB', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.2))

ax.annotate(f'H={max(h_score):.2f}%',
            xy=(pct_vals[best_h_idx], max(h_score)),
            xytext=(pct_vals[best_h_idx] + 0.5, max(h_score) + 0.6),
            fontsize=10, color='#7C3AED', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=1.2))

# ── CiPR Baseline horizontal lines ───────────────────────────────────────────
ax.axhline(baseline_all, color='#2563EB', linewidth=1.2, linestyle=':', alpha=0.6)
ax.axhline(baseline_old, color='#16A34A', linewidth=1.2, linestyle=':', alpha=0.6)
ax.axhline(baseline_new, color='#DC2626', linewidth=1.2, linestyle=':', alpha=0.6)

# Label baseline ở lề phải
xmax = 20.8
ax.text(xmax, baseline_all - 0.35, f'CiPR {baseline_all}%',
        fontsize=9, color='#2563EB', alpha=0.75, ha='left')
ax.text(xmax, baseline_old + 0.15, f'CiPR {baseline_old}%',
        fontsize=9, color='#16A34A', alpha=0.75, ha='left')
ax.text(xmax, baseline_new + 0.15, f'CiPR {baseline_new}%',
        fontsize=9, color='#DC2626', alpha=0.75, ha='left')

# ── Shade vùng tối ưu PCT=10-15 ──────────────────────────────────────────────
ax.axvspan(10, 15, alpha=0.08, color='#7C3AED', label='Optimal PCT range')
ax.axvline(15, color='#7C3AED', linewidth=1.0, linestyle=':', alpha=0.5)

# ── PCT=0 annotation ─────────────────────────────────────────────────────────
ax.annotate('Pure SNC\n(no GEN)', xy=(0, 62.41),
            xytext=(1.5, 60.8), fontsize=9, color='#64748B',
            arrowprops=dict(arrowstyle='->', color='#64748B', lw=1.0))

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlabel('GEN Pseudo-label Percentile PCT (%)', labelpad=8)
ax.set_ylabel('Accuracy (%)', labelpad=8)
ax.set_title('Effect of GEN Filtering Threshold on CUB-200-2011\n'
             '(CiPR backbone, epoch 60, K=200)', pad=12)
ax.set_xticks(pct_vals)
ax.set_xticklabels([f'PCT={p}%\n({[1500,1886,2221,2542,2862][i]} anchors)'
                    for i, p in enumerate(pct_vals)], fontsize=9)
ax.set_ylim(53, 76)
ax.set_xlim(-1.5, 23)

# ── Legend ────────────────────────────────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
optimal_patch = mpatches.Patch(color='#7C3AED', alpha=0.15, label='Optimal range')
ax.legend(handles + [optimal_patch],
          labels + ['Optimal range'],
          loc='lower right', framealpha=0.9, edgecolor='#CBD5E1')

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig4_3_pct_curve_cub200.pdf')
out_png  = os.path.join(out_dir, 'fig4_3_pct_curve_cub200.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_png,  dpi=300, bbox_inches='tight')
print(f"Saved:\n  {out_path}\n  {out_png}")
plt.show()
