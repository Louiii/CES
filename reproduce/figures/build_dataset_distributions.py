"""Restyle the original Figure 4 statistics with the Figure 2 theme.

Run with --install to update the manuscript asset. The archived histogram
counts, 50-character bins, overflow bin, and bar estimates are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
from build_method_scaling import STYLE, ROOT
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

PAPER = ROOT / 'inputs/paper'
SOURCE = PAPER / 'evidence/dataset_distribution_stats.json'
OUTPUT = ROOT / 'output/pdf/dataset_distributions'
SERIES = [('generated', 'GPT-generated', '#326EA8'),
          ('random_fallback', 'Random', '#44988B'),
          ('copy_candidates', 'Copy-mined', '#8874B3')]
OPS = ['insert', 'delete', 'replace', 'move', 'copy']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    payload = json.loads(SOURCE.read_text())
    data = payload['datasets']
    assert payload['direction'] == 'inverse'
    for name, _, _ in SERIES:
        for op, hist in data[name]['length_histograms'].items():
            assert sum(hist['counts']) == data[name]['total_counts'][op]
    with plt.rc_context({**STYLE, 'font.size': 16, 'axes.titlesize': 18,
                         'axes.labelsize': 16, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14}):
        fig = plt.figure(figsize=(11.8, 9.2), facecolor='white')
        outer = fig.add_gridspec(1, 2, left=.075, right=.985, bottom=.085,
                                top=.88, width_ratios=[1.15, 1], wspace=.25)
        left = outer[0].subgridspec(5, 1, hspace=.30)
        right = outer[1].subgridspec(2, 1, hspace=.43)
        hist_axes = []
        for i, op in enumerate(OPS):
            ax = fig.add_subplot(left[i], sharex=hist_axes[0] if hist_axes else None)
            hist_axes.append(ax)
            for name, _, color in SERIES:
                hist = data[name]['length_histograms'].get(op)
                if not hist:
                    continue
                edges = np.asarray(hist['bin_edges'])
                counts = np.asarray(hist['counts'])
                ax.stairs(counts, edges, fill=True, color=color, alpha=.17, lw=0)
                ax.stairs(counts, edges, color=color, alpha=.85, lw=1.7)
            ax.text(.98, .82, op.title(), transform=ax.transAxes,
                    ha='right', va='top', fontsize=16, weight='bold')
            ax.set_xlim(0, 3000)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k' if v >= 1000 else f'{v:g}'))
            ax.set_xticks([0, 1000, 2000, 3000])
            ax.tick_params(axis='x', labelbottom=i == 4)
            ax.grid(axis='y', color='#E5E9ED', lw=.7)
            ax.set_axisbelow(True)
        hist_axes[0].set_title('A  Edit-length distributions', loc='left', pad=16)
        hist_axes[2].set_ylabel('Number of edits', labelpad=13)
        hist_axes[-1].set_xlabel('Edit length (characters)', labelpad=9)
        for row, key, title, ylabel in [
            (0, 'mean_counts_per_edited_doc', 'B  Edits per document', 'Mean count per edited document'),
            (1, 'op_fraction', 'C  Operation mix', 'Share of edits (%)')]:
            ax = fig.add_subplot(right[row])
            x = np.arange(len(OPS)); bar_width = .23
            for j, (name, _, color) in enumerate(SERIES):
                values = [data[name][key].get(op, 0) for op in OPS]
                ax.bar(x+(j-1)*bar_width, values, width=bar_width*.92,
                       color=color, alpha=.82, edgecolor='none', zorder=3)
            ax.set_title(title, loc='left', pad=16)
            ax.set_xticks(x, [op.title() for op in OPS])
            ax.tick_params(axis='x', length=0, pad=9)
            ax.set_ylabel(ylabel, labelpad=10)
            ax.set_ylim(0, 2.5 if row == 0 else 1.06)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if row == 1:
                ax.set_yticks([0,.25,.5,.75,1])
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=''))
            ax.grid(axis='y', color='#E5E9ED', lw=.7)
            ax.set_axisbelow(True)
        fig.legend(handles=[Patch(facecolor=c, alpha=.82, label=l) for _, l, c in SERIES],
                   loc='upper center', bbox_to_anchor=(.53,.985), ncol=3,
                   frameon=False, fontsize=17, handlelength=1.3, columnspacing=2.4)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.06)
        fig.savefig(OUTPUT.with_suffix('.png'), dpi=180, bbox_inches='tight', pad_inches=.06)
        plt.close(fig)
    if args.install:
        dest = PAPER / 'figures/dataset_distributions.pdf'
        shutil.copy2(OUTPUT.with_suffix('.pdf'), dest)
        path = PAPER / 'evidence/selected_panel_assets.json'
        assets = json.loads(path.read_text())
        assets[dest.name] = dict(source=str(OUTPUT.with_suffix('.pdf')),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            data=str(SOURCE), data_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            generator=str(Path(__file__).resolve()),
            original_plot_sha256='10765fe49289235b09585ae1052b68743b9f411355233fdbde58e43a7a13173b')
        path.write_text(json.dumps(assets, indent=2)+'\n')
    print(OUTPUT.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
