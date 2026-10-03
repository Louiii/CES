"""Figure 5: retained position-loss sweep, styled like Figure 2.

The frozen three-seed means and 95% Student-t intervals are preserved.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import math
import sys
sys.dont_write_bytecode = True
from build_method_scaling import STYLE, ROOT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator

PAPER = ROOT / 'inputs/paper'
RAW = PAPER / 'evidence/position_weight_raw.csv'
SUMMARY = PAPER / 'evidence/position_weight_summary.csv'
OUTPUT = ROOT / 'output/pdf/position_weight_sweep'
BLUE, PURPLE = '#326EA8', '#8874B3'
METRICS = [
    ('val_bpb', 'A  Clean language modelling', 'Bits per byte (lower is better)'),
    ('core_centered', 'B  General capabilities', 'CORE score'),
    ('synthetic_position_accuracy', 'C  Synthetic localisation', 'Position accuracy'),
    ('human_position_accuracy', 'D  CoEdIT localisation', 'Position accuracy'),
    ('controlled_valid_rate', 'E  Valid edit programs', 'Valid-program rate'),
    ('controlled_exact_target_rate', 'F  Exact repair', 'Exact-target rate'),
    ('controlled_similarity_improvement_rate', 'G  Repair direction', 'Similarity-improvement rate'),
    ('controlled_mean_char_similarity_delta', 'H  Repair magnitude', 'Mean similarity change'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows = list(csv.DictReader(RAW.open()))
    summary = sorted(csv.DictReader(SUMMARY.open()), key=lambda r: float(r['pos_weight']))
    weights = [float(r['pos_weight']) for r in summary]
    assert len(rows) == 18 and len(summary) == 6
    for metric, _, _ in METRICS:
        for r in summary:
            vals = [float(v[metric]) for v in rows if float(v['pos_weight']) == float(r['pos_weight'])]
            assert len(vals) == 3
            mean = statistics.mean(vals)
            half = 4.303 * statistics.stdev(vals) / math.sqrt(3)
            assert math.isclose(mean, float(r[metric+'_mean']), abs_tol=1e-12)
            assert math.isclose(mean-half, float(r[metric+'_ci95_low']), abs_tol=1e-12)
            assert math.isclose(mean+half, float(r[metric+'_ci95_high']), abs_tol=1e-12)
    with plt.rc_context({**STYLE, 'font.size': 15, 'axes.labelsize': 15,
                         'axes.titlesize': 17, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14}):
        fig = plt.figure(figsize=(13.8, 11.3), facecolor='white')
        grid = fig.add_gridspec(3, 3, left=.075, right=.985, bottom=.075, top=.915,
                                wspace=.34, hspace=.39)
        axes = []
        for i, (metric, title, ylabel) in enumerate(METRICS):
            ax = fig.add_subplot(grid[i//3, i%3], sharex=axes[0] if axes else None)
            axes.append(ax)
            for seed in ['42','43','44']:
                selected = sorted((r for r in rows if r['model_seed'] == seed), key=lambda r: float(r['pos_weight']))
                ax.plot(weights, [float(r[metric]) for r in selected], color=BLUE,
                        alpha=.22, lw=1.1, marker='o', markersize=3, zorder=2)
            means = [float(r[metric+'_mean']) for r in summary]
            errors = [[v-float(r[metric+'_ci95_low']) for v,r in zip(means,summary)],
                      [float(r[metric+'_ci95_high'])-v for v,r in zip(means,summary)]]
            ax.errorbar(weights, means, yerr=errors, color=BLUE, lw=1.8,
                        marker='o', markersize=5, elinewidth=1.1, capsize=3, zorder=3)
            ax.axvline(.01, color=PURPLE, linestyle=':', lw=1.3, alpha=.7, zorder=1)
            ax.set_xscale('log'); ax.set_xlim(.0017,.12)
            ax.xaxis.set_major_locator(FixedLocator([.002,.01,.1]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(title, loc='left', pad=12)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.tick_params(axis='x', labelbottom=i>=6)
            if i>=6:
                ax.set_xlabel('Position-loss weight', labelpad=8)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis='y', color='#E5E9ED', lw=.7); ax.set_axisbelow(True)
            if metric == 'controlled_mean_char_similarity_delta':
                ax.axhline(0, color='#C8D1DC', lw=.9, zorder=1)
        ax = fig.add_subplot(grid[2,2])
        offsets = [(-8,-18),(-24,-5),(2,20),(7,-14),(0,-18),(-12,8)]
        for r, offset in zip(summary, offsets):
            x,y = float(r['val_bpb_mean']),float(r['human_position_accuracy_mean'])
            selected = float(r['pos_weight']) == .01
            color = PURPLE if selected else BLUE
            xe = [[x-float(r['val_bpb_ci95_low'])],[float(r['val_bpb_ci95_high'])-x]]
            ye = [[y-float(r['human_position_accuracy_ci95_low'])],[float(r['human_position_accuracy_ci95_high'])-y]]
            ax.errorbar(x,y,xerr=xe,yerr=ye,color=color,alpha=.9 if selected else .6,
                        marker='o',ms=6 if selected else 4.5,capsize=3,lw=1.1)
            ax.annotate(f"{float(r['pos_weight']):g}",(x,y),xytext=offset,
                        textcoords='offset points',fontsize=13,ha='center',color=color,
                        weight='bold' if selected else 'normal',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.5))
        ax.set_title('I  Retention / localisation', loc='left',pad=12)
        ax.set_xlabel('Bits per byte (lower is better)',labelpad=8)
        ax.set_ylabel('CoEdIT position accuracy',labelpad=8)
        ax.set_xlim(.887,.95)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y',color='#E5E9ED',lw=.7);ax.set_axisbelow(True)
        fig.legend(handles=[Line2D([],[],color=BLUE,alpha=.25,lw=1.5,label='Individual seeds'),
                            Line2D([],[],color=BLUE,lw=1.8,marker='o',label='Mean ± 95% t interval'),
                            Line2D([],[],color=PURPLE,lw=1.3,ls=':',label='Selected weight: 0.01')],
                   loc='upper center',bbox_to_anchor=(.53,.988),ncol=3,frameon=False,fontsize=16)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUTPUT.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.07)
        fig.savefig(OUTPUT.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=.07)
        plt.close(fig)
    if args.install:
        dest=PAPER/'figures/position_weight_sweep.pdf'
        shutil.copy2(OUTPUT.with_suffix('.pdf'),dest)
        path=PAPER/'evidence/selected_panel_assets.json'
        assets=json.loads(path.read_text())
        assets[dest.name]=dict(source=str(OUTPUT.with_suffix('.pdf')),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator=str(Path(__file__).resolve()),
            data=[str(RAW),str(SUMMARY)],
            data_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW,SUMMARY]],
            original_figure_sha256='07f385884dfe9dbe0b39f8b74a0cf141a0d35c626ae39ac39a5df97f0a8c8710')
        path.write_text(json.dumps(assets,indent=2)+'\n')
    print(OUTPUT.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
