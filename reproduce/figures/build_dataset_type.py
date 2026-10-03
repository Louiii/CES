"""Figure 6: dataset-source ablation with the shared paper theme."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
sys.dont_write_bytecode = True
from build_method_scaling import STYLE, ROOT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

PAPER = ROOT / 'inputs/paper'
RAW = PAPER / 'evidence/dataset_type_raw.csv'
SUMMARY = PAPER / 'evidence/dataset_type_summary.csv'
OUTPUT = ROOT / 'output/pdf/dataset_type_sweep'
VARIANTS = ['random', 'llm', 'mixed']
COLORS = ['#44988B', '#326EA8', '#8874B3']
METRICS = [
    ('val_bpb', 'A  Clean language\n     modelling', 'Bits per byte (lower is better)', False),
    ('core_centered', 'B  General\n     capabilities', 'CORE score', False),
    ('synthetic_position_accuracy', 'C  Synthetic\n     localisation', 'Position accuracy (%)', True),
    ('human_position_accuracy', 'D  CoEdIT\n     localisation', 'Position accuracy (%)', True),
    ('controlled_valid_rate', 'E  Valid edit programs', 'Valid-program rate (%)', True),
    ('controlled_exact_target_rate', 'F  Exact repair', 'Exact-target rate (%)', True),
    ('controlled_similarity_improvement_rate', 'G  Repair direction', 'Similarity-improvement rate (%)', True),
    ('loader_edit_mass_rho', 'H  Training edit mass', 'Measured edit mass (ρ)', False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows = list(csv.DictReader(RAW.open()))
    by_variant = {r['dataset_variant']:r for r in csv.DictReader(SUMMARY.open())}
    summary = [by_variant[v] for v in VARIANTS]
    assert len(rows) == 9 and len(summary) == 3
    for metric, _, _, _ in METRICS:
        for variant, r in zip(VARIANTS, summary):
            vals = [float(v[metric]) for v in rows if v['dataset_variant'] == variant]
            assert len(vals) == 3
            mean = statistics.mean(vals)
            half = 4.303 * statistics.stdev(vals) / math.sqrt(3)
            for suffix, value in [('mean',mean),('ci95_low',mean-half),('ci95_high',mean+half)]:
                assert math.isclose(value,float(r[metric+'_'+suffix]),abs_tol=1e-12)
    with plt.rc_context({**STYLE, 'font.size': 16, 'axes.labelsize': 16,
                         'axes.titlesize': 18, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15}):
        fig = plt.figure(figsize=(15.2,8.7),facecolor='white')
        grid = fig.add_gridspec(2,4,left=.07,right=.99,bottom=.105,top=.83,
                                wspace=.40,hspace=.34)
        axes=[]
        for i,(metric,title,ylabel,percent) in enumerate(METRICS):
            ax=fig.add_subplot(grid[i//4,i%4],sharex=axes[0] if axes else None)
            axes.append(ax)
            for seed in ['42','43','44']:
                group={r['dataset_variant']:r for r in rows if r['model_seed']==seed}
                assert set(group)==set(VARIANTS)
                ax.plot(range(3),[float(group[v][metric]) for v in VARIANTS],
                        color='#7F8A9B',alpha=.27,lw=1.1,marker='o',ms=3,zorder=2)
            means=[float(r[metric+'_mean']) for r in summary]
            ax.plot(range(3),means,color='#69788C',alpha=.65,lw=1.6,zorder=3)
            for x,(r,mean,color) in enumerate(zip(summary,means,COLORS)):
                err=[[mean-float(r[metric+'_ci95_low'])],[float(r[metric+'_ci95_high'])-mean]]
                ax.errorbar(x,mean,yerr=err,color=color,marker='o',ms=6,
                            lw=1.4,capsize=3,zorder=4)
            ax.set_title(title,loc='left',pad=14)
            ax.set_ylabel(ylabel,labelpad=8)
            ax.set_xticks(range(3),['Random','GPT','Mixed'])
            ax.set_xlim(-.20,2.20)
            ax.tick_params(axis='x',labelbottom=i>=4,length=0,pad=10)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1 if metric == 'controlled_exact_target_rate' else 0,symbol=''))
            ax.grid(axis='y',color='#E5E9ED',lw=.7)
            ax.set_axisbelow(True)
        fig.legend(handles=[Line2D([],[],color=c,marker='o',lw=1.4,label=l)
                            for c,l in zip(COLORS,['Random only','GPT-generated only','Random + GPT-generated'])],
                   loc='upper center',bbox_to_anchor=(.53,.988),ncol=3,
                   frameon=False,fontsize=17,columnspacing=2)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUTPUT.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.07)
        fig.savefig(OUTPUT.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=.07)
        plt.close(fig)
    if args.install:
        dest=PAPER/'figures/dataset_type_sweep.pdf'
        shutil.copy2(OUTPUT.with_suffix('.pdf'),dest)
        path=PAPER/'evidence/selected_panel_assets.json'
        assets=json.loads(path.read_text())
        assets[dest.name]=dict(source=str(OUTPUT.with_suffix('.pdf')),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator=str(Path(__file__).resolve()),
            data=[str(RAW),str(SUMMARY)],
            data_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW,SUMMARY]],
            original_figure_sha256='36fcd51bf42be23c431c0e8c5fcb13c0f88a5e0b379d37b039aa085287bd8c7e')
        path.write_text(json.dumps(assets,indent=2)+'\n')
    print(OUTPUT.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
