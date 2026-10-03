"""Figure 2: pretraining comparisons and CoEdIT representation lengths.

Uses retained metric tables; styling does not change the estimates or uncertainty.
Run from any directory with --install to refresh the manuscript asset.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ces-figure2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'inputs/discussion'
PDF = ROOT / 'output/pdf'
RUNS = OUT / 'method_runs_audited.csv'
CLEAN_RUNS = ROOT / 'inputs/paper/evidence/clean_ar_reference_runs.csv'
METHODS = ['ces','ar_dsl','flat_relative','regeneration','fim','fim_predicted','levenshtein','pretrained_ces']
LABELS = ['CES','AR-DSL','Flat-relative','Regeneration','Oracle-span FIM','Predicted-span FIM','Iterative Levenshtein','Adapted CES']
COLORS = ['#326EA8','#8874B3','#B08A57','#D07C58','#7F8A9B','#B9678D','#44988B','#8E9E61']
MARKERS = ['o','s','^','D','v','P','X','h']
STYLE = {
    'font.family': 'Lato', 'font.size': 19, 'text.color': '#263449',
    'axes.labelcolor': '#69788C', 'xtick.color': '#69788C',
    'ytick.color': '#263449', 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#C8D1DC', 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'hatch.linewidth': .65,
}

def save(fig, name):
    PDF.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def length_panel(ax, standalone=False, statistic='mean', tall=False):
    """Three clearly separated accounting conventions on the same CoEdit pairs."""
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    refs=json.loads((OUT/'dashboard_data/reference_lengths.json').read_text())['aggregate']
    observed=json.loads((OUT/'dashboard_data/observed_gec_lengths_seed42.json').read_text())
    opt={r['method']:r for r in json.loads((OUT/'dashboard_data/optimal_gec_lengths.json').read_text())['summary']}
    assert statistic in ('mean','median')
    if statistic=='median':
        reference_rows=list(csv.DictReader((OUT/'reference_lengths_per_example.csv').open()))
        optimal_rows=list(csv.DictReader((OUT/'optimal_gec_lengths_per_example.csv').open()))
        audit={}
        for method in METHODS:
            rr=[r for r in reference_rows if r['method']==method]
            oo=[r for r in optimal_rows if r['method']==method]
            assert len(rr)==len(oo)==485
            refs[method]['reference_tokens']=statistics.median(float(r['reference_tokens']) for r in rr)
            for key in ['prompt_at_min_program','min_program_tokens','min_total_tokens']:
                opt[method][key]=statistics.median(float(r[key]) for r in oo)
            observed[method]['mean_tokens']=observed[method]['median_tokens']
            audit[method]={'n':485,'reference_median':refs[method]['reference_tokens'],
                'observed_median':observed[method]['median_tokens'],
                'prompt_median':opt[method]['prompt_at_min_program'],
                'program_median':opt[method]['min_program_tokens'],
                'total_median':opt[method]['min_total_tokens']}
        (OUT/'dashboard_data/length_panel_medians.json').write_text(json.dumps(audit,indent=2)+'\n')
    fs=8.5 if standalone else (18 if tall else 6.8)
    value_align = 'right' if tall else 'left'
    for i,(method,color) in enumerate(zip(METHODS,COLORS)):
        value=refs[method]['reference_tokens'];lev=method=='levenshtein'
        ax.barh(i-.25,value,height=.20,color=color,alpha=.9,hatch='//' if lev else None,edgecolor='#263449' if lev else 'none',lw=0)
        ax.text(value+1.5,i-.25,f'{value:.1f}',ha=value_align,va='center',fontsize=fs)
        obs=observed[method]['mean_tokens']
        if obs is not None:
            ax.barh(i,obs,height=.20,facecolor='white',edgecolor=color,lw=1.1)
            ax.text(obs+1.5,i,f'{obs:.1f}' if obs>=.05 or statistic=='median' else '<0.1',ha=value_align,va='center',fontsize=fs,color=color)
        else:
            ax.text(1,i,'0 inserts / 2 passes',fontsize=fs-.5,ha=value_align,va='center',color='#65758B')
        r=opt[method];p=r['prompt_at_min_program'];v=r['min_program_tokens']
        if statistic=='median':
            # Component medians are non-additive: plot the median paired total.
            total=r['min_total_tokens']
            ax.barh(i+.25,total,height=.20,facecolor=color,alpha=.6,edgecolor='#263449',hatch='////',lw=0)
            ax.text(total+1.5,i+.25,f'{total:.1f}',ha=value_align,va='center',fontsize=fs)
            continue
        ax.barh(i+.25,p,height=.20,color='#D5DAE1',edgecolor='#65758B' if lev else 'none',hatch='//' if lev else None,lw=0)
        ax.barh(i+.25,v,left=p,height=.20,facecolor=color,alpha=.6,edgecolor='#263449',hatch='////',lw=0)
        ax.text(p+v+1.5,i+.25,f'{p+v:.1f}',ha=value_align,va='center',fontsize=fs)
        if standalone:
            ax.text(p/2,i+.25,f'{p:.1f}',ha='center',va='center',fontsize=6.5)
            ax.text(p+v/2,i+.25,f'{v:.1f}',ha='center',va='center',fontsize=6.5,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.1))
    ax.set_yticks(range(8),['CES','AR-DSL','Flat-relative','Regen.','Oracle FIM','Pred. FIM','Iter. Lev.*','Adapted CES'],fontsize=9 if standalone else (19 if tall else 8))
    ax.set_ylim(7.65,-.65);ax.set_xlim(190,0) if tall else ax.set_xlim(0,180);ax.set_xticks([0,50,100,150]);ax.set_xlabel(f'{statistic.capitalize()} tokens\n(prompt / output as labelled)' if tall else f'{statistic.capitalize()} tokens (prompt / output as labelled)',fontsize=10 if standalone else (19 if tall else 8))
    ax.set_title('G  CoEdIT representation /\noutput length' if tall else 'G  CoEdit representation / output length',loc='right' if tall else 'left',fontsize=21 if tall else (11 if standalone else 10),pad=14)
    if tall:
        ax.tick_params(axis='x',labelsize=19)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.xaxis.label.set_horizontalalignment('right')
        ax.xaxis.set_label_coords(1, -.055)
        ax.set_xlabel(f'{statistic.capitalize()} tokens\n(prompt / output as labelled)', fontsize=19, ha='right')
    ax.grid(axis='x',color='#E5E9ED',lw=.65);ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor='#7C8796'),Patch(facecolor='white',edgecolor='#7C8796'),
        (Patch(facecolor='#D5DAE1'),Patch(facecolor='#7C8796',hatch='////')) if statistic=='mean' else Patch(facecolor='#7C8796',edgecolor='#263449',hatch='////')],
        labels=['Reference encoding\n(output only)','Observed d24 seed 42,\nafter SFT','Prompt + minimum\nprogram'] if tall else ['Reference encoding (output only)','Observed d24 seed 42, after SFT','Prompt + minimum program'],handler_map={tuple:HandlerTuple(ndivide=None)},
        loc='upper right' if tall else 'upper left',bbox_to_anchor=(1 if tall else 0,-.15 if standalone else (-.12 if tall else -.24)),borderaxespad=0,frameon=False,fontsize=9 if standalone else (18 if tall else 7.5))

@plt.rc_context(STYLE)
def scaling(include_lengths=False, name='method_scaling_discussion'):
    rows = [r for r in csv.DictReader(RUNS.open()) if r['phase']=='pretrain']
    clean_rows = list(csv.DictReader(CLEAN_RUNS.open()))
    assert {(int(r['depth']), int(r['model_seed'])) for r in clean_rows} == {(d,s) for d in (12,18,24) for s in (42,43,44)}
    metrics = [('val_bpb','A  Clean language modelling','Bits per byte (lower is better)',1),
        ('core','B  General capabilities','CORE score (higher is better)',1),
        ('controlled_valid_rate','C  Complete-program validity','Valid outputs (%)',100),
        ('controlled_exact_rate','D  Exact repair','Exact target matches (%)',100),
        ('controlled_improvement_rate','E  Repair direction','Outputs closer to target (%)',100),
        ('controlled_char_delta','F  Repair magnitude','Mean character-similarity change\n(symmetric log scale)',1)]
    fig = plt.figure(figsize=(17.0 if include_lengths else 12.6,7.8 * 1.30 * 1.20))
    grid = fig.add_gridspec(2,4 if include_lengths else 3)
    axs=[]
    for row in [0,1]:
        for col in [0,1,2]:
            axs.append(fig.add_subplot(grid[row,col],sharex=axs[0] if axs else None))
    # Keep A's outer panel geometry while allocating most of its height to
    # the tightly clustered language models. Both segments are linear.
    bpb_low = axs[0].inset_axes([0, 0, 1, .52], sharex=axs[0])
    bpb_high = axs[0].inset_axes([0, .60, 1, .40], sharex=axs[0])
    for ax,(key,title,ylabel,factor) in zip(axs,metrics):
        plot_axes = [bpb_low, bpb_high] if key == 'val_bpb' else [ax]
        for method,color,marker in zip(METHODS,COLORS,MARKERS):
            xs=[];ys=[];es=[]
            for depth in [12,18,24]:
                vals=[float(r[key])*factor for r in rows if r['method']==method and int(r['depth'])==depth and r[key]]
                if vals:
                    assert len(vals)==3
                    xs.append(depth);ys.append(statistics.mean(vals));es.append(statistics.stdev(vals))
            if xs:
                for plot_ax in plot_axes:
                    plot_ax.errorbar(xs,ys,yerr=es,color=color,marker=marker,markersize=5.6,
                        linewidth=1.8,alpha=.68,
                        capsize=2,elinewidth=.8,linestyle='--' if method=='fim' else '-',zorder=2)
        # Clean AR: matched d12/d18 reruns and retained d24 reference; language metrics only.
        if key in ('val_bpb', 'core'):
            field = 'clean_bpb' if key == 'val_bpb' else 'clean_core'
            values = [[float(r[field]) for r in clean_rows if int(r['depth']) == d] for d in (12,18,24)]
            assert all(len(v) == 3 for v in values)
            for plot_ax in plot_axes:
                plot_ax.errorbar([12,18,24], [statistics.mean(v) for v in values],
                    yerr=[statistics.stdev(v) for v in values], color='#30343B',
                    marker='d', markersize=5.6, linewidth=1.8, alpha=.68,
                    capsize=2, elinewidth=.8, linestyle=':', zorder=3)
        ax.set_title(title.replace('Complete-program validity','Program validity'),loc='left',fontsize=21,pad=14)
        ax.set_ylabel(ylabel.replace('Mean character-similarity change','Mean similarity change'),fontsize=19)
        ax.set_xticks([12,18,24]);ax.set_xlim(11,25)
        ax.tick_params(labelsize=19)
        if ax in axs[:3]:
            ax.tick_params(axis='x',labelbottom=False)
        else:
            ax.set_xlabel('Transformer depth',fontsize=19)
        ax.grid(axis='y',color='#E5E9ED',lw=.65);ax.set_axisbelow(True)
        if key=='val_bpb':
            ax.grid(False)
            ax.patch.set_visible(False)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
            ax.text(-.33,.5,ylabel,transform=ax.transAxes,rotation=90,
                    va='center',ha='center',fontsize=19,color=STYLE['axes.labelcolor'])
            bpb_low.set_ylim(.74,.90)
            bpb_low.set_yticks([.75,.80,.85,.90],labels=['0.75','0.80','0.85','0.90'])
            bpb_high.set_ylim(.92,5.2)
            bpb_high.set_yticks([1,2,3,4,5])
            bpb_high.spines['bottom'].set_visible(False)
            for segment in (bpb_low,bpb_high):
                segment.tick_params(axis='x',bottom=False,labelbottom=False)
                segment.tick_params(axis='y',labelsize=19)
                segment.grid(axis='y',color='#E5E9ED',lw=.65)
                segment.set_axisbelow(True)
            # Constant-size diagonal marks identify the two sides of the break.
            for segment,y in ((bpb_low,1),(bpb_high,0)):
                segment.plot([0],[y],transform=segment.transAxes,linestyle='none',
                    marker=[(-1,-.5),(1,.5)],markersize=10,markeredgewidth=1.2,
                    color='#69788C',clip_on=False,zorder=5)
        elif key=='controlled_char_delta':
            ax.set_yscale('symlog',linthresh=.002);ax.set_ylim(-.4,.016)
            ax.set_yticks([-.3,-.1,-.01,0,.01],labels=['-0.30','-0.10','-0.01','0','+0.01'])
            ax.axhline(0,color='#94A3B8',lw=.7)
        elif key=='controlled_valid_rate': ax.set_ylim(-6,118);ax.set_yticks([0,25,50,75,100])
        elif key=='controlled_exact_rate': ax.set_ylim(-.2,6.5)
        elif key=='controlled_improvement_rate': ax.set_ylim(-2,65)
    if include_lengths:
        ax=fig.add_subplot(grid[:,3])
        length_panel(ax,tall=True)
    handles=[Line2D([0],[0],color=c,marker=m,label=l,lw=1.8,alpha=.68,linestyle='--' if k=='fim' else '-') for k,l,c,m in zip(METHODS,LABELS,COLORS,MARKERS)]
    handles.append(Line2D([0],[0],color='#30343B',marker='d',label='Clean AR (A/B)',lw=1.8,alpha=.68,linestyle=':'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.385 if include_lengths else .5,.015),ncol=3,frameon=False,fontsize=19,columnspacing=1.1,handlelength=1.7)
    fig.subplots_adjust(left=.06 if include_lengths else .075,right=.985,top=.93,bottom=.24,wspace=.60 if include_lengths else .38,hspace=.25)
    if include_lengths:
        # Reclaim the space previously occupied by G's baseline labels.
        bounds = ax.get_position()
        left = axs[2].get_position().x1 + .028
        ax.set_position([left, bounds.y0, bounds.x1 - left, bounds.height])
    assert all(ax.get_shared_x_axes().joined(axs[0],ax) for ax in axs)
    assert all(not ax.get_xlabel() for ax in axs[:3])
    if include_lengths:
        assert len({tuple(round(x,6) for x in (axis.get_position().width,axis.get_position().height)) for axis in axs})==1
        assert abs(ax.get_position().y1-axs[0].get_position().y1)<1e-8
        assert abs(ax.get_position().y0-axs[-1].get_position().y0)<1e-8
    save(fig,name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    name = 'method_scaling_with_lengths'
    scaling(include_lengths=True, name=name)
    if args.install:
        paper = ROOT / 'inputs/paper'
        source = PDF / (name + '.pdf')
        shutil.copyfile(source, paper / 'figures' / source.name)
        manifest_path = paper / 'evidence/selected_panel_assets.json'
        manifest = json.loads(manifest_path.read_text())
        record = manifest[source.name]
        record.update(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), generator=str(Path(__file__).resolve()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(PDF / (name + '.pdf'))

if __name__ == '__main__':
    main()
