"""Figures 7 and 8: archived corruption and temperature sweeps, shared theme."""
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
from matplotlib.ticker import MaxNLocator

PAPER=ROOT/'inputs/paper'
EVIDENCE=PAPER/'evidence'
OUTPUT=ROOT/'output/pdf'
BLUE,TEAL,PURPLE='#326EA8','#44988B','#8874B3'
THEME={**STYLE,'font.size':16,'axes.labelsize':16,'axes.titlesize':18,
       'xtick.labelsize':15,'ytick.labelsize':15}
RHO_METRICS=[
    ('val_bpb','A  Clean language modelling','Bits per byte (lower is better)'),
    ('human_position_accuracy','B  CoEdIT localisation','Position accuracy'),
    ('controlled_validity','C  Valid edit programs','Valid-program rate'),
    ('controlled_exact','D  Exact repair','Exact-target rate'),
    ('gec_f0.5','E  CoEdIT correction','ERRANT F0.5'),
    ('gec_harm','F  Reference-based harm','Single-reference harm rate'),
]


def polish(ax):
    ax.grid(axis='y',color='#E5E9ED',lw=.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def save(fig,name,install,inputs,original_hash):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/(name+'.pdf')
    fig.savefig(path,bbox_inches='tight',pad_inches=.07)
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
    if install:
        dest=PAPER/'figures'/path.name
        shutil.copy2(path,dest)
        record=EVIDENCE/'selected_panel_assets.json'
        assets=json.loads(record.read_text())
        assets[path.name]=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            generator=str(Path(__file__).resolve()),data=[str(p) for p in inputs],
            data_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs],
            original_figure_sha256=original_hash)
        record.write_text(json.dumps(assets,indent=2)+'\n')
    print(path)


@plt.rc_context(THEME)
def rho(install):
    source=EVIDENCE/'rho_summary.json'; raw=EVIDENCE/'rho_runs.csv'
    data=json.loads(source.read_text());rows=list(csv.DictReader(raw.open()))
    assert len(data)==8 and len(rows)==24
    for r in data:
        samples=[v for v in rows if v['point']==r['point'] and v['phase']==r['phase']]
        assert len(samples)==3
        for metric in ['measured_rho']+[m[0] for m in RHO_METRICS]:
            values=[float(v[metric]) for v in samples]
            assert math.isclose(statistics.mean(values),r[metric+'_mean'],abs_tol=1e-12)
            assert math.isclose(statistics.stdev(values),r[metric+'_sd'],abs_tol=1e-12)
    fig,axs=plt.subplots(2,3,figsize=(13.8,8.5),sharex=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.10,top=.88,wspace=.32,hspace=.32)
    phases=[('pretrain','Pretrained',BLUE,'o'),('finetuned','After CoEdIT fine-tuning',PURPLE,'s')]
    for i,(ax,(metric,title,ylabel)) in enumerate(zip(axs.flat,RHO_METRICS)):
        for phase,label,color,marker in phases:
            group=sorted([r for r in data if r['phase']==phase],key=lambda r:r['control'])
            ax.errorbar([r['measured_rho_mean'] for r in group],[r[metric+'_mean'] for r in group],
                        yerr=[r[metric+'_sd'] for r in group],color=color,marker=marker,
                        ms=5,lw=1.8,elinewidth=1.1,capsize=3,alpha=.88)
        ax.set_title(title,loc='left',pad=13)
        ax.set_ylabel(ylabel,labelpad=8)
        ax.set_xticks([0,.01,.02],['0','0.01','0.02'])
        if i>=3: ax.set_xlabel('Measured training edit mass (ρ)',labelpad=8)
        polish(ax)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=1.8,label=l) for _,l,c,m in phases],
               loc='upper center',bbox_to_anchor=(.54,.985),ncol=2,frameon=False,fontsize=17)
    save(fig,'rho_tradeoffs',install,[source,raw],
         '85148cddda7d44416e36910f1cb2dd6299e46071830b8babca4ad2942af40607')


@plt.rc_context(THEME)
def temperature(install):
    source=EVIDENCE/'ces_temperature_summary.json'
    data=json.loads(source.read_text())
    policies=['greedy','temperature-0p2','temperature-0p5','temperature-0p8','temperature-1p0']
    specs=[('pretrained','validity','A  Valid edit programs','Valid-program rate'),
           ('pretrained','similarity_delta','B  Repair magnitude','Mean similarity change'),
           ('finetuned','f0_5','C  CoEdIT correction','ERRANT F0.5')]
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.8),sharex=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.79,wspace=.34)
    series=[(12,BLUE,'o'),(18,TEAL,'s'),(24,PURPLE,'^')]
    for ax,(phase,metric,title,ylabel) in zip(axs,specs):
        for depth,color,marker in series:
            group=[]
            for policy in policies:
                matches=[r for r in data if r['depth']==depth and r['phase']==phase and r['policy']==policy]
                assert len(matches)==1
                r=matches[0]
                assert r['decoding_seeds']==(1 if policy=='greedy' else 3)
                if policy=='greedy': assert r[metric+'_sd']==0
                group.append(r)
            ax.errorbar(range(5),[r[metric+'_mean'] for r in group],
                        yerr=[r[metric+'_sd'] for r in group],color=color,marker=marker,
                        ms=5,lw=1.8,elinewidth=1.1,capsize=3,alpha=.85)
        ax.set_title(title,loc='left',pad=13)
        ax.set_ylabel(ylabel,labelpad=8)
        ax.set_xticks(range(5),['Greedy','0.2','0.5','0.8','1.0'])
        ax.set_xlim(-.2,4.2)
        ax.set_xlabel('Decoding temperature',labelpad=9)
        polish(ax)
        if metric=='validity': ax.set_ylim(0,1)
        if metric=='f0_5': ax.set_ylim(bottom=0)
        if metric=='similarity_delta': ax.axhline(0,color='#C8D1DC',lw=.9,zorder=1)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=1.8,label=f'd{d}') for d,c,m in series],
               loc='upper center',bbox_to_anchor=(.54,.995),ncol=3,frameon=False,fontsize=17,columnspacing=3)
    save(fig,'sampling_policy_ces_scaling',install,[source],
         'b4dca6276deb3df848895ecaa5ca11c624727cf1a34fe46fe34a3152ea79f0ec')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true')
    args=p.parse_args();rho(args.install);temperature(args.install)


if __name__=='__main__':main()
