"""Figures 11 and 12: retained continuation estimates with shared paper styling."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
sys.dont_write_bytecode=True
from build_method_scaling import STYLE,ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PAPER=ROOT/'inputs/paper'
SOURCE=PAPER/'evidence/continuation_figure_data.json'
OUT=ROOT/'output/pdf'
CONDITIONS=[('clean_noop_program','Clean + no-op','#AEB9C7'),
            ('corrupt_noop_program','Corrupt + no-op','#326EA8'),
            ('corrupt_sham_program','Corrupt + sham','#B08A57'),
            ('corrupt_gold_program','Corrupt + repair','#44988B')]
THEME={**STYLE,'font.size':16,'axes.labelsize':16,'axes.titlesize':18,
       'xtick.labelsize':15,'ytick.labelsize':15}


def save(fig,name,install,old_hash):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(name+'.pdf')
    fig.savefig(path,bbox_inches='tight',pad_inches=.08)
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    if install:
        shutil.copy2(path,PAPER/'figures'/path.name)
        record=PAPER/'evidence/selected_panel_assets.json'
        assets=json.loads(record.read_text())
        assets[path.name]=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            generator=str(Path(__file__).resolve()),data=str(SOURCE),
            data_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),original_figure_sha256=old_hash)
        record.write_text(json.dumps(assets,indent=2)+'\n')
    print(path)


@plt.rc_context(THEME)
def overview(runs,install):
    fig,axs=plt.subplots(1,2,figsize=(13.2,5.5),gridspec_kw={'width_ratios':[1,1.12]})
    fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.80,wspace=.39)
    ax=axs[0];x=np.arange(2);width=.19
    for i,(condition,label,color) in enumerate(CONDITIONS):
        values=[]
        for run in runs:
            c=run['analysis']['overall']['conditions']
            values.append(1000*(c[condition]['mean_nll']-c['clean_noop_program']['mean_nll']))
        ax.bar(x+(i-1.5)*width,values,width*.93,color=color,alpha=.86,zorder=3)
    ax.set_xticks(x,[r['short_label'] for r in runs]);ax.tick_params(axis='x',length=0,pad=12)
    ax.set_ylabel('Excess tail NLL (millinats/token)',labelpad=9)
    ax.set_title('A  Continuation likelihood',loc='left',pad=15)
    ax.axhline(0,color='#AEB9C7',lw=1)
    ax.grid(axis='y',color='#E5E9ED',lw=.7);ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    mature=runs[1];assert mature['short_label']=='Mature CES'
    groups=[('operation:replace','Replace'),('operation:delete','Delete'),('operation:insert','Insert'),
            ('span_length:1','Span 1'),('span_length:4','Span 4'),('span_length:16','Span 16'),
            ('edit_distance:16','Distance 16'),('edit_distance:64','Distance 64'),('edit_distance:128','Distance 128')]
    y=np.array([0,1,2,3.5,4.5,5.5,7,8,9])
    values=[];low=[];high=[]
    for key,label in groups:
        m=mature['analysis'][key]['repair_gain_vs_noop']
        value=1000*m['estimate'];lo,hi=np.array(m['ci95'])*1000
        assert lo<=value<=hi
        values.append(value);low.append(value-lo);high.append(hi-value)
    ax=axs[1]
    ax.barh(y,values,height=.68,color='#44988B',alpha=.82,zorder=3)
    ax.errorbar(values,y,xerr=[low,high],fmt='none',ecolor='#263449',elinewidth=1.1,capsize=3,zorder=4)
    ax.set_yticks(y,[label for _,label in groups]);ax.invert_yaxis()
    ax.tick_params(axis='y',length=0,pad=8)
    ax.set_xlim(left=0)
    ax.set_title('B  Mature CES: repair gains',loc='left',pad=15)
    ax.set_xlabel('NLL gain (millinats/token)',labelpad=10)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis='x',color='#E5E9ED',lw=.7);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=c,alpha=.86,label=l) for _,l,c in CONDITIONS],
               loc='upper center',bbox_to_anchor=(.54,.99),frameon=False,ncol=4,fontsize=16,
               handlelength=1.2,columnspacing=1.6)
    save(fig,'post_edit_tail_overview',install,'a1b04209d59413ca714f6ada0c0cac13e3552dc2b01880b894ed1a1ed4cc7644')


@plt.rc_context(THEME)
def positions(runs,install):
    fig,axs=plt.subplots(1,2,figsize=(13.2,5.0),sharey=True,sharex=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.78,wspace=.12)
    styles=['-','--','-']
    for i,(ax,run) in enumerate(zip(axs,runs)):
        curves=run['smoothed_excess_nll_by_tail_position']
        for (condition,label,color),ls in zip(CONDITIONS[1:],styles):
            curve=curves[condition]
            x=np.asarray(curve['positions']);mean=1000*np.asarray(curve['mean'])
            low=1000*np.asarray(curve['ci95_low']);high=1000*np.asarray(curve['ci95_high'])
            assert len(x)==121 and np.all(low<=mean) and np.all(mean<=high)
            ax.plot(x,mean,color=color,lw=1.8,alpha=.9,ls=ls,zorder=3)
            ax.fill_between(x,low,high,color=color,alpha=.13,lw=0,zorder=2)
        ax.axhline(0,color='#AEB9C7',lw=1)
        ax.set_title(f'{chr(65+i)}  '+run['short_label'],loc='left',pad=14)
        ax.set_xlabel('Tail-token distance after edit-end',labelpad=10)
        ax.set_xlim(0,128);ax.set_xticks([0,32,64,96,128])
        ax.grid(axis='y',color='#E5E9ED',lw=.7);ax.set_axisbelow(True)
    axs[0].set_ylabel('Excess tail NLL (millinats/token)',labelpad=9)
    fig.legend(handles=[Line2D([],[],color=c,lw=1.8,ls=ls,label=l)
                        for (_,l,c),ls in zip(CONDITIONS[1:],styles)],
               loc='upper center',bbox_to_anchor=(.54,.99),frameon=False,ncol=3,fontsize=16,columnspacing=2.3)
    save(fig,'post_edit_tail_by_position',install,'9557aa26ec186274e0876b4ca67daa83ad032a40f8be81ece033d44c3cd12edc')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true')
    args=p.parse_args();runs=json.loads(SOURCE.read_text())['runs']
    assert [r['short_label'] for r in runs]==['Adapted CES','Mature CES']
    assert all(r['analysis']['overall']['examples']==540 for r in runs)
    overview(runs,args.install);positions(runs,args.install)


if __name__=='__main__':main()
