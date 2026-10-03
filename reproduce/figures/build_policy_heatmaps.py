"""Figure 9: depth-24 policy heatmaps using the shared paper theme."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
sys.dont_write_bytecode = True
from build_method_scaling import STYLE, ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

PAPER=ROOT/'inputs/paper'
SOURCE=PAPER/'evidence/depth24_policy_summary.json'
OUTPUT=ROOT/'output/pdf/sampling_policy_depth24'
METHODS=['ces','adapted-ces','ar-dsl','flat-relative','regeneration','oracle-span-fim','predicted-span-fim','iterative-levenshtein']
LABELS=['CES','Adapted CES','AR-DSL','Flat-relative','Regeneration','Oracle-span FIM','Predicted-span FIM','Iterative Levenshtein']
POLICIES=['greedy','temperature-0p2','temperature-0p5','temperature-0p8','temperature-1p0','nucleus-p0p9','nucleus-p0p95','top-k-20','top-k-50']
TICKS=['Greedy','T\n0.2','T\n0.5','T\n0.8','T\n1.0','p\n0.90','p\n0.95','k\n20','k\n50']


def palette(name, colors):
    cmap=LinearSegmentedColormap.from_list(name,colors)
    cmap.set_bad('#EDF0F4')
    return cmap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');args=p.parse_args()
    data=json.loads(SOURCE.read_text())
    lookup={(r['method'],r['phase'],r['policy']):r for r in data}
    assert len(lookup)==len(data)
    blue=palette('ces_blue',['#F5F7FA','#A9C2D9','#326EA8'])
    teal=palette('ces_teal',['#F5F7FA','#B3D3CC','#368477'])
    rose=palette('ces_rose',['#F5F7FA','#E5BDC2','#AD5963'])
    diverging=palette('ces_direction',['#AD5963','#F7F8FA','#44988B'])
    specs=[('pretrained','validity_mean','A  Controlled validity',blue,Normalize(0,1),[0,.5,1],2),
           ('pretrained','similarity_delta_mean','B  Repair magnitude',diverging,TwoSlopeNorm(vmin=-.75,vcenter=0,vmax=.03),[-.75,-.375,0,.03],3),
           ('finetuned','f0_5_mean','C  CoEdIT ERRANT F0.5',teal,Normalize(0,.12),[0,.06,.12],3),
           ('finetuned','harm_rate_mean','D  CoEdIT reference-based harm',rose,Normalize(0,1),[0,.5,1],2)]
    with plt.rc_context({**STYLE,'font.size':16,'axes.titlesize':19,'xtick.labelsize':14,'ytick.labelsize':16}):
        fig=plt.figure(figsize=(16,11.5),facecolor='white')
        grid=fig.add_gridspec(2,2,left=.155,right=.97,bottom=.105,top=.94,wspace=.15,hspace=.16)
        first=None
        for idx,(phase,metric,title,cmap,norm,ticks,decimals) in enumerate(specs):
            ax=fig.add_subplot(grid[idx//2,idx%2],sharex=first,sharey=first)
            if first is None:first=ax
            values=np.full((len(METHODS),len(POLICIES)),np.nan)
            for i,method in enumerate(METHODS):
                for j,policy in enumerate(POLICIES):
                    r=lookup.get((method,phase,policy))
                    if r and r.get(metric) is not None:
                        assert r['decoding_seeds']==(1 if policy=='greedy' else 3)
                        values[i,j]=r[metric]
            im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=norm,aspect='auto')
            ax.set_title(title,loc='left',pad=15)
            ax.set_yticks(range(8),LABELS)
            ax.set_xticks(range(9),TICKS)
            ax.tick_params(axis='both',length=0,pad=9,labelleft=idx%2==0,labelbottom=idx>=2)
            ax.set_xticks(np.arange(-.5,9,1),minor=True)
            ax.set_yticks(np.arange(-.5,8,1),minor=True)
            ax.grid(which='minor',color='white',lw=1.4)
            ax.tick_params(which='minor',bottom=False,left=False)
            for spine in ax.spines.values():spine.set_visible(False)
            for i in range(8):
                for j in range(9):
                    value=values[i,j]
                    if not np.isfinite(value):
                        label='—';color='#8B96A5'
                    else:
                        rounded=round(float(value),decimals)
                        label=f'{rounded:+.{decimals}f}' if metric=='similarity_delta_mean' and rounded!=0 else f'{abs(rounded) if rounded==0 else rounded:.{decimals}f}'
                        r,g,b,_=cmap(norm(value))
                        # Pick text contrast from the rendered cell colour.
                        luminance=.2126*r+.7152*g+.0722*b
                        color='white' if luminance<.56 else '#263449'
                    ax.text(j,i,label,ha='center',va='center',fontsize=13,color=color)
            cbar=fig.colorbar(im,ax=ax,fraction=.025,pad=.025,aspect=30,ticks=ticks)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=12,length=2,color='#C8D1DC')
            if metric=='similarity_delta_mean':
                cbar.ax.set_yticklabels(['−0.75','−0.375','0','+0.03'])
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUTPUT.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.09)
        fig.savefig(OUTPUT.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=.09)
        plt.close(fig)
    if args.install:
        dest=PAPER/'figures'/OUTPUT.with_suffix('.pdf').name
        shutil.copy2(OUTPUT.with_suffix('.pdf'),dest)
        path=PAPER/'evidence/selected_panel_assets.json'
        assets=json.loads(path.read_text())
        assets[dest.name]=dict(source=str(OUTPUT.with_suffix('.pdf')),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator=str(Path(__file__).resolve()),
            data=str(SOURCE),data_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            original_figure_sha256='f16ec3feb3b60b483e33ae0ce0d5861b69504fef4602b0ce0da9e0705dd255dc')
        path.write_text(json.dumps(assets,indent=2)+'\n')
    print(OUTPUT.with_suffix('.pdf'))


if __name__=='__main__':main()
