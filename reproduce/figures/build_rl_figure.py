"""Render aggregate RL panels; optionally add a locally supplied example panel."""
from pathlib import Path
import argparse, difflib, hashlib, json, os, re, shutil
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ces-figure3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT=ROOT/'inputs/discussion'
PDF=ROOT/'output/pdf'
SOURCE=ROOT/'inputs/filtered-summary.json'
INK='#263449'; MUTED='#69788C'; GRID='#E7ECF1'; LIGHT='#F3F5F8'
UNCHANGED='#CDD5DE'; CES='#326EA8'; IMPROVED='#44988B'
NEUTRAL='#D6AD63'; WORSENED='#C6606C'; HUMAN='#69788C'
INSERT='#1A7F37'; DELETE='#CF222E'
TITLE=14; LABEL=12; TICK=11; LEGEND=10.5; EXAMPLE=14
STYLE={'font.family':'Lato','font.size':TICK,'text.color':INK,
    'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':INK,
    'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#C8D1DC','pdf.fonttype':42}
data=json.loads(SOURCE.read_text())['tasks']['controlled'];final=data['retained_d24']
N=final['n'];CHANGED=final['changed_only']['n'];UNCH=N-CHANGED
COUNTS=[final['changed_only']['direction_counts'][k] for k in ['improves','same','worse']]
assert (N,CHANGED,COUNTS)==(5000,880,[601,141,138])
OC=[IMPROVED,NEUTRAL,WORSENED];OL=['Improved','Neutral','Worsened']

def count_bars(ax, rows):
    """Stack outcome counts inside a rounded silhouette; total height is n."""
    selected=[r for r in rows if r['mode'] in ['plain','-1','-0.5','0','0.5','0.75','1']]
    n=len(selected);upper=3500
    ax.set_xlim(-.6,n-.4);ax.set_ylim(0,upper)
    for i,r in enumerate(selected):
        x=(i-.34+.6)/(n+.2);w=.68/(n+.2);height=r['changed']/upper
        radius=min(.007,height/3,w/3)
        clip=FancyBboxPatch((x,0),w,height,boxstyle=f'round,pad=0,rounding_size={radius}',
            mutation_aspect=ax.bbox.width/ax.bbox.height,transform=ax.transAxes,facecolor='none',edgecolor='none')
        ax.add_patch(clip);bottom=0
        for field,color in [('improves',IMPROVED),('neutral',NEUTRAL),('worsens',WORSENED),('pending',UNCHANGED)]:
            value=r[field]
            if not value:continue
            patch=Rectangle((x,bottom/upper),w,value/upper,transform=ax.transAxes,facecolor=color,edgecolor='none',zorder=3)
            patch.set_clip_path(clip);ax.add_patch(patch)
            if value>=240:
                ax.text(i,bottom+value/2,str(value),ha='center',va='center',fontsize=8.58,color=INK if color in [NEUTRAL,UNCHANGED] else 'white',fontweight='bold',zorder=4)
            bottom+=value
        assert bottom==r['changed']
        ax.annotate(f"n={r['changed']:,}",(i,r['changed']),xytext=(0,6+11*(i%2)),textcoords='offset points',ha='center',fontsize=9,color=INK)
    ax.set_xticks(range(n),[r['mode'].replace('plain','Greedy') for r in selected],fontsize=10.4)
    ax.set_yticks([0,1000,2000,3000]);ax.set_ylabel('Changed GEC outputs (n)',fontsize=11.7)
    ax.set_xlabel('Greedy or margin threshold',fontsize=11.7)
    handles=[Rectangle((0,0),1,1,facecolor=color,label=label) for label,color in [('Improved',IMPROVED),('Neutral',NEUTRAL),('Worsened',WORSENED)]]
    if any(r['pending'] for r in selected):handles.append(Rectangle((0,0),1,1,facecolor=UNCHANGED,label='Awaiting judge'))
    ax.legend(handles=handles,frameon=False,fontsize=9.1,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.24),columnspacing=1.1,handlelength=1.5)
    return selected

def unfiltered_panels(axes, rows):
    a,b,c=axes
    rs=[r for r in rows if r['mode'] not in ['plain','forced'] and float(r['mode'])>=-1]
    x=np.array([float(r['mode']) for r in rs]);plain=next(r for r in rows if r['mode']=='plain')
    a.plot(x,[r['coverage'] for r in rs],'-o',color=CES,ms=3,label='GEC changed',lw=1.8)
    a.plot(x,[r['clean_change_rate'] for r in rs],'-s',color=WORSENED,ms=3,label='Clean changed',lw=1.8)
    a.axhline(plain['coverage'],color=CES,ls='--',lw=1,label='Greedy GEC')
    a.axhline(plain['clean_change_rate'],color=WORSENED,ls='--',lw=1,label='Greedy clean')
    a.axvline(.5,color=MUTED,ls=':',lw=.8)
    a.set_xlim(-1,3);a.set_ylim(0,70);a.set_xlabel('First-decision margin threshold',fontsize=11.7);a.set_ylabel('Inputs changed (%)',fontsize=11.7)
    a.legend(frameon=False,fontsize=9.1,loc='upper right')
    count_bars(b,rows)
    if all(r['net_pp'] is not None for r in rs):
        values=np.array([r['net_pp'] for r in rs]);ci=np.array([r['net_ci95'] for r in rs])
        c.plot(x,values,'-o',color=CES,ms=3,lw=1.8)
        c.fill_between(x,ci[:,0],ci[:,1],color=CES,alpha=.13,lw=0)
        c.axhline(plain['net_pp'],color=MUTED,ls='--',lw=1,label=f"Greedy: {plain['net_pp']:+.2f} pp")
        c.legend(frameon=False,fontsize=9.1,loc='lower right')
    else:
        c.text(.5,.5,'Awaiting complete\nBatch judgments',transform=c.transAxes,ha='center',color=MUTED)
        c.set_yticks([])
    c.axhline(0,color=MUTED,lw=.7);c.set_xlim(-1,3)
    c.set_xlabel('First-decision margin threshold',fontsize=11.7);c.set_ylabel('Improved − worsened (pp)',fontsize=11.7)
    for ax,title in zip(axes,['A  Editing coverage','B  Judged outcome counts','C  Net judged improvement']):
        ax.set_title(title,loc='left',fontsize=TITLE,pad=12)
        ax.set_axisbelow(True);ax.grid(axis='y',color=GRID,lw=.6);ax.tick_params(labelsize=10.4)
    assert a.get_xlim()[0]==c.get_xlim()[0]==-1

def head(ax,text,pad=12):ax.set_title(text,loc='left',fontsize=14.56,pad=pad,bbox=dict(facecolor='white',edgecolor='none',pad=1.5))

def rounded(ax,x,y,w,h,color,r=.12,edge='none',lw=0,transform=None,z=3):
    # Draw in axes coordinates so corner radii are independent of data scales.
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        mutation_aspect=ax.bbox.width/ax.bbox.height,facecolor=color,edgecolor=edge,
        linewidth=lw,transform=transform or ax.transAxes,clip_on=False,zorder=z)
    ax.add_patch(p);return p

def axis100(ax,labels):
    ax.set_xlim(0,100);ax.set_ylim(len(labels)-.5,-.5);ax.set_yticks(range(len(labels)),labels)
    ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',color=GRID,lw=.7);ax.set_axisbelow(True)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=9,labelsize=11.7)
    ax.tick_params(axis='x',length=3,labelsize=11.05)

def outcome_bars(ax,all_inputs=False,tracks=False):
    labels=['Unchanged','Improved','Neutral change','Worsened'] if all_inputs else OL
    counts=[UNCH,*COUNTS] if all_inputs else COUNTS
    cols=[UNCHANGED,*OC] if all_inputs else OC
    den=N if all_inputs else CHANGED
    axis100(ax,labels)
    n=len(labels);h=.50/n
    for i,(count,color) in enumerate(zip(counts,cols)):
        value=count/den*100;cy=1-(i+.5)/n
        if tracks:rounded(ax,0,cy-h/2,1,h,LIGHT,r=.006,z=1)
        rounded(ax,0,cy-h/2,value/100,h,color,r=min(.006,value/400))
        if all_inputs:
            ax.text(value+1.6,i,f'{value:.2f}%  ·  n={count:,}',va='center',fontsize=11.05,zorder=5)
        else:
            ax.text(value/2,i,f'n={count}',ha='center',va='center',fontsize=11.7,color='white' if i!=1 else INK,fontweight='bold',zorder=5)
            offset=value
            if i in (0,2):
                key='semantic_precision' if i==0 else 'harm_rate';stat=final['changed_only'][key]
                lo,hi=np.array(stat['bootstrap_95ci'])*100
                ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='none',color=INK,capsize=3,lw=1,zorder=6);offset=hi
            ax.text(offset+1.7,i,f'{value:.2f}%',va='center',fontsize=11.05,zorder=5)
    ax.set_xlabel('All 5,000 inputs (%)' if all_inputs else 'Changed outputs (%)',fontsize=11.7,labelpad=7)

def net(ax):
    head(ax,'D  Net judged improvement')
    for y,key,color in [(0,'retained_d24',CES),(1,'human_anchor',HUMAN)]:
        s=data[key]['direction_score'];v=s['mean']*100;lo,hi=np.array(s['bootstrap_95ci'])*100
        ax.plot([0,v],[y,y],color=color,alpha=.18,lw=4,solid_capstyle='round',zorder=1)
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=color,markersize=6,capsize=3,lw=1.1,zorder=3)
        ax.text(hi+2,y,f'+{v:.2f}',fontsize=12.35,color=color,fontweight='bold',va='center')
    ax.set_xlim(0,80);ax.set_ylim(1.5,-.5);ax.set_yticks([0,1],['CES · 17.6% edited','Human · 100% edited'],fontsize=11.05)
    ax.set_xticks([0,20,40,60,80]);ax.tick_params(axis='both',length=0,labelsize=10.4);ax.grid(axis='x',color=GRID,lw=.7)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ax.set_xlabel('Improved − worsened (percentage points)',fontsize=11.05,labelpad=6)


def diff_spans(source, prediction):
    """Token-level inline diff; both original strings reconstruct exactly."""
    tokenize=lambda text: re.findall(r'\s+|\w+|[^\w\s]',text)
    a,b=tokenize(source),tokenize(prediction)
    spans=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag=='equal': spans.append((''.join(a[i:j]),'same'))
        else:
            if i!=j: spans.append((''.join(a[i:j]),'delete'))
            if k!=l: spans.append((''.join(b[k:l]),'insert'))
    assert ''.join(t for t,s in spans if s!='insert')==source
    assert ''.join(t for t,s in spans if s!='delete')==prediction
    return spans

def examples(fig, ax, selected):
    """Lay out coloured inline text and strike-throughs directly on the canvas."""
    fig.canvas.draw()
    width=ax.get_position().width*fig.get_figwidth()*72
    height=ax.get_position().height*fig.get_figheight()*72
    ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
    renderer=fig.canvas.get_renderer()
    fonts={s:FontProperties(family='Lato',size=EXAMPLE,weight='bold' if s=='insert' else 'normal') for s in ['same','insert','delete']}
    def measure(t,s):
        return renderer.get_text_width_height_descent(t,fonts[s],ismath=False)[0]*72/fig.dpi
    x0=24;right=width-7;leading=EXAMPLE*1.10;y=9+EXAMPLE
    sections=[]
    for group,color in zip(OL,OC):
        start=y-EXAMPLE-5
        for item in [r for r in selected if r['panel']==group]:
            x=x0
            previous=None
            for text,state in diff_spans(item['source_text'],item['prediction_text']):
                if previous=='delete' and state=='insert': x+=3
                for chunk in re.findall(r'\s+|\S+',text):
                    w=measure(chunk,state)
                    if x+w>right and not chunk.isspace():
                        x=x0+10;y+=leading
                    if chunk.isspace() and x+w>right: continue
                    ax.text(x,y,chunk,fontproperties=fonts[state],color={'same':INK,'insert':INSERT,'delete':DELETE}[state],va='baseline')
                    if state=='delete' and not chunk.isspace():
                        ax.plot([x,x+w],[y-EXAMPLE*.31]*2,color=DELETE,lw=.65)
                    x+=w
                previous=state
            y+=leading+.35
        end=y-leading+5
        sections.append((group,color,start,end))
        y+=8
    bottom=sections[-1][3]+6
    assert bottom<=height, (bottom,height)
    frame=FancyBboxPatch((.5,1),width-1,bottom,boxstyle='round,pad=0,rounding_size=7',facecolor='none',edgecolor='#CBD5E1',lw=.8)
    ax.add_patch(frame)
    for i,(group,color,start,end) in enumerate(sections):
        ax.text(11,(start+end)/2,group,rotation=90,ha='center',va='center',color=color,fontsize=11)
        if i<2:
            sep=(end+sections[i+1][2])/2
            ax.plot([6,width-6],[sep]*2,color='#D9E0E7',lw=.6)
    return bottom

@plt.rc_context(STYLE)
def render(examples_path=None):
    summary=json.loads((OUT/'unfiltered_confirmation/summary.json').read_text())
    assert summary['quality_status']=='complete' and summary['unique_judgments_complete']==2983
    selected = None
    if examples_path is not None:
        selected = json.loads(examples_path.read_text())['examples']
        assert [sum(r['panel'] == group for r in selected) for group in OL] == [5, 3, 4]
    PDF.mkdir(parents=True, exist_ok=True)
    fig=plt.figure(figsize=(12.6,8.8),facecolor='white')
    axes=[fig.add_axes([x,.66,.245,.27]) for x in [.07,.398,.726]]
    unfiltered_panels(axes,summary['results'])
    d=fig.add_axes([.09,.32,.305,.17]);outcome_bars(d,False,tracks=True)
    d.set_title('D  Within the 880 changed outputs',loc='left',fontsize=TITLE,pad=12)
    d.set_xlabel('Changed outputs (%)',fontsize=LABEL,labelpad=5)
    d.tick_params(labelsize=TICK)
    for t in d.texts:t.set_fontsize(11)
    e=fig.add_axes([.07,.065,.14,.125]);e.set_xlim(0,1);e.set_ylim(0,1);e.axis('off')
    e.set_title('E  Clean-input false edits',loc='left',fontsize=12,pad=10)
    e.text(0,.79,'1.48%',fontsize=18,fontweight='bold',color=WORSENED,va='center',transform=e.transAxes)
    e.text(0,.46,'n=74 / 5,000 changed',fontsize=10,color=INK,transform=e.transAxes)
    rounded(e,0,.25,1,.105,UNCHANGED,r=.003)
    rounded(e,0,.25,.0148,.105,WORSENED,r=.001)
    e.text(0,.02,'98.52% retained',fontsize=10,color=MUTED,transform=e.transAxes)
    f=fig.add_axes([.29,.07,.13,.12]);net(f)
    f.set_title('F  Net judged\nimprovement',loc='left',fontsize=12,pad=10)
    f.set_yticks([0,1],['CES · 17.6%','Human · 100%'],fontsize=9.5)
    f.set_xticks([0,40,80]);f.tick_params(axis='x',labelsize=10)
    f.set_xlabel('Improved − worsened (pp)',fontsize=9.5,labelpad=4)
    for t in f.texts:t.set_fontsize(11)
    if selected is not None:
        g = fig.add_axes([.44, .01, .55, .48])
        g.set_title('G  Example edits', loc='left', fontsize=TITLE, pad=12)
        examples(fig, g, selected)
        name = 'rl_filtered_unfiltered_with_examples_wrapped'
    else:
        fig.set_size_inches(12.6, 7.0)
        d.set_position([.07, .13, .245, .27])
        e.set_position([.398, .13, .245, .27])
        f.set_position([.726, .13, .245, .27])
        name = 'rl_filtered_unfiltered_aggregates'
    fig.savefig(PDF/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(PDF/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    return PDF/(name+'.pdf')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--examples', type=Path, help='Local example-selection JSON; source text is not bundled.')
    args=parser.parse_args()
    source=render(args.examples)
    if args.install:
        paper=ROOT/'inputs/paper'
        shutil.copyfile(source,paper/'figures'/source.name)
        p=paper/'evidence/selected_panel_assets.json';manifest=json.loads(p.read_text())
        manifest[source.name].update(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generator=str(Path(__file__).resolve()))
        p.write_text(json.dumps(manifest,indent=2)+'\n')
    print(source)

if __name__=='__main__': main()
