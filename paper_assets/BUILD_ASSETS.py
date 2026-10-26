"""Rebuild tables/figures from frozen reports; no training or scientific selection."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent
I=ROOT/'inputs';T=ROOT/'tables';F=ROOT/'figures'
V=('none','channel','temporal','full'); COLORS=['#416b86','#c18a27','#6b7280','#b84960']
def read(n):return json.loads((I/n).read_text(encoding='utf-8'))
def csvout(name,rows):
 with (T/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
 for ext in ('svg','pdf','png'):fig.savefig(F/(name+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 T.mkdir(parents=True,exist_ok=True);F.mkdir(parents=True,exist_ok=True)
 ab=read('ablation_d25_summary.json');ft=read('feature_summary.json');pr=read('primary_reference.json');tr=read('transfer_reference.json')
 assert len(ab['jobs'])==60 and len(ft['jobs'])==120 and len(pr['jobs'])==12 and len(tr['jobs'])==12
 csvout('01_ablation_cv_d25.csv',ab['comparison']['table'])
 csvout('02_ablation_paired_folds.csv',ab['comparison']['paired_folds'])
 rows=[];binary=[]
 for dataset,source in [('primary_official_test',pr),('cross_subject_separate_recipe',tr)]:
  for a,b in sorted({(j['source'],j['target']) for j in source['jobs']}):
   for v in ('none','full'):
    js=sorted([j for j in source['jobs'] if (j['source'],j['target'],j['variant'])==(a,b,v)],key=lambda j:j['seed'])
    counts=[j['character_curve'][-1]['correct'] for j in js];assert all(j['character_curve'][-1]['characters']==100 for j in js)
    rows.append({'protocol':dataset,'source':a,'target':b,'variant':v,'seeds':json.dumps([j['seed'] for j in js]),'correct_out_of_100':json.dumps(counts),'accuracy15_mean_percent':float(np.mean(counts)),'seed_sample_sd_pp':float(np.std(counts,ddof=1)),'mean_accuracy_1_to_15_percent':100*float(np.mean([r['accuracy'] for j in js for r in j['character_curve']]))})
    for j in js:
     binary.append({'protocol':dataset,'source':a,'target':b,'variant':v,'seed':j['seed'],**{k:j['binary'][k] for k in ('roc_auc','average_precision','accuracy','target_precision','target_recall','target_f1','unweighted_bce')}})
 csvout('03_official_test_and_transfer_SEPARATE_PROTOCOLS.csv',rows);csvout('04_flash_metrics_by_seed.csv',binary)
 metrics=[];summary=[];bykey={}
 for j in ft['jobs']:
  metrics.append({k:j[k] for k in ('subject','variant','seed','fold','silhouette_48d','silhouette_tsne_2d')});bykey[j['subject'],j['variant'],j['seed'],j['fold']]=j
 for sub in 'AB':
  for v in V:
   seedmeans=[float(np.mean([bykey[sub,v,se,f]['silhouette_48d'] for f in range(1,6)])) for se in (42,123,2026)]
   summary.append({'subject':sub,'variant':v,'mean_within_fold_silhouette_48d':float(np.mean(seedmeans)),'sample_sd_seed_means':float(np.std(seedmeans,ddof=1)),'seed42':seedmeans[0],'seed123':seedmeans[1],'seed2026':seedmeans[2]})
 csvout('05_silhouette_all_models.csv',metrics);csvout('06_silhouette_summary.csv',summary)
 params=[{'variant':v,'total_parameters':n,'trainable_parameters':n-80,'nontrainable_parameters':80} for v,n in zip(V,(2001,2129,3057,3185))];csvout('07_parameter_counts.csv',params)
 plt.rcParams.update({'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
 fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
 for ax,sub in zip(axs,'AB'):
  for v,color in zip(V,COLORS):
   per=ab['comparison']['by_subject'][sub][v]['per_seed'];curves=np.array([[r['accuracy'] for r in se['character_curve']] for se in per])*100
   ax.plot(range(1,16),curves.mean(0),label=v,color=color,linewidth=2)
  ax.set(title='Subject '+sub,xlabel='Repetitions',xticks=[1,3,5,7,9,11,13,15],ylim=(0,100));ax.grid(alpha=.2)
 axs[0].set_ylabel('OOF character accuracy (%)');axs[1].legend(frameon=False)
 fig.suptitle('Matched ablation | dropout 0.25 | mean of three seeds');fig.tight_layout();save(fig,'01_ablation_cv_curves')
 fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
 for ax,sub in zip(axs,'AB'):
  for v,color in [('none',COLORS[0]),('full',COLORS[-1])]:
   js=[j for j in pr['jobs'] if j['source']==sub and j['variant']==v];curves=np.array([[r['accuracy'] for r in j['character_curve']] for j in js])*100
   ax.plot(range(1,16),curves.mean(0),color=color,label=v,lw=2)
  ax.set(title='Subject '+sub,xlabel='Repetitions',xticks=[1,3,5,7,9,11,13,15],ylim=(0,100));ax.grid(alpha=.2)
 axs[0].set_ylabel('Official Test character accuracy (%)');axs[1].legend(frameon=False)
 fig.suptitle('Primary within-subject Test | mean of three seeds');fig.tight_layout();save(fig,'02_primary_test_curves')
 fig,axs=plt.subplots(2,4,figsize=(15,7.8))
 for i,sub in enumerate('AB'):
  for k,v in enumerate(V):
   with np.load(I/f'{sub}_{v}_42_fold1_tsne.npz',allow_pickle=False) as z:
    e,y=z['embedding'],z['truth'];assert e.shape==(1200,2)
   ax=axs[i,k]
   for label,color,name in [(0,'#627d98','Non-target'),(1,'#d1495b','Target')]:
    m=y==label;ax.scatter(e[m,0],e[m,1],s=5,alpha=.65,c=color,label=name,edgecolors='none')
   j=bykey[sub,v,42,1];ax.set_title(f"{sub} | {v}\nS48={j['silhouette_48d']:.4f}; S2D={j['silhouette_tsne_2d']:.4f}",fontsize=11)
   ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
 fig.suptitle('Fixed outer-fold illustrations: model seed 42, fold 1 | independent embeddings',fontsize=13)
 fig.tight_layout(rect=(0,.05,1,.95));save(fig,'03_tsne_all_fixed_panels')
 architecture()
 manifest={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(I.iterdir()) if f.is_file()}
 (O/'SOURCE_HASHES.json').write_text(json.dumps(manifest,indent=2))
 print('ASSETS_COMPLETED: 7 tables, 4 figures in SVG/PDF/PNG; no training or Test rerun')

def architecture():
 fig,ax=plt.subplots(figsize=(15,10));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 def box(x,y,w,h,text,color='#eaf1f5',size=11):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor=color,edgecolor='#385363',linewidth=1.3))
  ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
 def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#385363',lw=1.4))
 ax.text(.22,.98,'A. Convolutional backbone',ha='center',weight='bold',fontsize=14)
 ax.text(.75,.98,'B. Residual attention and classifier',ha='center',weight='bold',fontsize=14)
 box(.03,.86,.38,.07,'EEG epoch\n64 electrodes × 52 samples × 1')
 box(.03,.71,.38,.10,'Temporal Conv2D: 8 filters, 1 × 32, same\nBatch normalization (no ELU here)\n64 × 52 × 8')
 box(.03,.53,.38,.13,'Depthwise Conv2D: 64 × 1, valid, multiplier 2\nBN → ELU → AvgPool 1 × 4 → Dropout 0.25\n1 × 13 × 16',size=10.5)
 box(.03,.35,.38,.13,'Separable Conv2D: 16 filters, 1 × 16, same\nBN → ELU → AvgPool 1 × 4 → Dropout 0.25\n1 × 3 × 16',size=10.5)
 box(.03,.23,.38,.07,'Backbone feature tensor X: 1 × 3 × 16')
 for y1,y2 in ((.86,.81),(.71,.66),(.53,.48),(.35,.30)):arrow((.22,y1),(.22,y2))
 # Route backbone output to attention input, avoiding the interior of either panel.
 ax.plot([.41,.46,.46,.63],[.265,.265,.89,.89],color='#385363',lw=1.4)
 arrow((.63,.89),(.652,.89))
 box(.66,.85,.18,.08,'X: 1 × 3 × 16')
 box(.52,.65,.20,.13,'Channel gate C(X)\nGAP → Dense 4 ReLU\n→ Dense 16 sigmoid\n1 × 1 × 16',size=10)
 box(.77,.65,.20,.13,'Temporal gate T(X)\nreshape 3 × 16\nConv1D 16, k=3, ReLU\nConv1D 16, k=1, sigmoid\n1 × 3 × 16',size=9.5)
 arrow((.70,.85),(.62,.78));arrow((.80,.85),(.87,.78))
 box(.60,.49,.29,.08,'G = C(X) ⊙ T(X)\nBroadcast product: 1 × 3 × 16')
 arrow((.62,.65),(.69,.57));arrow((.87,.65),(.80,.57))
 box(.60,.34,.29,.09,'Y = X + X ⊙ G\nResidual output: 1 × 3 × 16',color='#f8ecef')
 arrow((.745,.49),(.745,.43))
 ax.plot([.84,.99,.99,.89],[.89,.89,.385,.385],color='#385363',lw=1.3);arrow((.94,.385),(.89,.385))
 ax.text(.975,.56,'X',ha='center',fontsize=12)
 box(.60,.18,.29,.10,'Flatten: 48 features\nDense 1 + sigmoid\nP(target | epoch)')
 arrow((.745,.34),(.745,.28))
 ax.text(.22,.14,'Ablations: none = X; channel = X + X⊙C\ntemporal = X + X⊙T; full = X + X⊙C⊙T',ha='center',va='center',fontsize=10)
 ax.text(.5,.055,'Full model: 3,185 total parameters = 3,105 trainable + 80 non-trainable\nFeature-channel attention operates on 16 learned feature maps, not directly on 64 electrodes.\nDiagram excludes preprocessing and character decoding; tensor dimensions omit batch size.',ha='center',va='center',fontsize=10)
 save(fig,'04_architecture_verified')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path)
 args=parser.parse_args();O=(args.output_dir or ROOT).resolve()
 if O==I.resolve() or I.resolve() in O.parents:raise ValueError('Output overlaps input sources')
 T=O/'tables';F=O/'figures';main()
