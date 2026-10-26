"""Descriptive supplementary analysis of frozen primary Test outputs; no fitting/tuning."""
from pathlib import Path
import csv,json,math,sys,platform,argparse
from statistics import NormalDist
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef
import msta as c
ROOT=Path(__file__).resolve().parent

def wilson(k,n):
    c.require(isinstance(k,int) and isinstance(n,int) and n>0 and 0<=k<=n,'Invalid binomial counts')
    z=NormalDist().inv_cdf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)

def save_csv(out,name,rows):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compare(a,b):
    if isinstance(a,dict):
        c.require(set(a)==set(b),'Metric keys differ')
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        c.require(len(a)==len(b),'Metric lengths differ')
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(float,int)):c.require(np.isclose(a,b,rtol=1e-9,atol=1e-10),'Metric value differs')
    else:c.require(a==b,'Metric differs')

def pair_counts(a,b):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    c.require(a.shape==b.shape and a.ndim==1,'Pair shape differs')
    return {'both_correct':int((a&b).sum()),'both_wrong':int((~a&~b).sum()),
        'corrected_by_attention':int((~a&b).sum()),'new_errors_with_attention':int((a&~b).sum()),
        'attention_minus_none_pp':100*float(np.mean(b.astype(int)-a.astype(int)))}

def self_test():
    lo,hi=wilson(0,100);c.require(lo<1e-12 and abs(hi-.03699349820698568)<1e-12,'Wilson zero test')
    lo2,hi2=wilson(100,100);c.require(abs(lo2-(1-hi))<1e-12 and abs(hi2-1)<1e-12,'Wilson endpoint symmetry')
    lo3,hi3=wilson(50,100);c.require(abs(lo3+hi3-1)<1e-12,'Wilson midpoint symmetry')
    q=pair_counts([1,0,0,1],[1,1,0,0]);c.require(q==dict(both_correct=1,both_wrong=1,corrected_by_attention=1,new_errors_with_attention=1,attention_minus_none_pp=0.),'Pair accounting test')
    c.require(balanced_accuracy_score([0,0,1],[0,0,0])==.5 and matthews_corrcoef([0,0,1],[0,0,0])==0,'Majority baseline test')
    print('SUPPLEMENTARY_SELF_TEST_PASSED')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--evaluation-dir',type=Path,help='Optional msta.py evaluation/within_subject folder')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();self_test()
    if args.self_test:return
    inputs=(args.evaluation_dir or ROOT/'inputs').resolve()
    out=(args.output_dir or ROOT/'results').resolve()
    c.require(out!=inputs and inputs not in out.parents and out not in inputs.parents,'Input/output overlap')
    out.mkdir(parents=True,exist_ok=True)
    folders={}
    if args.evaluation_dir:
        source=c.read(inputs/'summary.json')
        c.require(source['status']=='EVALUATION_COMPLETED' and source['identity']['mode']=='within_subject','Expected completed within-subject evaluation')
        refs={}
        for j in source['jobs']:
            t=j['task'];key=(j['source'],t['variant'],t['seed'])
            c.require(j['source']==j['target'] and t['dropout_rate']==.25,'Wrong evaluation protocol')
            name=t['key'];c.require(Path(name).name==name and name not in ('.','..'),'Invalid job path')
            c.require(key not in refs,'Duplicate evaluation job')
            refs[key]=j;folders[key]=inputs/name
    else:
        reference=c.read(inputs/'primary_reference.json')
        refs={(j['source'],j['variant'],j['seed']):j for j in reference['jobs']}
        folders={k:inputs/f'{k[0]}_{k[1]}_d25_{k[2]}' for k in refs}
    c.require(set(refs)=={(a,v,se) for a in 'AB' for v in ('none','full') for se in c.SEEDS},'Incomplete evaluation coverage')
    metrics=[];intervals=[];errors=[];allrows={};hashes={}
    for sub in 'AB':
        base_meta=None
        for variant in ('none','full'):
            for seed in c.SEEDS:
                folder=folders[sub,variant,seed];report=c.read(folder/'report.json');task=report['task']
                c.require((task['subject'],task['variant'],task['seed'],task['dropout_rate'])==(sub,variant,seed,.25),'Task mismatch')
                c.require(c.digest(folder/'predictions.npz')==report['predictions_sha256'],'Prediction hash mismatch')
                ref=refs[sub,variant,seed];c.require(report['checkpoint_sha256']==ref['checkpoint_sha256'],'Wrong reference checkpoint')
                with np.load(folder/'predictions.npz',allow_pickle=False) as z:p=z['probability'];meta=z['metadata']
                c.require(p.shape==(18000,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
                c.require(meta.shape==(18000,4) and set(meta[:,0])==set(range(100)),'Wrong Test coverage')
                if base_meta is None:base_meta=meta.copy()
                else:c.require(np.array_equal(base_meta,meta),'Unmatched Test event identities')
                for char in range(100):
                    m=meta[meta[:,0]==char];c.require(len(m)==180 and len(set(map(tuple,m[:,1:3])))==180,'Duplicate/missing events')
                y=c.truth(c,c.LABELS[sub],meta);c.require(int(y.sum())==3000,'Wrong class counts')
                bm=c.binary_metrics(y,p);compare(bm,report['binary']);compare(bm,ref['binary'])
                curve,rows=c.character_metrics(y,p,meta)
                c.require(rows==c.read(folder/'characters.json') and curve==report['character_curve'] and curve==ref['character_curve'],'Character decoding mismatch')
                allrows[sub,variant,seed]={(r['character_id'],r['sequences']):r for r in rows}
                tn,fp,fn,tp=np.array(bm['confusion_matrix_rows_true_cols_pred_0_1']).ravel()
                metrics.append({'subject':sub,'variant':variant,'seed':seed,'threshold':.5,'target_prevalence':float(y.mean()),
                    'roc_auc':bm['roc_auc'],'average_precision':bm['average_precision'],'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),
                    'mcc':float(matthews_corrcoef(y,p>=.5)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),
                    'target_precision':bm['target_precision'],'target_f1':bm['target_f1'],'flash_accuracy':bm['accuracy'],
                    'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
                k=curve[-1]['correct'];lo,hi=wilson(k,100)
                intervals.append({'subject':sub,'variant':variant,'seed':seed,'repetitions':15,'correct':k,'characters':100,'accuracy_percent':k,
                    'wilson95_low_percent':lo*100,'wilson95_high_percent':hi*100,'assumption':'Nominal binomial interval within this fixed model/subject; not a CI across seeds or subjects'})
                for r in rows:
                    if r['sequences']!=15:continue
                    ti=c.MATRIX.index(r['true_character']);pi=c.MATRIX.index(r['predicted_character'])
                    rw=ti//6!=pi//6;cw=ti%6!=pi%6
                    errors.append({'subject':sub,'variant':variant,'seed':seed,'character_id_zero_based':r['character_id'],
                      'true_character':r['true_character'],'predicted_character':r['predicted_character'],
                      'error_type':('both_row_and_column' if rw and cw else 'row_only' if rw else 'column_only' if cw else 'correct')})
    pairs=[];details=[]
    for sub in 'AB':
        for seed in c.SEEDS:
            a,b=[allrows[sub,v,seed] for v in ('none','full')]
            for rep in range(1,16):
                aa=[a[ch,rep]['correct'] for ch in range(100)];bb=[b[ch,rep]['correct'] for ch in range(100)]
                pairs.append({'subject':sub,'seed':seed,'repetitions':rep,**pair_counts(aa,bb)})
                if rep==15:
                    for ch in range(100):
                        x,z=a[ch,rep],b[ch,rep];c.require(x['true_character']==z['true_character'],'Unmatched truth')
                        category='both_correct' if x['correct'] and z['correct'] else 'corrected_by_attention' if z['correct'] else 'new_errors_with_attention' if x['correct'] else 'both_wrong'
                        details.append({'subject':sub,'seed':seed,'character_id_zero_based':ch,'truth':x['true_character'],'none_prediction':x['predicted_character'],'full_prediction':z['predicted_character'],'category':category})
    summary=[]
    names=['roc_auc','average_precision','balanced_accuracy','mcc','sensitivity','specificity','target_precision','target_f1','flash_accuracy']
    for sub in 'AB':
        for v in ('none','full'):
            selected=[r for r in metrics if r['subject']==sub and r['variant']==v]
            row={'subject':sub,'variant':v}
            for name in names:
                vals=[r[name] for r in selected];row[name+'_mean']=float(np.mean(vals));row[name+'_seed_sd']=float(np.std(vals,ddof=1))
            summary.append(row)
    errorcounts=[]
    for sub in 'AB':
        for v in ('none','full'):
            for se in c.SEEDS:
                selected=[e for e in errors if (e['subject'],e['variant'],e['seed'])==(sub,v,se)]
                errorcounts.append({'subject':sub,'variant':v,'seed':se,**{k:sum(e['error_type']==k for e in selected) for k in ('correct','row_only','column_only','both_row_and_column')}})
    for name,rows in [('binary_by_seed.csv',metrics),('binary_summary.csv',summary),('wilson_character_accuracy15.csv',intervals),('paired_errors_all_repetitions.csv',pairs),('paired_errors_at15.csv',[r for r in pairs if r['repetitions']==15]),('paired_character_details_at15.csv',details),('row_column_errors_by_seed.csv',errorcounts),('character_error_details.csv',errors)]:save_csv(out,name,rows)
    save_csv(out,'chance_and_baselines.csv',[
        {'level':'character','baseline':'Uniform independent row and column choice','metric':'accuracy','value':1/36},
        {'level':'flash','baseline':'Always non-target','metric':'accuracy','value':5/6},
        {'level':'flash','baseline':'Always non-target','metric':'balanced_accuracy','value':.5},
        {'level':'flash','baseline':'Always non-target; sklearn zero-denominator convention','metric':'MCC','value':0},
        {'level':'flash','baseline':'Constant score for every flash','metric':'ROC-AUC','value':.5},
        {'level':'flash','baseline':'Constant score; prevalence reference','metric':'AP','value':1/6}])
    save_csv(out,'design_quantities.csv',[
        {'quantity':'Nominal first temporal kernel duration','calculation':'32 / 64','value':.5,'unit':'seconds','limit':'Not proof of P300 selectivity; excludes later receptive-field expansion'},
        {'quantity':'Attention-grid temporal stride','calculation':'4 * 4 / 64','value':.25,'unit':'seconds','limit':'Not 0.8/3; receptive fields overlap and use padding'},
        {'quantity':'Input samples','calculation':'stored preprocessing output','value':52,'unit':'samples','limit':'Last center at 51/64 seconds'},
        {'quantity':'Spatial kernel height','calculation':'Depthwise kernel 64 x 1, multiplier 2','value':64,'unit':'electrodes','limit':'Learned spatial combinations; not direct electrode-importance scores'},
        {'quantity':'Attention feature channels','calculation':'8 temporal filters * depth multiplier 2','value':16,'unit':'feature maps','limit':'Different from 64 acquisition electrodes'}])
    for f in inputs.rglob('*'):
        if f.is_file():hashes[f.relative_to(inputs).as_posix()]=c.digest(f)
    c.write(out/'analysis_manifest.json',{'status':'SUPPLEMENTARY_ANALYSIS_COMPLETED_EXPLORATORY','source_hashes':hashes,'script_sha256':c.digest(Path(__file__)),'core_sha256':c.digest(Path(c.__file__)),
        'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
        'training_performed':False,'new_inference_performed':False,'threshold_optimized':False,'threshold':.5,'primary_endpoint':'Character accuracy at repetition15 remains unchanged',
        'limits':'Previously inspected Test; descriptive supplementary analysis. No p-values, no confidence interval pooling seeds/subjects/flashes, no selective metric reporting.'})
    print('SUPPLEMENTARY_ANALYSIS_COMPLETED_EXPLORATORY')
    print(json.dumps([r for r in pairs if r['repetitions']==15],indent=2))

if __name__=='__main__':main()
