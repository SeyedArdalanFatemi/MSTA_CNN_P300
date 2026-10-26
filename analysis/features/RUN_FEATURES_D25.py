"""Frozen-model outer-fold features; no training, official Test, or model selection."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,gc,os,sys,time,zipfile,traceback,json
import numpy as np
import sklearn
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
import msta as c
from reference_check import check_reference
V=('none','channel','temporal','full')
STATUS='FEATURES_D25_COMPLETED_NOT_OFFICIAL_TEST_NOT_CONFIRMATORY'

def sample_indices(y):
    rng=np.random.default_rng(314159)
    chosen=np.sort(np.concatenate([rng.choice(np.flatnonzero(y==label),count,replace=False) for label,count in ((0,1000),(1,200))]))
    c.require(len(chosen)==1200 and len(np.unique(chosen))==1200,'Invalid sample')
    return chosen

def metric(features,y):
    c.require(features.shape==(1200,48) and np.isfinite(features).all(),'Invalid features')
    return float(silhouette_score(features,y,metric='euclidean'))

def plot_embedding(embedding,y,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    for label,color,name in ((0,'#627d98','Non-target'),(1,'#d1495b','Target')):
        mask=y==label;ax.scatter(embedding[mask,0],embedding[mask,1],s=10,alpha=.65,c=color,label=name,edgecolors='none')
    ax.set(title=title,xlabel='t-SNE 1',ylabel='t-SNE 2');ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('svg','png'):fig.savefig(path.with_suffix('.'+ext),dpi=300)
    plt.close(fig)

def verify_inputs(ablation,reference):
    p,s=check_reference(reference)
    a=c.read(ablation/'ablation_d25_summary.json');ap=c.read(ablation/'protocol.json')
    c.require(a['status']=='ABLATION_D25_CV_COMPLETED_NOT_OFFICIAL_TEST_NOT_CONFIRMATORY','Incomplete d25 ablation')
    c.require(a['protocol']==ap and ap['reference_protocol_sha256']==c.digest(reference/'protocol.json') and ap['reference_summary_sha256']==c.digest(reference/'cv_summary.json'),'Reference identity differs')
    c.require(ap['code_sha256']['msta.py']==c.digest(Path(c.__file__)) and ap['environment']==p['environment'],'Ablation model code/runtime differs')
    c.require(ap['configuration']=={**c.configuration('primary'),'profile':'matched_ablation_d25','variants':list(V)},'Ablation settings differ')
    expected={(sub,v,se,f) for sub in 'AB' for v in ('channel','temporal') for se in c.SEEDS for f in range(1,6)}
    c.require(len(a['jobs'])==60 and {(j['subject'],j['variant'],j['seed'],j['fold']) for j in a['jobs']}==expected,'Ablation coverage differs')
    result={}
    for base,jobs in ((reference,s['jobs']),(ablation,a['jobs'])):
        for j in jobs:
            key=(j['subject'],j['variant'],j['seed'],j['fold'])
            folder=base/'cv'/f'{key[0]}_{key[1]}_{key[2]}_fold{key[3]}'
            c.require(c.read(folder/'report.json')==j,'Job report differs')
            c.verify_cv_job(folder,j,c.digest(base/'protocol.json'))
            result[key]=(folder,j)
    return p,result

def exports(run,rows):
    with (run/'silhouette_48d.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subject','variant','seed','fold','silhouette_48d','silhouette_tsne_2d','prediction_max_abs_difference']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    if len(rows)!=120:return
    groups=[]
    for sub in 'AB':
        for v in V:
            means=[float(np.mean([r['silhouette_48d'] for r in rows if (r['subject'],r['variant'],r['seed'])==(sub,v,se)])) for se in c.SEEDS]
            groups.append({'subject':sub,'variant':v,'mean_fold_seed_silhouette_48d':float(np.mean(means)),
                'sample_sd_of_seed_means':float(np.std(means,ddof=1)),'seed_means':means})
    c.write(run/'silhouette_aggregate.json',{'rows':groups,'note':'Average of within-fold scores; feature coordinates from separate trained models are NOT concatenated. Descriptive; seeds/folds are not independent participants.'})
    lookup={(r['subject'],r['seed'],r['fold'],r['variant']):r['silhouette_48d'] for r in rows}
    pairs=[{'subject':sub,'seed':se,'fold':f,**{v+'_minus_none':lookup[sub,se,f,v]-lookup[sub,se,f,'none'] for v in V[1:]}} for sub in 'AB' for se in c.SEEDS for f in range(1,6)]
    c.write(run/'paired_silhouette.json',pairs)

def export_zip(run):
    target=run/'SEND_features_d25_review.zip';temp=run/'pending.zip'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run.rglob('*')):
            if f.is_file() and 'cache' not in f.relative_to(run).parts and f.suffix in ('.json','.csv','.npz','.svg','.png'):
                z.write(f,run.name+'/'+f.relative_to(run).as_posix())
    temp.replace(target)

def self_test():
    y=np.tile(np.r_[np.ones(30,dtype=int),np.zeros(150,dtype=int)],17)
    ix=sample_indices(y);c.require(np.array_equal(ix,sample_indices(y)) and y[ix].sum()==200,'Sampling test failed')
    rng=np.random.default_rng(4);features=rng.normal(size=(1200,48));features[y[ix]==1]+=5
    score=metric(features,y[ix]);c.require(score>.6,'Separation test failed')
    c.require(abs(score-metric(features*2+3,y[ix]))<1e-10,'Silhouette invariance failed')
    print('FEATURE_ANALYSIS_SELF_TEST_PASSED; no TensorFlow inference tested')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('ablation-run','reference-run','data-dir','resume-run','output-dir'):ap.add_argument('--'+arg,type=Path)
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return 0
    if not all((args.ablation_run,args.reference_run,args.data_dir)):ap.error('Use START_FEATURES_D25.py or supply all three input folders')
    ablation,reference,data=[p.resolve() for p in (args.ablation_run,args.reference_run,args.data_dir)]
    print('Verifying 120 existing CV jobs...',flush=True)
    p,jobs=verify_inputs(ablation,reference)
    for sub in 'AB':c.require(c.digest(data/f'Subject_{sub}_Train.mat')==c.TRAIN_HASHES[sub],'Train identity differs')
    tf,env=c.runtime(4);c.require(env==p['environment'],'Use the original Python environment')
    import matplotlib
    protocol={'stage':'features_d25','ablation_run':str(ablation),'reference_run':str(reference),'data_dir':str(data),
      'ablation_summary_sha256':c.digest(ablation/'ablation_d25_summary.json'),
      'reference_summary_sha256':c.digest(reference/'cv_summary.json'),
      'code_sha256':{f.name:c.digest(f) for f in Path(__file__).parent.glob('*.py')},
      'environment':env,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__,
      'feature_layer':'features: flattened post-attention / pre-classifier, 48 dimensions',
      'sampling':{'partition':'outer only','total':1200,'target':200,'nontarget':1000,'seed':314159,'same_indices_across_all_variants_and_model_seeds':True},
      'silhouette':{'metric':'euclidean','feature_scaling':'none','all_120_models':True},
      'tsne':{'visualization_model_seed':42,'fold':1,'random_state':42,'perplexity':30,'init':'pca','learning_rate':'auto','max_iter':1000,'method':'barnes_hut','angle':.5,'n_jobs':1},
      'official_test_read':False,'training_performed':False,'limits':'Exploratory class-label separation, not evidence of causal attention mechanism. 2D silhouette differs from original-space silhouette; plots are fixed illustrations, never selected by appearance.'}
    run=(args.resume_run or args.output_dir or Path(__file__).parent/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    for source in (ablation,reference,data):c.require(run!=source and source not in run.parents and run not in source.parents,'Overlapping input/output')
    if args.resume_run:c.require(c.read(run/'protocol.json')==protocol,'Resume inputs/code/environment changed')
    else:run.mkdir(parents=True,exist_ok=False);c.write(run/'protocol.json',protocol)
    fd=os.open(run/'RUNNING.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    start=time.perf_counter();summary={'status':'RUNNING','protocol':protocol,'jobs':[]}
    try:
        for sub in 'AB':
            X,y,meta,_=c.training_data(data,run,sub)
            for split in c.make_splits():
                arrays,idx,stats=c.prepare_fold(X,meta,split)
                selected=sample_indices(y[idx['outer']]);truth=y[idx['outer']][selected]
                for seed in c.SEEDS:
                    for variant in V:
                        key=f'{sub}_{variant}_{seed}_fold{split["fold"]}';folder,job=jobs[sub,variant,seed,split['fold']]
                        dest=run/'features'/key;dest.mkdir(parents=True,exist_ok=True)
                        c.require(stats==job['normalization'],'Normalization differs')
                        with np.load(folder/'outer_predictions.npz',allow_pickle=False) as z:
                            c.require(np.array_equal(z['indices'],idx['outer']) and np.array_equal(z['truth'],y[idx['outer']]) and np.array_equal(z['metadata'],meta[idx['outer']]),'Outer sample identity differs')
                            stored=z['probability'].copy()
                        if (dest/'report.json').exists():
                            row=c.read(dest/'report.json');c.require(row['checkpoint_sha256']==job['checkpoint_sha256'],'Checkpoint changed')
                            c.require((row['subject'],row['variant'],row['seed'],row['fold'])==(sub,variant,seed,split['fold']),'Analysis task changed')
                            for name,h in row['files'].items():c.require(c.digest(dest/name)==h,'Saved analysis file changed')
                            with np.load(dest/'features.npz',allow_pickle=False) as z:
                                c.require(np.array_equal(z['global_indices'],idx['outer'][selected]) and np.array_equal(z['truth'],truth),'Saved samples differ')
                                c.require(np.isclose(metric(z['features'],truth),row['silhouette_48d'],rtol=1e-9,atol=1e-10),'Saved metric differs')
                            print('REUSE',key,flush=True)
                        else:
                            print('START',key,flush=True)
                            tf.keras.backend.clear_session();model=tf.keras.models.load_model(folder/'best.keras',safe_mode=True)
                            c.check_dropout(tf,model,.25);c.require(model.count_params()==c.EXPECTED_PARAMS[variant],'Model architecture differs')
                            prob=c.predict_batches(model,arrays['outer']);diff=float(np.max(np.abs(prob-stored)))
                            c.require(np.allclose(prob,stored,rtol=1e-6,atol=1e-7),'Stored predictions do not reproduce')
                            extractor=tf.keras.Model(model.inputs,model.get_layer('features').output)
                            features=np.concatenate([np.asarray(extractor(arrays['outer'][selected[i:i+128]],training=False)) for i in range(0,1200,128)])
                            score=metric(features,truth)
                            np.savez_compressed(dest/'features.npz',features=features,truth=truth,metadata=meta[idx['outer']][selected],global_indices=idx['outer'][selected])
                            row={'subject':sub,'variant':variant,'seed':seed,'fold':split['fold'],'silhouette_48d':score,'silhouette_tsne_2d':None,'prediction_max_abs_difference':diff,'checkpoint_sha256':job['checkpoint_sha256']}
                            files=['features.npz']
                            if seed==42 and split['fold']==1:
                                settings={k:v for k,v in protocol['tsne'].items() if k not in ('visualization_model_seed','fold')}
                                with threadpool_limits(limits=4):
                                    embedding=TSNE(n_components=2,metric='euclidean',**settings).fit_transform(features)
                                row['silhouette_tsne_2d']=float(silhouette_score(embedding,truth,metric='euclidean'))
                                np.savez_compressed(dest/'tsne.npz',embedding=embedding,truth=truth,global_indices=idx['outer'][selected])
                                plot_embedding(embedding,truth,dest/'tsne',f'{sub} | {variant} | seed 42, fold 1')
                                files+=['tsne.npz','tsne.svg','tsne.png']
                            row['files']={f:c.digest(dest/f) for f in files};c.write(dest/'report.json',row)
                            del model,extractor,features;tf.keras.backend.clear_session();gc.collect()
                            print('DONE',key,flush=True)
                        summary['jobs'].append(row);c.write(run/'feature_summary.json',summary)
                del arrays;gc.collect()
            del X;gc.collect()
        c.require(len(summary['jobs'])==120,'Incomplete analysis');exports(run,summary['jobs']);summary['status']=STATUS
    except BaseException as exc:summary.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
    finally:
        summary['wall_seconds_this_invocation']=time.perf_counter()-start;c.write(run/'feature_summary.json',summary)
        try:export_zip(run)
        finally:(run/'RUNNING.lock').unlink(missing_ok=True)
    print(summary['status'],flush=True)
    if 'error' in summary:print(summary['error'],flush=True)
    print('SEND:',run/'SEND_features_d25_review.zip',flush=True)
    return 0 if summary['status']==STATUS else 1

if __name__=='__main__':raise SystemExit(main())
