"""Complete matched dropout=.25 ablation: reuse 60 verified jobs, train 60; Train/CV only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, gc, json, os, time, traceback, zipfile
import numpy as np
import msta as core
from reference_check import check_reference
read,write,require=core.read,core.write,core.require
VARIANTS=('none','channel','temporal','full')


def summarize(jobs):
    keys=[(j['subject'],j['variant'],j['seed'],j['fold']) for j in jobs]
    expected={(s,v,se,f) for s in 'AB' for v in VARIANTS for se in core.SEEDS for f in range(1,6)}
    require(len(keys)==120 and len(set(keys))==120 and set(keys)==expected,'Incomplete or duplicate 120-job ablation')
    for j in jobs:
        curve=j['outer_character_curve']
        require(len(curve)==15,'Incomplete repetition curve')
        for n,c in enumerate(curve,1):
            require(c['sequences']==n and c['characters']==17 and 0<=c['correct']<=17,'Invalid fold counts')
            require(abs(c['accuracy']-c['correct']/17)<1e-12,'Accuracy/count mismatch')
    by_subject=core.task_summary(jobs);table=[];overall={}
    for v in VARIANTS:
        per=[]
        for seed in core.SEEDS:
            selected=[j for j in jobs if j['variant']==v and j['seed']==seed]
            curve=[sum(j['outer_character_curve'][i]['correct'] for j in selected)/170 for i in range(15)]
            per.append({'seed':seed,'accuracy_15':curve[-1],'mean_accuracy_1_to_15':float(np.mean(curve)),'curve':curve})
        overall[v]={'per_seed':per,**{m:{'mean':float(np.mean([r[m] for r in per])),
          'sample_sd':float(np.std([r[m] for r in per],ddof=1))} for m in ('accuracy_15','mean_accuracy_1_to_15')}}
    for subject,groups in [*by_subject.items(),('A+B',overall)]:
        for v in VARIANTS:
            r=groups[v]
            table.append({'subject':subject,'variant':v,'total_parameters':core.EXPECTED_PARAMS[v],
              'mean_accuracy_15_percent':100*r['accuracy_15']['mean'],
              'seed_sample_sd_percentage_points':100*r['accuracy_15']['sample_sd'],
              'mean_accuracy_1_to_15_percent':100*r['mean_accuracy_1_to_15']['mean']})
    bykey={k:j for k,j in zip(keys,jobs)};paired=[]
    for sub in 'AB':
        for seed in core.SEEDS:
            for fold in range(1,6):
                row={'subject':sub,'seed':seed,'fold':fold}
                base=bykey[sub,'none',seed,fold]['outer_character_curve']
                for v in VARIANTS[1:]:
                    c=bykey[sub,v,seed,fold]['outer_character_curve']
                    row[v+'_minus_none_accuracy15_pp']=100*(c[-1]['accuracy']-base[-1]['accuracy'])
                    row[v+'_minus_none_mean_1_to_15_pp']=100*float(np.mean([a['accuracy']-b['accuracy'] for a,b in zip(c,base)]))
                paired.append(row)
    return {'by_subject':by_subject,'overall':overall,'table':table,'paired_folds':paired,
      'limits':'Exploratory OOF CV, not official Test. Seed/fold rows are correlated, not independent participants. No automatic selection or significance claim.'}


def aggregation_test():
    jobs=[]
    for sub in 'AB':
        for v in VARIANTS:
            for seed in core.SEEDS:
                for fold in range(1,6):
                    correct=16 if v=='full' else 15
                    jobs.append({'subject':sub,'variant':v,'seed':seed,'fold':fold,
                        'outer_character_curve':[{'sequences':n,'characters':17,'correct':correct,'accuracy':correct/17} for n in range(1,16)]})
    result=summarize(jobs)
    require(abs(result['overall']['full']['accuracy_15']['mean']-16/17)<1e-12,'Wrong pooled accuracy')
    require(all(abs(r['full_minus_none_accuracy15_pp']-100/17)<1e-12 for r in result['paired_folds']),'Wrong paired delta')
    for bad in (jobs[:-1],jobs[:-1]+[jobs[0]]):
        try:summarize(bad)
        except ValueError:pass
        else:raise ValueError('Missing/duplicate job not rejected')
    print('ABLATION_AGGREGATION_SELF_TEST_PASSED',flush=True)
    return {'pooled_counts':True,'paired_difference':True,'missing_and_duplicate_rejected':True}


def export(run):
    tmp=run/'review_pending.zip';target=run/'SEND_ablation_d25_review.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run.rglob('*')):
            if f.is_file() and 'cache' not in f.relative_to(run).parts and f.suffix in ('.json','.txt','.npz','.csv','.keras'):
                z.write(f,run.name+'/'+f.relative_to(run).as_posix())
    tmp.replace(target)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('reference-run','data-dir','resume-run','output-dir'):ap.add_argument('--'+k,type=Path)
    ap.add_argument('--self-test',action='store_true');ap.add_argument('--smoke-test',action='store_true')
    args=ap.parse_args()
    checks=aggregation_test()
    if args.self_test:core.self_test();return 0
    if args.smoke_test:
        tf,_=core.runtime(4);core.smoke_test(tf);return 0
    if args.reference_run is None or args.data_dir is None:ap.error('Run START_ABLATION_D25.py or supply --reference-run and --data-dir')
    root=args.reference_run.resolve();data=args.data_dir.resolve();p,s=check_reference(root)
    for sub in 'AB':
        source=data/f'Subject_{sub}_Train.mat'
        require(source.is_file() and core.digest(source)==core.TRAIN_HASHES[sub],'Missing or unexpected Train data: '+sub)
    print('REFERENCE_VERIFIED: 60 completed primary CV jobs',flush=True)
    tf,env=core.runtime(4)
    require(env==p['environment'],'Reference runtime differs; use the same original .venv')
    protocol={'stage':'matched_ablation_d25_cv','reference_run':str(root),'data_dir':str(data),
      'reference_protocol_sha256':core.digest(root/'protocol.json'),'reference_summary_sha256':core.digest(root/'cv_summary.json'),
      'code_sha256':{f.name:core.digest(f) for f in Path(__file__).parent.glob('*.py')},'environment':env,
      'configuration':{**p['configuration'],'profile':'matched_ablation_d25','variants':['none','channel','temporal','full']},'threads':4,'additional_fits':60,
      'normalization':'Unchanged core scalar fit-only normalization',
      'selection':'earliest minimum INNER unweighted BCE; outer results never select checkpoint',
      'primary_endpoint':'OOF character accuracy at repetition15, both subjects and all seeds',
      'secondary_endpoint':'mean character accuracy repetitions1..15',
      'official_test_read':False,'reference_results_reused':60,
      'limitations':'Exploratory on previously inspected subjects/folds; prior official Test exposure is not erased. No final training, Test scoring, significance test or automatic model selection.'}
    run=(args.resume_run or args.output_dir or Path(__file__).parent/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    for protected in (root,data):require(run!=protected and protected not in run.parents and run not in protected.parents,'Output overlaps input')
    if args.resume_run:require(read(run/'protocol.json')==protocol,'Resume code/config/runtime identity changed')
    else:run.mkdir(parents=True,exist_ok=False);write(run/'protocol.json',protocol)
    fd=os.open(run/'RUNNING.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    result={'status':'RUNNING','protocol':protocol,'jobs':[],'aggregation_tests':checks};start=time.perf_counter()
    try:
        core.smoke_test(tf)
        result['tensorflow_smoke_test']='PASSED'
        write(run/'reference_cv_summary.json',s);ph=core.digest(run/'protocol.json')
        refs={(j['subject'],j['variant'],j['seed'],j['fold']):j for j in s['jobs']}
        for sub in 'AB':
            X,y,meta,_=core.training_data(data,run,sub)
            for seed in core.SEEDS:
                for split in p['configuration']['splits']:
                    arrays,idx,stats=core.prepare_fold(X,meta,split)
                    # Match raw data, fit statistics and held-out membership to BOTH reference variants.
                    for rv in ('none','full'):
                        ref=refs[sub,rv,seed,split['fold']]
                        require(ref['normalization']==stats,'Reference fit statistics differ')
                        rf=root/'cv'/f'{sub}_{rv}_{seed}_fold{split["fold"]}'
                        for part in ('inner','outer'):
                            with np.load(rf/(part+'_predictions.npz'),allow_pickle=False) as z:
                                require(np.array_equal(z['indices'],idx[part]) and np.array_equal(z['truth'],y[idx[part]]) and np.array_equal(z['metadata'],meta[idx[part]]),'Reference sample identity differs')
                    for variant in ('channel','temporal'):
                        key=f'{sub}_{variant}_{seed}_fold{split["fold"]}'
                        folder=run/'cv'/key;folder.mkdir(parents=True,exist_ok=True)
                        task={'subject':sub,'variant':variant,'seed':seed,'fold':split['fold'],'dropout_rate':.25}
                        if (folder/'report.json').exists():
                            j=read(folder/'report.json');require(all(j[k]==v for k,v in task.items()),'Resume task changed')
                            require(j['normalization']==stats,'Resume normalization changed');core.verify_cv_job(folder,j,ph)
                            print('REUSE',key,flush=True)
                        else:
                            print('START',key,'matched dropout=0.25 ablation',flush=True)
                            j=core.train_cv_fold(tf,core,arrays,idx,y,meta,variant,folder,stats,seed,.25)
                            model=tf.keras.models.load_model(folder/'best.keras',safe_mode=True)
                            require(model.count_params()==core.EXPECTED_PARAMS[variant],'Architecture changed')
                            prob=core.predict_batches(model,arrays['outer']);curve,rows=core.character_metrics(y[idx['outer']],prob,meta[idx['outer']])
                            np.savez_compressed(folder/'outer_predictions.npz',probability=prob,truth=y[idx['outer']],metadata=meta[idx['outer']],indices=idx['outer'])
                            write(folder/'outer_characters.json',rows)
                            j.update(**task,status='COMPLETED',protocol_sha256=ph,outer_character_curve=curve,
                              outer_binary=core.binary_metrics(y[idx['outer']],prob),normalization_method='scalar_fit_only',
                              inner_predictions_sha256=core.digest(folder/'inner_predictions.npz'),outer_predictions_sha256=core.digest(folder/'outer_predictions.npz'))
                            require(j['best_epoch']==int(np.argmin(j['history']['val_loss']))+1,'Checkpoint rule changed')
                            write(folder/'report.json',j);core.verify_cv_job(folder,j,ph)
                            del model;tf.keras.backend.clear_session();gc.collect();print('DONE',key,flush=True)
                        result['jobs'].append(j);write(run/'ablation_d25_summary.json',result)
                    del arrays;gc.collect()
            del X;gc.collect()
        result['comparison']=summarize(s['jobs']+result['jobs'])
        rows=result['comparison']['table']
        with (run/'matched_ablation.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (run/'paired_folds.csv').open('w',encoding='utf-8-sig',newline='') as f:
            pairs=result['comparison']['paired_folds']
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
        curve_rows=[]
        for sub in 'AB':
            for variant in VARIANTS:
                group=result['comparison']['by_subject'][sub][variant]['per_seed']
                for n in range(15):
                    vals=[g['character_curve'][n]['accuracy'] for g in group]
                    curve_rows.append({'subject':sub,'variant':variant,'repetitions':n+1,
                        'mean_accuracy_percent':100*float(np.mean(vals)),
                        'seed_sample_sd_percentage_points':100*float(np.std(vals,ddof=1))})
        with (run/'repetition_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(curve_rows[0]));writer.writeheader();writer.writerows(curve_rows)
        result['status']='ABLATION_D25_CV_COMPLETED_NOT_OFFICIAL_TEST_NOT_CONFIRMATORY'
    except BaseException as exc:
        result.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
    finally:
        result['wall_seconds_this_invocation']=time.perf_counter()-start
        write(run/'ablation_d25_summary.json',result)
        (run/'ablation_d25_summary.txt').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        try:export(run)
        finally:(run/'RUNNING.lock').unlink(missing_ok=True)
    print(result['status'],flush=True)
    if 'error' in result:print('ERROR:',result['error'],flush=True)
    if 'comparison' in result:
        for row in result['comparison']['table']:print(row,flush=True)
    print('SEND:',run/'SEND_ablation_d25_review.zip',flush=True)
    return 0 if result['status']=='ABLATION_D25_CV_COMPLETED_NOT_OFFICIAL_TEST_NOT_CONFIRMATORY' else 1

if __name__=='__main__':raise SystemExit(main())
