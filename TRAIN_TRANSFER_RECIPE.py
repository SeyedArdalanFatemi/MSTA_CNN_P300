"""Refit the archived source-only transfer recipes; no search or Test access."""
from pathlib import Path
import argparse,gc,os,copy
import numpy as np
import msta as c

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();data=args.data_dir.resolve();run=args.run_dir.resolve();home=Path(__file__).resolve().parent
    c.require(args.threads>0,'Invalid thread count')
    reference=home/'references/transfer';old,_=c.check_models(reference)
    for source in (data,reference):c.require(run!=source and source not in run.parents and run not in source.parents,'Input/output overlap')
    tf,env=c.runtime(args.threads);c.require(env==old['environment'],'Use the recorded reference environment')
    c.require(tf.keras.backend.image_data_format()=='channels_last','Expected channels_last')
    run.mkdir(parents=True,exist_ok=True);lock=run/'RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        frozen=copy.deepcopy(old)
        frozen.update(environment=env,threads=args.threads,origin_frozen_recipe_sha256=c.digest(reference/'protocol.json'),
            training_entry_sha256=c.digest(Path(__file__)),core_sha256=c.digest(Path(c.__file__)),
            description='Source-only refit of archived fixed recipes; does not repeat historical hyperparameter search.')
        modelroot=run/'final';modelroot.mkdir(exist_ok=True)
        if (modelroot/'protocol.json').exists():c.require(c.read(modelroot/'protocol.json')==frozen,'Frozen recipe changed')
        else:c.write(modelroot/'protocol.json',frozen)
        jobs=[]
        for sub in 'AB':
            X,y,meta,_=c.training_data(data,run,sub)
            stats=frozen['normalization'][sub]
            c.require(np.isclose(float(X.mean(dtype=np.float64)),stats['mean'],rtol=0,atol=1e-10) and np.isclose(float(X.std(dtype=np.float64)),stats['std'],rtol=1e-12),'Source normalization differs')
            X=np.array(X,dtype=np.float32,copy=True);X-=stats['mean'];X/=stats['scale'];c.validate_partition(X,y,meta,85)
            for task in frozen['tasks']:
                if task['subject']==sub:jobs.append(c.fit_task(tf,c,modelroot,task,c.digest(modelroot/'protocol.json'),X,y))
            del X;gc.collect()
        c.write(modelroot/'manifest.json',{'status':'FINAL_TRAINING_COMPLETED','protocol':frozen,'jobs':jobs})
        c.check_models(modelroot);print('TRANSFER_RECIPE_REFIT_COMPLETED_TEST_NOT_READ')
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
