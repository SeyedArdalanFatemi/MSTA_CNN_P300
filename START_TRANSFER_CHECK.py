"""Run frozen transfer refit, evaluation and comparison; no hyperparameter search."""
from pathlib import Path
import argparse,subprocess,sys,json,zipfile
import msta as c
HOME=Path(__file__).resolve().parent

def compare(actual,reference):
    def key(j):
        t=j.get('task',j)
        return j['source'],j['target'],t['variant'],t['seed']
    expected={(a,b,v,s) for a,b in [('A','B'),('B','A')] for v in ('none','full') for s in c.SEEDS}
    left={key(j):j for j in actual['jobs']};right={key(j):j for j in reference['jobs']}
    c.require(len(actual['jobs'])==len(reference['jobs'])==12 and set(left)==set(right)==expected,'Incomplete or duplicate transfer jobs')
    rows=[]
    for k in sorted(expected):
        a,b=left[k],right[k]
        c.require(len(a['character_curve'])==len(b['character_curve'])==15,'Incomplete curves')
        for x,y in zip(a['character_curve'],b['character_curve']):
            c.require(x['sequences']==y['sequences'] and x['characters']==y['characters']==100,'Wrong evaluation geometry')
        diffs=[x['correct']-y['correct'] for x,y in zip(a['character_curve'],b['character_curve'])]
        rows.append(dict(source=k[0],target=k[1],variant=k[2],seed=k[3],new_correct15=a['character_curve'][-1]['correct'],reference_correct15=b['character_curve'][-1]['correct'],correct_differences_1_to_15=diffs,all_character_counts_match=not any(diffs)))
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path);ap.add_argument('--run-dir',type=Path,default=HOME/'runs/transfer_check');ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if args.data_dir is None:
        import tkinter as tk
        from tkinter import filedialog
        ui=tk.Tk();ui.withdraw()
        selected=filedialog.askdirectory(title='Select folder containing Subject_A/B_Train/Test.mat')
        ui.destroy()
        if not selected:raise SystemExit('No data folder selected; no training started.')
        args.data_dir=Path(selected)
    data=args.data_dir.resolve();run=args.run_dir.resolve()
    c.require(args.threads>0,'Threads must be positive')
    for name in [f'Subject_{s}_{part}.mat' for s in 'AB' for part in ('Train','Test')]:c.require((data/name).is_file(),'Missing file: '+name)
    c.require(data!=run and data not in run.parents and run not in data.parents,'Data/output overlap')
    run.mkdir(parents=True,exist_ok=True)
    final=run/'transfer_check.json';final.unlink(missing_ok=True)
    review=run/'SEND_transfer_check.zip';review.unlink(missing_ok=True)
    def call(script,*parameters):subprocess.run([sys.executable,str(HOME/script),*map(str,parameters)],check=True)
    try:
        call('TRAIN_TRANSFER_RECIPE.py','--data-dir',data,'--run-dir',run/'training','--threads',args.threads)
        call('msta.py','transfer','--data-dir',data,'--run-dir',run/'evaluation_run','--model-dir',run/'training/final','--threads',args.threads)
        summary=run/'evaluation_run/evaluation/cross_subject/summary.json'
        actual=c.read(summary);c.require(actual['status']=='EVALUATION_COMPLETED' and actual['identity']['mode']=='cross_subject','Incomplete evaluation')
        reference=HOME/'results/transfer_reference.json';rows=compare(actual,c.read(reference));match=all(r['all_character_counts_match'] for r in rows)
        report={'status':'TRANSFER_CHECK_COMPLETED_COUNTS_MATCH' if match else 'TRANSFER_CHECK_COMPLETED_DIFFERENCES_REQUIRE_REVIEW','scope':'Character counts at all 15 repetitions; matching counts do not prove identical weights or event probabilities. Not a new confirmatory test.','evaluation_sha256':c.digest(summary),'reference_sha256':c.digest(reference),'rows':rows}
        c.write(final,report)
        with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
            for f in [final,summary,run/'training/final/manifest.json',run/'training/final/protocol.json']:z.write(f,f.relative_to(run))
        print(report['status']);print('SEND:',review)
        return 0 if match else 2
    except Exception as exc:
        c.write(final,{'status':'FAILED','error':str(exc)});raise
if __name__=='__main__':raise SystemExit(main())
