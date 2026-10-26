"""Run using the existing project interpreter; no command-line parameters required."""
from pathlib import Path
import sys,subprocess,tkinter as tk
from tkinter import filedialog,messagebox
from msta import read

def choose(root,title,required):
    value=filedialog.askdirectory(parent=root,title=title)
    if not value:return None
    p=Path(value)
    if not (p/required).is_file():raise ValueError('Select folder containing '+required)
    return p

def main():
    root=tk.Tk();root.withdraw();home=Path(__file__).resolve().parent
    try:
        resume=messagebox.askyesnocancel('Features d25','Resume this feature analysis?\nNO: new analysis (no training).\nYES: resume features output.\nCANCEL: exit.',parent=root)
        if resume is None:return 0
        cmd=[sys.executable,str(home/'RUN_FEATURES_D25.py')]
        if resume:
            run=choose(root,'Select FEATURES output timestamp folder','protocol.json')
            if run is None:return 0
            p=read(run/'protocol.json')
            if p.get('stage')!='features_d25':raise ValueError('Not a features_d25 output')
            if (run/'RUNNING.lock').exists():raise ValueError('Lock exists: confirm previous process has stopped before removing a stale RUNNING.lock')
            ablation,reference,data=[Path(p[k]) for k in ('ablation_run','reference_run','data_dir')]
            cmd+=['--resume-run',str(run)]
        else:
            ablation=choose(root,'Select COMPLETED ABLATION d25 timestamp folder','ablation_d25_summary.json')
            if ablation is None:return 0
            p=read(ablation/'protocol.json');reference=Path(p['reference_run']);data=Path(p['data_dir'])
            if not (reference/'cv').is_dir():reference=choose(root,'Select original PRIMARY CV run (with cv/ and all weights)','cv_summary.json')
            if reference is None:return 0
            if not all((data/f'Subject_{s}_Train.mat').is_file() for s in 'AB'):
                data=choose(root,'Select original Train data folder','Subject_A_Train.mat')
            if data is None:return 0
        cmd+=['--ablation-run',str(ablation),'--reference-run',str(reference),'--data-dir',str(data)]
    except Exception as exc:messagebox.showerror('Configuration error',str(exc),parent=root);return 1
    finally:root.destroy()
    return subprocess.call(cmd)

if __name__=='__main__':raise SystemExit(main())
