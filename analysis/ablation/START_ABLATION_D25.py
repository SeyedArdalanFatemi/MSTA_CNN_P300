"""Graphical launcher; run with the original project .venv and empty parameters."""
from pathlib import Path
import sys,subprocess
import tkinter as tk
from tkinter import filedialog,messagebox
from msta import read

def choose(root,title,required):
    selected=filedialog.askdirectory(parent=root,title=title)
    if not selected:return None
    p=Path(selected)
    if not (p/required).is_file():raise ValueError('Select folder containing '+required)
    return p

def main():
    home=Path(__file__).resolve().parent
    root=tk.Tk();root.withdraw()
    try:
        answer=messagebox.askyesnocancel('Matched ablation d25',
          'Resume interrupted matched ablation d25 experiment?\n\nNO: new experiment (60 fits, Train/CV only).\nYES: resume existing matched ablation d25 output folder.\nCANCEL: exit.',parent=root)
        if answer is None:return 0
        cmd=[sys.executable,str(home/'RUN_ABLATION_D25.py')]
        if answer:
            run=choose(root,'Select matched ablation d25 output timestamp folder','protocol.json')
            if run is None:return 0
            p=read(run/'protocol.json')
            if p.get('stage')!='matched_ablation_d25_cv':raise ValueError('This is not an matched ablation d25 run')
            if (run/'RUNNING.lock').exists():raise ValueError('RUNNING.lock exists. Confirm the previous process has stopped before removing a stale lock.')
            reference=Path(p['reference_run']);data=Path(p['data_dir']);cmd+=['--resume-run',str(run)]
        else:
            candidates=[home.parent/'MSTA_CNN_release'/'runs'/'full_reproduction',Path.cwd()/'MSTA_CNN_release'/'runs'/'full_reproduction']
            reference=next((p for p in candidates if (p/'cv_summary.json').is_file()),None)
            if reference is None:reference=choose(root,'Select completed runs/full_reproduction folder (NOT final or references)','cv_summary.json')
            if reference is None:return 0
            release=reference.parent.parent
            candidates=[release.parent/'BCI_Comp_III_Wads_2004',release.parent/'MSTA_CNN_Speller'/'BCI_Comp_III_Wads_2004']
            data=next((p for p in candidates if all((p/f'Subject_{s}_Train.mat').is_file() for s in 'AB')),None)
            if data is None:data=choose(root,'Select original dataset folder containing Subject_A/B_Train.mat','Subject_A_Train.mat')
            if data is None:return 0
            if not (data/'Subject_B_Train.mat').is_file():raise ValueError('Subject_B_Train.mat is missing')
        cmd+=['--reference-run',str(reference),'--data-dir',str(data)]
    except Exception as exc:
        messagebox.showerror('Configuration error',str(exc),parent=root);return 1
    finally:root.destroy()
    return subprocess.call(cmd)

if __name__=='__main__':raise SystemExit(main())
