"""Unified entry point. Running without arguments prints instructions; no training starts."""
from pathlib import Path
import subprocess,sys
HOME=Path(__file__).resolve().parent
HELP='''MSTA-CNN release v2 -- CPU/offline reference workflow
Usage: python run.py COMMAND [arguments]

verify          Check the complete release manifest and frozen model bundles
self-test       Data, decoder, aggregation and supplementary-metric tests (no TF)
smoke-test      TensorFlow graph/update/save/reload tests
preprocess, cv, train, test, transfer, all
                Existing msta.py commands; pass --data-dir and --run-dir
ablation        Complete matched d25 ablation; --reference-run --data-dir --output-dir
features        Extract d25 features; --ablation-run --reference-run --data-dir --output-dir
supplementary   Recompute supplementary metrics from archived outputs (or --evaluation-dir)
assets          Rebuild the reported tables and figures from archived sources
transfer-train  Refit archived source-only recipes; --data-dir --run-dir

No command changes the scientific configuration by selecting on Test performance.
See RUN_FA.md for complete copyable examples. Legacy msta.py --profile ablation is d50;
use this entry point's ablation command for the retained d25 analysis.
'''
def main():
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help','help'):print(HELP);return 0
    cmd,*args=sys.argv[1:]
    if cmd=='self-test':
        if args:raise SystemExit('self-test takes no additional arguments')
        tasks=[['msta.py','self-test'],['analysis/ablation/RUN_ABLATION_D25.py','--self-test'],['analysis/features/RUN_FEATURES_D25.py','--self-test'],['analysis/supplementary/ANALYZE_METRICS.py','--self-test']]
        for script,*rest in tasks:
            code=subprocess.call([sys.executable,str(HOME/script),*rest])
            if code:return code
        print('ALL_NON_TF_SELF_TESTS_PASSED');return 0
    scripts={'ablation':'analysis/ablation/RUN_ABLATION_D25.py','features':'analysis/features/RUN_FEATURES_D25.py',
             'supplementary':'analysis/supplementary/ANALYZE_METRICS.py','assets':'paper_assets/BUILD_ASSETS.py','transfer-train':'TRAIN_TRANSFER_RECIPE.py'}
    if cmd in scripts:
        if cmd in ('supplementary','assets') and '--output-dir' not in args:args+=['--output-dir',str(HOME/'runs'/cmd)]
        return subprocess.call([sys.executable,str(HOME/scripts[cmd]),*args])
    if cmd in ('verify','smoke-test','preprocess','cv','train','test','transfer','all'):
        return subprocess.call([sys.executable,str(HOME/'msta.py'),cmd,*args])
    print(HELP);return 2
if __name__=='__main__':raise SystemExit(main())
