import msta as core
from pathlib import Path
read,require=core.read,core.require

def check_reference(root):
    require((root/'cv_summary.json').is_file() and (root/'protocol.json').is_file(),
            'Select the original completed primary CV run containing cv_summary.json and protocol.json.')
    require((root/'cv').is_dir(),
            'This is a summary-only folder. Select the ORIGINAL completed CV run with its cv/ folder and all 60 best.keras checkpoints, not release/results or references. The same reference used for channel normalization is suitable.')
    p=read(root/'protocol.json');s=read(root/'cv_summary.json')
    require(p['configuration']==core.configuration('primary'),'Reference must be completed PRIMARY batch32/dropout0.25 CV')
    require(p['code_sha256']==core.digest(Path(core.__file__)) and p['source_sha256']==core.TRAIN_HASHES,'Reference code or source differs')
    require(s['status']=='CV_COMPLETED' and s['protocol_sha256']==core.digest(root/'protocol.json'),'Reference incomplete')
    expected={(a,v,se,f) for a in 'AB' for v in ('none','full') for se in core.SEEDS for f in range(1,6)}
    require(len(s['jobs'])==60 and {(j['subject'],j['variant'],j['seed'],j['fold']) for j in s['jobs']}==expected,'Reference coverage differs')
    for j in s['jobs']:
        folder=root/'cv'/f'{j["subject"]}_{j["variant"]}_{j["seed"]}_fold{j["fold"]}'
        require(read(folder/'report.json')==j,'Reference job changed')
        core.verify_cv_job(folder,j,s['protocol_sha256'])
    require(core.task_summary(s['jobs'])==s['summary'],'Reference aggregate mismatch')
    return p,s

