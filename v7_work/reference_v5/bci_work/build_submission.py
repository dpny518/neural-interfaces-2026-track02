"""Build a standalone, tensor-only checkpoint and two-file submission ZIP."""
from __future__ import annotations
import argparse, json, hashlib, sys, zipfile
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def tensorize(value):
    if isinstance(value,np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {k:tensorize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [tensorize(v) for v in value]
    return value


def views(name):
    if name=='early_late' or name=='temporal':
        return [dict(crop=[0,240]),dict(crop=[240,480])]
    crop={'early':[0,240],'late':[240,480]}.get(name,[0,480])
    return [dict(crop=crop,car=name=='full_car',zscore=name=='full_z')]


def pack_member(selection):
    kind=selection['kind']
    path=HERE/selection['file']
    base=dict(weight=float(selection['weight']),temperature=float(selection.get('temperature',1.)))
    if kind in ('temporal_ridge','classical_ridge','euclidean_ridge'):
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        cfg=json.loads(str(arrays['config_json']))
        m=dict(base,type='linear',shrinkage=.05)
        for field in ('scaler_mean','scaler_scale','coef','intercept'):
            m[field]=arrays[field]
        if kind=='temporal_ridge':
            m.update(family='logcenter',views=views(cfg['view']),raw_kind='logcov',
                     source_mean=arrays['source_mean'],center_kwargs=cfg['center'])
            m['coef']=m['coef'].T
        elif kind=='classical_ridge':
            m.update(family='logcenter',views=views('full'),raw_kind=cfg['kind'],
                     source_mean=arrays['source_mean'],center_kwargs=cfg['center_kwargs'],
                     feature_indices=arrays['feature_indices'])
        else:
            temporal=isinstance(cfg['crop'][0],list)
            m.update(family='euclidean',views=views('temporal' if temporal else 'full'),
                     center_kwargs=cfg['center_kwargs'],feature_indices=arrays['feature_indices'])
            for field in ('source_cov','source_logcov','source_logpower'):
                m[field]=arrays[field]
        m['provenance']=dict(filename=path.name,config=cfg)
        return tensorize(m)
    if kind in ('neural_cov','raw'):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        return tensorize(dict(base,type=kind,views=views(selection.get('view','full')),checkpoint=ckpt))
    raise ValueError(kind)


def strip_future(source):
    return '\n'.join(line for line in source.splitlines() if not line.startswith('from __future__ import'))+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'deliverables')
    args=ap.parse_args()
    selection=json.loads(args.selection.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    folder=args.out/'submission'
    folder.mkdir(exist_ok=True)
    members=[pack_member(x) for x in selection['members']]
    info=json.loads((HERE/'cache/dataset_info.json').read_text())
    checkpoint=dict(schema_version=1,ch_names=info['ch_names'],members=members,
                    dataset='nm000290',version='1.0.0',
                    training_subjects=info['subjects_by_split']['train'],
                    validation_subjects=info['subjects_by_split']['val'],
                    selection=selection)
    torch.save(checkpoint,folder/'weights.pt')
    header='''"""Neural Interfaces 2026 Track 2: causal EEG decoding.

Input is the official microvolt-scaled (B,41,480) tensor at 120 Hz.
Learned parameters and source priors use only the six reference training subjects.
All online state is reset by reset_state; batching processes windows sequentially.
There are no network calls, installs, label lookups, or external runtime files.
"""\n'''
    sources=[ROOT/'methods_draft/riemann_features.py']
    if any(x['type']=='neural_cov' for x in members):sources.append(HERE/'neural_cov.py')
    if any(x['type']=='raw' for x in members):sources.append(HERE/'neural_raw.py')
    # Runtime classes resolve network names only when the corresponding type
    # occurs in the checkpoint. Always import torch for safe weight loading.
    source=header+'\nimport numpy as np\nimport torch\n'
    for p in sources+[HERE/'deploy_runtime.py']:
        source+='\n# ---- '+p.name+' ----\n'+strip_future(p.read_text())
    compile(source,'submission.py','exec')
    (folder/'submission.py').write_text(source)
    archive=args.out/'submission.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in ('submission.py','weights.pt'):
            z.write(folder/name,arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==['submission.py','weights.pt']
        assert z.testzip() is None
    summary=dict(zip=str(archive),bytes=archive.stat().st_size,
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 members=len(members),selection=selection)
    (args.out/'build_info.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
