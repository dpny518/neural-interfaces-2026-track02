"""Fixed original ridge and replacement inner-LOSO classical panel."""
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve
from protocol import ROOT, subset, fitting_record, save_checkpoint, load_checkpoint, cell_scores
from runtime import ClassicalMember
from riemann_features import estimate_source, CausalCovarianceFeatures
from riemann_v5_runtime import V5AlignedCovarianceStream
from classical_sweep import class_cell_weights, ridge_path

CENTER=dict(kind='logcov',alignment='euclidean',strength=1.,prior_count=8.,
            ema_rate=.02,shrinkage=.05,update_clip=10.)


def tensorize(obj):
    if isinstance(obj,np.ndarray):return torch.from_numpy(obj.copy())
    if isinstance(obj,dict):return {k:tensorize(v) for k,v in obj.items()}
    if isinstance(obj,list):return [tensorize(v) for v in obj]
    return obj


def features(covs, groups, source, kind):
    result=[]
    # Process in supplied chronological row order, restoring per-session state.
    active=None
    for cov, group in zip(covs,groups):
        if group!=active:
            stream=(CausalCovarianceFeatures(source,**CENTER) if kind=='covariance'
                    else V5AlignedCovarianceStream(source,**CENTER))
            active=group
        if kind=='covariance':f=stream.transform_one(cov)
        else:
            c=stream.transform_covariance_one(cov)
            f=np.log(np.maximum(np.diagonal(c,axis1=-2,axis2=-1),1e-12)).flatten()
        result.append(f)
    return np.asarray(result,dtype=np.float32)


def fit_classical(cov,m,rows,excluded,kind,path,stage='cv'):
    if Path(path).exists():return load_checkpoint(path)
    fm=subset(m,rows)
    assert not set(fm['subject'])&set(excluded)
    source=estimate_source(cov[rows],fm['stream_id'],shrinkage=.05)
    ft=features(cov[rows],fm['stream_id'],source,kind)
    weights=class_cell_weights(fm['y'],fm)
    if kind in ['covariance','linear']:
        alpha=10000. if kind=='covariance' else 10.
        scale_mode='global' if kind=='covariance' else 'standard'
        _,_,mean,scale,coef,bias=next(ridge_path(ft,fm['y'],ft[:1],weights,[alpha],scale_mode,861))
        if isinstance(coef,tuple):coef=coef[0]@(coef[1]@coef[2])
        extra={}
    else:
        assert kind=='rbf'
        wn=weights/weights.sum()
        mean=np.einsum('n,nd->d',wn,ft)
        scale=np.sqrt(np.maximum(np.einsum('n,nd,nd->d',wn,ft-mean,ft-mean),1e-8))
        z=(ft-mean)/scale
        gamma=1./z.shape[1]
        dist=np.maximum((z*z).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*z@z.T,0.)
        kernel=np.exp(-gamma*dist)
        bias=np.einsum('n,nc->c',wn,np.eye(3)[fm['y']])
        rw=np.sqrt(weights)
        system=kernel*rw[:,None]*rw[None,:]+np.eye(len(z))
        coef=rw[:,None]*solve(system,rw[:,None]*(np.eye(3)[fm['y']]-bias),assume_a='pos')
        extra={'centers':z,'gamma':gamma}
    data=tensorize(dict(kind=kind,source=source,mean=mean,scale=scale,coef=coef,
                        bias=bias,temperature=.125,**extra))
    provenance=fitting_record(m,rows,excluded,stage,'classical_'+kind)
    save_checkpoint(path,data,provenance)
    return load_checkpoint(path)


def predict_classical(data,covs,groups):
    member=ClassicalMember(data);out=[];active=None
    for c,g in zip(covs,groups):
        if g!=active:member.reset_state();active=g
        out.append(member.predict_one(c))
    return np.asarray(out)


def nested_choice(cov,m,fit_rows,outer_excluded,out,stage='cv'):
    from protocol import json_write, sha
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fitting_subjects=sorted(set(m['subject'][fit_rows]))
    all_cells={k:[] for k in ['linear','rbf']}; records=[]
    for inner in fitting_subjects:
        rows=fit_rows[m['subject'][fit_rows]!=inner]
        held=fit_rows[m['subject'][fit_rows]==inner]
        excluded=sorted(set(outer_excluded)|{inner})
        for kind in ['linear','rbf']:
            path=out/f'inner_{inner}_{kind}.pt'
            data=fit_classical(cov,m,rows,excluded,kind,path,stage='inner_'+stage)
            probs=predict_classical(data,cov[held],m['stream_id'][held])
            scores,keys=cell_scores(m['y'][held],probs.argmax(1),subset(m,held))
            all_cells[kind].extend(scores.tolist())
            pred_path=out/f'inner_{inner}_{kind}_predictions.npz'
            np.savez_compressed(pred_path,rows=held,probabilities=probs,y=m['y'][held])
            records.append({'inner_heldout':inner,'outer_excluded':outer_excluded,
                            'kind':kind,'checkpoint_sha256':sha(path),
                            'predictions_sha256':sha(pred_path),'cells':scores.tolist()})
    means={k:float(np.mean(v)) for k,v in all_cells.items()}
    choice=max(['linear','rbf'],key=lambda k:means[k])
    json_write(out/'selection.json',{'choice':choice,'inner_scores':means,
                                    'fitting_subjects':fitting_subjects,'outer_excluded':outer_excluded,
                                    'inner_records':records})
    return choice
