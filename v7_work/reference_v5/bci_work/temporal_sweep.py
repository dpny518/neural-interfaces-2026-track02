"""Validation-only comparison of temporal covariance views.

All source statistics, scalers, and classifier coefficients use train only.
Cached features are window-local. Session means are updated after prediction.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'methods_draft'))
import numpy as np
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from riemann_features import bank_covariances, static_features, mean_by_group, center_feature_sessions
from evaluation import load_cache, cell_mean_score


def cell_class_weights(meta):
    keys = list(zip(meta['subject'], meta['session'], meta['interface'], meta['y']))
    counts = {k: keys.count(k) for k in set(keys)}
    w = np.array([1. / counts[k] for k in keys])
    return w / w.mean()


def fit_head(F, y, weights, alpha, scaling='feature'):
    mean = np.average(F, axis=0, weights=weights)
    scale = np.sqrt(np.average((F-mean)**2, axis=0, weights=weights))
    if scaling == 'global':
        scale[:] = np.sqrt(np.mean(scale**2))
    scale = np.maximum(scale, 1e-5)
    Z = ((F-mean)/scale).astype(np.float32)
    model = RidgeClassifier(alpha=alpha, solver='cholesky')
    model.fit(Z, y, sample_weight=weights)
    return dict(scaler_mean=mean.astype(np.float32), scaler_scale=scale.astype(np.float32),
                coef=model.coef_.astype(np.float32), intercept=model.intercept_.astype(np.float32))


def logits_head(F, head):
    return ((F-head['scaler_mean'])/head['scaler_scale']) @ head['coef'].T + head['intercept']


def main():
    threadpool_limits(limits=2)
    X, meta, info = load_cache()
    cache = Path(__file__).parent / 'cache'
    out = Path(__file__).parent / 'temporal_results'
    out.mkdir(exist_ok=True)
    train, val = meta['split']=='train', meta['split']=='val'
    trainmeta = {k:v[train] for k,v in meta.items()}
    valmeta = {k:v[val] for k,v in meta.items()}
    weights = cell_class_weights(trainmeta)
    plans = [
        ('early', 0, 240, False, False),
        ('late', 240, 480, False, False),
        ('full_car', 0, 480, True, False),
        ('full_z', 0, 480, False, True),
    ]
    featurefiles = {}
    for name, lo, hi, car, zscore in plans:
        fpath = cache / f'raw_{name}_log_features.npy'
        if not fpath.exists():
            xx = np.array(X[:,:,lo:hi], copy=True)
            if zscore:
                xx -= xx.mean(-1,keepdims=True)
                xx /= np.maximum(xx.std(-1,keepdims=True), 1e-6)
            C = bank_covariances(xx, common_average=car)
            F = static_features(C, 'logcov', shrinkage=.05).astype(np.float32)
            np.save(fpath,F)
            del xx,C,F
        featurefiles[name]=fpath
        print('CACHED',name,flush=True)
    records=[]
    for name in ['early','late','early_late','full_car','full_z']:
        if name=='early_late':
            F=np.concatenate([np.load(featurefiles['early']),np.load(featurefiles['late'])],axis=1)
        else:
            F=np.load(featurefiles[name])
        source=mean_by_group(F[train],meta['stream_id'][train])
        for strength, prior, ema in [(0.,16.,.02),(1.,8.,.02),(1.,32.,.02)]:
            cfg=dict(strength=strength,prior_count=prior,ema_rate=ema)
            Z=center_feature_sessions(F,meta['stream_id'],source,**cfg)
            for scaling in ['global','feature']:
                for alpha in [100.,1000.,10000.]:
                    t=time.time()
                    head=fit_head(Z[train],meta['y'][train],weights,alpha,scaling)
                    scores=logits_head(Z[val],head)
                    metric=cell_mean_score(meta['y'][val],scores.argmax(1),valmeta)
                    key=f'{name}_s{strength:g}_p{prior:g}_{scaling}_a{alpha:g}'
                    record=dict(key=key,view=name,center=cfg,scaling=scaling,alpha=alpha,
                                val=metric['balanced_accuracy_cell_mean'],cells=metric['cells'])
                    records.append(record)
                    np.savez(out/(key+'.npz'),source_mean=source,**head,val_logits=scores,
                             config_json=np.array(json.dumps(record)))
                    print(key,round(record['val'],5),'seconds',round(time.time()-t,2),flush=True)
            (out/'scores.json').write_text(json.dumps(sorted(records,key=lambda r:-r['val']),indent=2))
    print('TOP',json.dumps(sorted(records,key=lambda r:-r['val'])[:8]),flush=True)


if __name__=='__main__':
    main()
