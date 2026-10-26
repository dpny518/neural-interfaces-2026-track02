"""Small validation-only multinomial logistic check on temporal log features."""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'methods_draft'))
from riemann_features import mean_by_group,center_feature_sessions
from classical_sweep import class_cell_weights
from evaluation import cell_mean_score


def main():
    start=time.perf_counter(); cache=ROOT/'cache';out=ROOT/'logreg';out.mkdir(exist_ok=True)
    with np.load(cache/'metadata.npz') as z:meta={k:z[k] for k in z.files}
    tr,va=meta['split']=='train',meta['split']=='val'
    trmeta,vmeta=({k:v[m] for k,v in meta.items()} for m in [tr,va])
    raw=np.concatenate([np.load(cache/'raw_early_log_features.npy'),np.load(cache/'raw_late_log_features.npy')],axis=1)
    weights=class_cell_weights(trmeta['y'],trmeta);wn=weights/weights.sum()
    source=mean_by_group(raw[tr],trmeta['stream_id']);center=dict(strength=1.,prior_count=8.,ema_rate=.02)
    ft=center_feature_sessions(raw[tr],trmeta['stream_id'],source,**center)
    fv=center_feature_sessions(raw[va],vmeta['stream_id'],source,**center)
    mean=np.einsum('n,nd->d',wn,ft);delta=ft-mean
    var=np.einsum('n,nd,nd->d',wn,delta,delta);records=[]
    for scaling in ['global','standard']:
        scale=np.sqrt(np.maximum(var,1e-8)) if scaling=='standard' else np.full(len(var),np.sqrt(var.mean()))
        xt,xv=delta/scale,(fv-mean)/scale
        for C in [1e-4,1e-3,1e-2,1e-1]:
            name=f'logreg_{len(records):03d}'
            model=LogisticRegression(C=C,solver='lbfgs',max_iter=250,tol=1e-5)
            model.fit(xt,trmeta['y'],sample_weight=weights)
            logits=model.decision_function(xv);pred=logits.argmax(1).astype('int64')
            scores=cell_mean_score(vmeta['y'],pred,vmeta)
            config=dict(name=name,family='logreg',kind='logcov',shrinkage=.05,bands=list(range(14)),crop=[[0,240],[240,480]],
                        center_kwargs=center,scale_mode=scaling,C=C,feature_dim=len(source),score_validation=scores['balanced_accuracy_cell_mean'])
            np.savez_compressed(out/(name+'.npz'),source_mean=source,scaler_mean=mean,scaler_scale=scale,coef=model.coef_.T,intercept=model.intercept_,
                                feature_indices=np.arange(len(source)),val_logits=logits,val_pred=pred,config_json=np.array(json.dumps(config)))
            (out/(name+'_validation.json')).write_text(json.dumps(scores,indent=2));records.append(config)
            (out/'results.json').write_text(json.dumps(sorted(records,key=lambda r:-r['score_validation']),indent=2))
            print(json.dumps({**config,'iterations':model.n_iter_.tolist(),'seconds':round(time.perf_counter()-start,1)}),flush=True)

if __name__=='__main__':main()
