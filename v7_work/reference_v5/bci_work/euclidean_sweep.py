"""Focused validation-only check of actual covariance whitening + tangent map."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "methods_draft"))
from riemann_features import bank_covariances, estimate_source, transform_sessions, spd_power
from classical_sweep import class_cell_weights, ridge_path
from evaluation import cell_mean_score


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--temporal',action='store_true');args=parser.parse_args()
    start = time.perf_counter()
    cache, out = ROOT / "cache", ROOT / ("euclidean_temporal" if args.temporal else "euclidean")
    out.mkdir(exist_ok=True)
    with np.load(cache / "metadata.npz") as z:
        meta = {k: z[k] for k in z.files}
    tr, va = meta["split"] == "train", meta["split"] == "val"
    train_meta, val_meta = ({k:v[m] for k,v in meta.items()} for m in [tr,va])
    if args.temporal:
        cp=cache/'temporal_band_covs.npy'
        if not cp.exists():
            x=np.load(cache/'windows.npy',mmap_mode='r')
            c=np.concatenate([bank_covariances(x[:,:,:240]),bank_covariances(x[:,:,240:])],axis=1)
            np.save(cp,c);del c
        cov=np.load(cp,mmap_mode='r')
    else:
        cov = np.load(cache / "band_covs.npy", mmap_mode="r")
    source = estimate_source(cov[tr], train_meta["stream_id"], shrinkage=.05)
    weights = class_cell_weights(train_meta["y"], train_meta)
    allresults=[]; counter=0; retained=[]
    options=([
        ("arithmetic",1.,8.,.02), ("logeuclidean",1.,8.,.02),
        ("arithmetic",1.,4.,.05)] if args.temporal else [
        ("arithmetic",1.,4.,.05), ("arithmetic",1.,16.,.02),
        ("logeuclidean",1.,4.,.05), ("arithmetic",0.,16.,.02)])
    for prior_kind, strength, prior_count, ema_rate in options:
        local_source = {k: np.array(v).copy() if k!='shrinkage' else v for k,v in source.items()}
        if prior_kind == 'logeuclidean':
            vals,vecs=np.linalg.eigh(source['logcov'])
            local_source['cov']=(vecs*np.exp(vals)[:,None,:])@np.swapaxes(vecs,-1,-2)
        kwargs=dict(kind='logcov',alignment='euclidean',strength=strength,
                    prior_count=prior_count,ema_rate=ema_rate,shrinkage=.05,update_clip=10.)
        ftr=transform_sessions(cov[tr],train_meta['stream_id'],local_source,**kwargs)
        fva=transform_sessions(cov[va],val_meta['stream_id'],local_source,**kwargs)
        views=([list(range(14)),[1,2,3,4,5,8,9,10,11,12],[6,13]] if args.temporal else [[0,1,2,3,4,5,6],[1,2,3,4,5],[6]])
        for bands in views:
            idx=np.concatenate([np.arange(b*861,(b+1)*861) for b in bands])
            for scaling in (['global','standard','band'] if args.temporal else ['standard','band']):
                group_scores=[]
                for alpha,logits,mean,scale,coef,bias in ridge_path(ftr[:,idx],train_meta['y'],fva[:,idx],weights,[100.,1000.,10000.],scaling,861):
                    pred=logits.argmax(1).astype('int64')
                    scores=cell_mean_score(val_meta['y'],pred,val_meta)
                    score=scores['balanced_accuracy_cell_mean']
                    config=dict(name=f'euclidean_{counter:03d}',family='ridge_euclidean',kind='logcov',bands=bands,crop=([[0,240],[240,480]] if args.temporal else [0,480]),shrinkage=.05,
                        source_prior=prior_kind,center_kwargs=kwargs,scale_mode=scaling,alpha=alpha,score_validation=score,feature_dim=len(idx))
                    allresults.append(config);group_scores.append(round(score,4));counter+=1
                    if len(retained)<12 or score>min(t[0] for t in retained):
                        if isinstance(coef,tuple):coef=coef[0]@(coef[1]@coef[2])
                        np.savez_compressed(out/(config['name']+'.npz'),source_cov=local_source['cov'],source_logcov=local_source['logcov'],source_logpower=local_source['logpower'],
                            scaler_mean=mean,scaler_scale=scale,coef=coef,intercept=bias,feature_indices=idx,val_logits=logits,val_pred=pred,config_json=np.array(json.dumps(config)))
                        (out/(config['name']+'_validation.json')).write_text(json.dumps(scores,indent=2))
                        retained.append((score,config['name']));retained.sort(reverse=True)
                        if len(retained)>12:
                            _,remove=retained.pop();(out/(remove+'.npz')).unlink(missing_ok=True);(out/(remove+'_validation.json')).unlink(missing_ok=True)
                print(json.dumps({'prior_kind':prior_kind,'adaptation':[strength,prior_count,ema_rate],'bands':bands,'scale':scaling,'scores':group_scores,'best':retained[0],'seconds':round(time.perf_counter()-start,1)}),flush=True)
        (out/'results.json').write_text(json.dumps(sorted(allresults,key=lambda r:-r['score_validation']),indent=2))
        (out/'retained.json').write_text(json.dumps(retained,indent=2))
    print('DONE',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
