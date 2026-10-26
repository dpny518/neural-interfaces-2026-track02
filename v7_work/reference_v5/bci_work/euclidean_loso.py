"""Leakage-free train-subject LOSO for two temporal Euclidean ridge strengths."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'methods_draft'))
from riemann_features import estimate_source,transform_sessions
from classical_sweep import class_cell_weights,ridge_path
from evaluation import cell_mean_score


def main():
    start=time.perf_counter();cache=ROOT/'cache';out=ROOT/'cv_results';out.mkdir(exist_ok=True)
    with np.load(cache/'metadata.npz') as z:
        idx=np.flatnonzero(z['split']=='train');meta={k:z[k][idx] for k in z.files}
    cov=np.load(cache/'temporal_band_covs.npy',mmap_mode='r')[idx]
    oof={alpha:np.full((len(idx),3),np.nan) for alpha in [1000.,10000.]}
    details={alpha:[] for alpha in oof}
    center=dict(kind='logcov',alignment='euclidean',strength=1.,prior_count=8.,ema_rate=.02,shrinkage=.05,update_clip=10.)
    for subject in np.unique(meta['subject']):
        tr,va=meta['subject']!=subject,meta['subject']==subject
        tm,vm=({k:v[m] for k,v in meta.items()} for m in [tr,va])
        source=estimate_source(cov[tr],tm['stream_id'],shrinkage=.05)
        ft=transform_sessions(cov[tr],tm['stream_id'],source,**center)
        fv=transform_sessions(cov[va],vm['stream_id'],source,**center)
        weights=class_cell_weights(tm['y'],tm)
        for alpha,logits,mean,scale,coef,bias in ridge_path(ft,tm['y'],fv,weights,[1000.,10000.],'global',861):
            oof[alpha][va]=logits
            result=cell_mean_score(vm['y'],logits.argmax(1).astype('int64'),vm)
            result['held_subject']=str(subject);details[alpha].append(result)
            print(json.dumps({'held_subject':str(subject),'alpha':alpha,'score':result['balanced_accuracy_cell_mean'],'seconds':round(time.perf_counter()-start,1)}),flush=True)
        # Reusable OOF outputs are saved only when every row is complete.
    report=[]
    for alpha,logits in oof.items():
        assert np.isfinite(logits).all()
        pred=logits.argmax(1).astype('int64');summary=cell_mean_score(meta['y'],pred,meta)
        config=dict(family='ridge_euclidean',kind='logcov',bands=list(range(14)),crop=[[0,240],[240,480]],
                    source_prior='arithmetic',center_kwargs=center,scale_mode='global',alpha=alpha,shrinkage=.05)
        name='euclidean_temporal_002' if alpha==10000 else 'euclidean_temporal_001'
        np.savez_compressed(out/(name+'_oof.npz'),oof_logits=2*logits-1,oof_logits_onehot=logits,oof_pred=pred,
                            train_row_indices=idx,config_json=np.array(json.dumps(config)),score_encoding=np.array('plus_minus_one'))
        result={'name':name,'config':config,'summary':summary,'folds':details[alpha]}
        (out/(name+'_summary.json')).write_text(json.dumps(result,indent=2));report.append(result)
        print(json.dumps({'name':name,'oof_score':summary['balanced_accuracy_cell_mean'],'n_windows':len(pred),'n_cells':summary['n_cells']}),flush=True)
    (out/'euclidean_temporal_loso.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
