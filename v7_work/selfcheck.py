"""Synthetic checks of streaming state and exclusion; never score public EEG."""
import copy
import json
import tempfile
import protocol as pr
import numpy as np
import torch
from runtime import MultiscaleNet, Decoder, normalized_covariance
from riemann_features import bank_covariances, estimate_source
from training import fit_neural


def run():
    torch.set_num_threads(2)
    rng=np.random.default_rng(901)
    x=rng.normal(size=(12,41,480)).astype(np.float32)
    p=pr.plan();bands,views=p['bands'],p['views']
    cov=np.concatenate([bank_covariances(x[:,:,lo:hi]) for lo,hi in views],axis=1)
    config=dict(n_bands=7,n_views=3,n_chans=41,n_filters=16,hidden=128,dropout=.5)
    neural=[]
    for seed in [33,42]:
        torch.manual_seed(seed)
        neural.append(dict(config=config,state_dict=MultiscaleNet(**config).state_dict(),
                           source=torch.tensor(normalized_covariance(cov,[6,13,20]).mean(0)),
                           broadband_indices=[6,13,20],alignment=p['neural_alignment']))
    d=14*861
    classic=dict(kind='covariance',source=estimate_source(cov[:,:14]),mean=np.zeros(d),
                 scale=np.ones(d),coef=np.zeros((d,3)),bias=np.zeros(3),temperature=.125)
    ck=dict(neural=neural,classical=[classic],weights=[.45,.45,.1],bands=bands,views=views,
            ch_names=[str(i) for i in range(41)])
    template=Decoder(ck,{'device':'cpu','n_classes':3})
    a,b=copy.deepcopy(template),copy.deepcopy(template)
    bulk=a.predict_proba(torch.tensor(x))
    seq=np.concatenate([b.predict_proba(torch.tensor(row[None])) for row in x])
    np.testing.assert_array_equal(bulk,seq)
    a.reset_state();np.testing.assert_array_equal(a.predict_proba(x),bulk)
    prefix=copy.deepcopy(template);prefix.predict_proba(x[:5]);clone=copy.deepcopy(prefix)
    np.testing.assert_array_equal(prefix.predict_proba(x[5:]),clone.predict_proba(x[5:]))
    changed=x.copy();changed[6:]*=19
    np.testing.assert_array_equal(copy.deepcopy(template).predict_proba(changed)[:6],bulk[:6])
    np.testing.assert_array_equal(copy.deepcopy(template).predict_proba(x[:6]),bulk[:6])
    assert template.n_seen==0 and all(m.aligner.n_seen==0 for m in template.neural)
    original=x.copy();copy.deepcopy(template).predict(x);np.testing.assert_array_equal(x,original)
    assert template.predict(x[:0]).shape==(0,)
    for n in [1,4]:
        y=copy.deepcopy(template).predict(x[:n]);assert y.shape==(n,) and y.dtype==np.int64
    subjects=np.array(['P01']*6+['P02']*6)
    try:
        fit_neural(cov,np.arange(12)%3,np.ones(12),subjects,['P01'],config,'reconstructed_v6',33,1,'cpu')
    except AssertionError:pass
    else:raise AssertionError('Held-out subject guard failed')
    fit,info=fit_neural(cov,np.arange(12)%3,np.ones(12),subjects,['P03'],config,'reconstructed_v6',33,1,'cpu')
    assert all(torch.isfinite(t).all() for t in fit.state_dict().values())
    fit,info=fit_neural(cov,np.arange(12)%3,np.ones(12),subjects,['P03'],config,'masked_ssl',33,1,'cpu')
    assert info['ssl']['labels_used'] is False and info['ssl']['n_fitting_windows']==12
    out={'passed':True,'data':'synthetic only','checks':['batch/sequential probability identity',
         'reset restoration','deepcopy after adaptation','prefix causality','template isolation',
         'input immutability','empty and integer output shapes','excluded-subject rejection',
         'one-epoch fit smoke','fit-only SSL smoke']}
    pr.json_write(pr.ROOT/'synthetic_check.json',out)
    print(json.dumps(out))


if __name__=='__main__':run()
