"""Resumable, frozen LOSO fits. No outer CV scores are computed here."""
import argparse
import json
import time
import platform
from pathlib import Path
import protocol as pr
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from runtime import MultiscaleNet, MultiscaleAligner, normalized_covariance
from riemann_features import bank_covariances
from classical_sweep import class_cell_weights
from training import fit_neural
from classical import fit_classical, nested_choice, predict_classical

ROOT=pr.ROOT


def grid(recipe):
    p=pr.plan()
    g=p['expanded_grid'] if recipe=='expanded_grid' else p
    return g['bands'],g['views']


def prepare_features():
    p=pr.plan();X=np.load(ROOT/'cache/windows.npy',mmap_mode='r')
    assert X.shape==(2655,41,480)
    for name in ['reconstructed_v6','expanded_grid']:
        bands,views=grid(name);path=ROOT/f'cache/{name}_cov.npy'
        manifest=path.with_suffix('.json')
        if path.exists() and manifest.exists():
            record=json.loads(manifest.read_text())
            assert record['sha256']==pr.sha(path) and record['windows_sha256']==pr.sha(ROOT/'cache/windows.npy')
            continue
        out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(X),len(bands)*len(views),41,41))
        for k,(lo,hi) in enumerate(views):
            out[:,k*len(bands):(k+1)*len(bands)]=bank_covariances(X[:,:,lo:hi],bands=bands)
        out.flush();del out
        pr.json_write(manifest,{'sha256':pr.sha(path),'windows_sha256':pr.sha(ROOT/'cache/windows.npy'),
                               'bands':bands,'views':views,'cross_window_statistics':False,'labels_used':False})
        print('FEATURES_READY',name,flush=True)


def aligned_cache(m,fit,held,excluded,recipe,out,stage='cv'):
    bands,views=grid(recipe)
    name='expanded_grid' if recipe=='expanded_grid' else 'reconstructed_v6'
    rawpath=ROOT/f'cache/{name}_cov.npy'
    raw=np.load(rawpath,mmap_mode='r')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sourcepath=out/'source.pt';recordpath=out/'cache_manifest.json'
    bb=[i*len(bands)+6 for i in range(len(views))]
    if recordpath.exists():
        r=json.loads(recordpath.read_text())
        assert r['raw_sha256']==pr.sha(rawpath)
        assert r['fit_rows']==fit.tolist() and r['held_rows']==held.tolist()
        for f,h in r['files'].items():assert pr.sha(out/f)==h
        source=pr.load_checkpoint(sourcepath)['source']
        return np.load(out/'fit.npy',mmap_mode='r'),np.load(out/'held.npy',mmap_mode='r'),source,bb
    if sourcepath.exists():source=pr.load_checkpoint(sourcepath)['source']
    else:
        groups=m['stream_id'][fit]
        source=np.mean([normalized_covariance(raw[fit[groups==g]],bb).mean(0) for g in np.unique(groups)],axis=0)
        pr.save_checkpoint(sourcepath,{'source':torch.from_numpy(source)},
                           pr.fitting_record(m,fit,excluded,stage,'source_'+name))
        source=torch.from_numpy(source)
    for role,rows in [('fit',fit),('held',held)]:
        dest=np.lib.format.open_memmap(out/f'{role}.npy',mode='w+',dtype=np.float32,shape=(len(rows),len(bb)*len(bands),41,41))
        active=None
        for j,row in enumerate(rows):
            g=m['stream_id'][row]
            if g!=active:
                stream=MultiscaleAligner(source,bb,**pr.plan()['neural_alignment']);active=g
            dest[j]=stream.transform_one(raw[row])
        dest.flush();del dest
    pr.json_write(recordpath,{'fit_rows':fit.tolist(),'held_rows':held.tolist(),
                             'raw_sha256':pr.sha(rawpath),
                             'files':{n:pr.sha(out/n) for n in ['source.pt','fit.npy','held.npy']}})
    return np.load(out/'fit.npy',mmap_mode='r'),np.load(out/'held.npy',mmap_mode='r'),source,bb


def fit_neural_stage(m,fit,held,excluded,recipe,out,seeds,device,stage='cv'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    name='expanded_grid' if recipe=='expanded_grid' else 'reconstructed_v6'
    cache=out.parent/f'align_{name}'
    ft,fh,source,bb=aligned_cache(m,fit,held,excluded,recipe,cache,stage)
    bands,views=grid(recipe)
    config=dict(n_bands=len(bands),n_views=len(views),n_chans=41,n_filters=16,hidden=128,dropout=.5)
    weights=class_cell_weights(m['y'][fit],pr.subset(m,fit))
    records=[]
    for seed,epochs in seeds:
        path=out/f'seed_{seed}.pt'
        if path.exists():data=pr.load_checkpoint(path)
        else:
            print('FIT_START',stage,excluded,recipe,seed,epochs,flush=True)
            started=time.monotonic()
            net,training_info=fit_neural(np.asarray(ft),m['y'][fit],weights,m['subject'][fit],
                                        excluded,config,recipe,seed,epochs,device)
            data=dict(config=config,state_dict=net.state_dict(),source=source,
                      broadband_indices=bb,alignment=pr.plan()['neural_alignment'])
            provenance=pr.fitting_record(m,fit,excluded,stage,recipe,seed=int(seed),fixed_epochs=int(epochs))
            pr.save_checkpoint(path,data,provenance)
            pr.json_write(out/f'seed_{seed}_training.json',training_info)
            data=pr.load_checkpoint(path)
            print('FIT_DONE',stage,excluded,recipe,seed,'seconds',round(time.monotonic()-started,1),flush=True)
        predpath=out/f'seed_{seed}_predictions.npz'
        if len(held):
            # Covariances were transformed strictly in causal session order.
            # Batched stateless evaluation is checked against actual streamed runtime later.
            net=MultiscaleNet(**config).to(device)
            net.load_state_dict(data['state_dict']);net.eval()
            probs=[]
            with torch.inference_mode():
                for lo in range(0,len(fh),64):
                    x=torch.tensor(np.asarray(fh[lo:lo+64]),device=device)
                    probs.append(torch.softmax(net(x),-1).cpu().numpy())
            np.savez_compressed(predpath,rows=held,probabilities=np.concatenate(probs))
        records.append(path)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['features','neural','classical'],required=True)
    ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    threadpool_limits(limits=2);torch.set_num_threads(2)
    pr.plan();pr.freeze_implementation()
    if args.part=='features':prepare_features();return
    m=pr.metadata();p=pr.plan()
    raw=np.load(ROOT/'cache/reconstructed_v6_cov.npy',mmap_mode='r')
    cov=raw[:,:14]
    env={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
         'device':args.device,'cuda_available':torch.cuda.is_available(),
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    pr.json_write(ROOT/f'environment_{args.part}.json',env)
    folds=[]
    for heldout in p['public_subjects']:
        fit=np.flatnonzero(m['subject']!=heldout);held=np.flatnonzero(m['subject']==heldout)
        out=ROOT/'runs'/f'hold_{heldout}'
        folds.append({'heldout':heldout,'fitting_rows':fit.tolist(),'heldout_rows':held.tolist()})
        if args.part=='neural':
            for recipe in ['reconstructed_v6','masked_ssl','expanded_grid','cross_subject_mixup']:
                seeds=[(s,e) for pair in p['seed_pairs'] for s,e in zip(pair,p['fixed_epochs'])]
                fit_neural_stage(m,fit,held,[heldout],recipe,out/recipe,seeds,args.device)
        else:
            data=fit_classical(cov,m,fit,[heldout],'covariance',out/'classical/covariance.pt')
            probs=predict_classical(data,cov[held],m['stream_id'][held])
            np.savez_compressed(out/'classical/covariance_predictions.npz',rows=held,probabilities=probs)
            choice=nested_choice(cov,m,fit,[heldout],out/'nested')
            data=fit_classical(cov,m,fit,[heldout],choice,out/'classical/nested.pt')
            probs=predict_classical(data,cov[held],m['stream_id'][held])
            np.savez_compressed(out/'classical/nested_predictions.npz',rows=held,probabilities=probs)
            print('CLASSICAL_FOLD_DONE',heldout,flush=True)
    pr.json_write(ROOT/f'{args.part}_complete.json',{'complete':True,'folds':folds,
                                                  'plan_sha256':pr.sha(ROOT/'plan.json')})


if __name__=='__main__':main()
