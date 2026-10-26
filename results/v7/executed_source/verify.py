"""Stage-specific provenance audit and complete actual official-harness replay."""
import argparse
import copy
import json
import sys
from pathlib import Path
import protocol as pr
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from runtime import Decoder
from run_cv import grid

ROOT=pr.ROOT


def audit():
    pr.freeze_implementation();m=pr.metadata();p=pr.plan();records=[]
    for path in sorted((ROOT/'runs').rglob('*.pt')):
        d=pr.load_checkpoint(path);r=d['provenance'];rows=np.asarray(r['fitting_rows'])
        excluded=set(r['excluded_subjects']);actual=set(m['subject'][rows])
        assert pr.array_sha(rows.astype(np.int64))==r['fitting_rows_sha256']
        assert not actual&excluded
        assert actual==set(p['public_subjects'])-excluded
        np.testing.assert_array_equal(rows,np.flatnonzero(np.isin(m['subject'],list(actual))))
        for field in ['fitted_subjects','normalization_subjects','reference_subjects',
                      'feature_scaling_subjects','checkpoint_selection_subjects']:
            assert set(r[field])==actual,(path,field)
        assert set(r['pretraining_subjects'])==(actual if r['recipe']=='masked_ssl' else set())
        if r['stage']=='cv':assert len(excluded)==1 and len(actual)==9
        elif r['stage']=='inner_cv':assert len(excluded)==2 and len(actual)==8
        else:raise AssertionError('Unexpected CV stage')
        outer=path.relative_to(ROOT/'runs').parts[0].removeprefix('hold_')
        assert outer in excluded
        if path.name.startswith('inner_'):
            inner=path.name.split('_')[1];assert inner in excluded
        if 'seed' in r:
            pairs=p['seed_pairs'];epochs=p['fixed_epochs']
            wanted={s:e for pair in pairs for s,e in zip(pair,epochs)}
            assert r['fixed_epochs']==wanted[r['seed']]
        records.append({'path':str(path.relative_to(ROOT)),'sha256':pr.sha(path),
                        'stage':r['stage'],'excluded':sorted(excluded),'fitting_subjects':sorted(actual)})
    assert len(records)==460,('Missing or extra CV artifacts',len(records))
    for outer in p['public_subjects']:
        folder=ROOT/'runs'/f'hold_{outer}'
        for recipe in ['reconstructed_v6','masked_ssl','expanded_grid','cross_subject_mixup']:
            assert len(list((folder/recipe).glob('seed_*.pt')))==6
        s=json.loads((folder/'nested/selection.json').read_text())
        assert s['outer_excluded']==[outer] and outer not in s['fitting_subjects']
        cells={k:[] for k in ['linear','rbf']}
        for inner in s['fitting_subjects']:
            held=np.flatnonzero(m['subject']==inner)
            for kind in ['linear','rbf']:
                path=folder/'nested'/f'inner_{inner}_{kind}_predictions.npz'
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['rows'],held)
                    np.testing.assert_array_equal(z['y'],m['y'][held])
                    values,_=pr.cell_scores(z['y'],z['probabilities'].argmax(1),pr.subset(m,held))
                cells[kind].extend(values)
        means={k:float(np.mean(v)) for k,v in cells.items()}
        assert means==s['inner_scores']
        assert max(['linear','rbf'],key=lambda k:means[k])==s['choice']
        assert pr.load_checkpoint(folder/'classical/nested.pt')['kind']==s['choice']
    result={'passed':True,'artifact_count':len(records),'records':records,
            'scope':'Executable exact-row/population/hash checks plus inspected fitting code; not an OS sandbox or formal proof'}
    pr.json_write(ROOT/'verification/cv_leakage_audit.json',result)
    return result


def checkpoint(candidate,folder,seeds):
    neural_recipe='reconstructed_v6' if candidate=='nested_classical' else candidate
    neural=[pr.load_checkpoint(folder/neural_recipe/f'seed_{s}.pt') for s in seeds]
    classical=[pr.load_checkpoint(folder/'classical/covariance.pt')]
    weights=[.45,.45,.1]
    if candidate=='nested_classical':
        classical.append(pr.load_checkpoint(folder/'classical/nested.pt'));weights=[.45,.45,.05,.05]
    bands,views=grid(candidate)
    info=json.loads((ROOT/'cache/dataset_info.json').read_text())
    return dict(neural=neural,classical=classical,weights=weights,bands=bands,views=views,
                ch_names=info['ch_names'],plan_sha256=pr.sha(ROOT/'plan.json'),reconstructed_v6=True)


class Sink:
    def __init__(self):self.probabilities=[];self.calls=0;self.copies=0;self.resets=0
    def __deepcopy__(self,memo):return self


class ObservedDecoder(Decoder):
    def __init__(self,ck,meta,sink):
        self.sink=sink;super().__init__(ck,meta)
    def __deepcopy__(self,memo):
        result=object.__new__(type(self));memo[id(self)]=result
        for k,v in self.__dict__.items():setattr(result,k,copy.deepcopy(v,memo))
        self.sink.copies+=1
        return result
    def reset_state(self):
        super().reset_state();self.sink.resets+=1
    def predict_proba(self,X):
        p=super().predict_proba(X)
        self.sink.probabilities.append(p.copy());return p
    def predict(self,X):
        assert len(X)==1
        self.sink.calls+=1
        return super().predict(X)


def replay_one(ck,rows,m,windows):
    sys.path.insert(0,str(ROOT/'official'))
    from benchmark_utils.streaming import predict_streams
    sink=Sink();model=ObservedDecoder(ck,{'device':'cpu','n_classes':3},sink)
    sink.resets=0
    def loader():
        for start in range(0,len(rows),32):
            r=rows[start:start+32]
            info={k:m[k][r] for k in ['stream_id','record_id','onset','subject','session','interface']}
            yield torch.tensor(np.array(windows[r],copy=True)),torch.tensor(m['y'][r]),info
    yt,yp,info=predict_streams(model,loader())
    probabilities=np.concatenate(sink.probabilities)
    np.testing.assert_array_equal(yt,m['y'][rows]);np.testing.assert_array_equal(yp,probabilities.argmax(1))
    assert sink.calls==len(rows)
    assert sink.copies==sink.resets==len(np.unique(m['stream_id'][rows]))
    return probabilities,{'calls':sink.calls,'copies':sink.copies,'resets':sink.resets}


def replay():
    threadpool_limits(limits=2);torch.set_num_threads(2)
    m=pr.metadata();p=pr.plan()
    for part in ['neural','classical']:
        assert json.loads((ROOT/f'{part}_complete.json').read_text())['complete']
    a=audit()
    official=json.loads((ROOT/'official/manifest.json').read_text())
    for name,h in official.items():assert pr.sha(ROOT/'official'/name)==h
    sys.path.insert(0,str(ROOT/'official'))
    from benchmark_utils.metrics import balanced_accuracy,group_scores
    windows=np.load(ROOT/'cache/windows.npy',mmap_mode='r')
    replay_inputs={'windows_sha256':pr.sha(ROOT/'cache/windows.npy'),
                   'metadata_sha256':pr.sha(ROOT/'cache/metadata.npz'),
                   'implementation_freeze_sha256':pr.sha(ROOT/'implementation_freeze.json'),
                   'replay_source_sha256':pr.sha(Path(__file__))}
    output=ROOT/'predictions';output.mkdir(exist_ok=True)
    totals={'calls':0,'copies':0,'resets':0};files={};checks=[]
    for candidate in p['candidates']:
        probabilities=np.full((3,len(m['y']),3),np.nan,dtype=np.float64)
        for rep,seeds in enumerate(p['seed_pairs']):
            for subject in p['public_subjects']:
                rows=np.flatnonzero(m['subject']==subject);folder=ROOT/'runs'/f'hold_{subject}'
                ck=checkpoint(candidate,folder,seeds)
                # Cache complete individual replay jobs, bound to all fitted inputs.
                job=output/'jobs'/f'{candidate}_{rep}_{subject}.npz';record=job.with_suffix('.json')
                paths=[folder/('reconstructed_v6' if candidate=='nested_classical' else candidate)/f'seed_{s}.pt' for s in seeds]
                paths+=[folder/'classical/covariance.pt']
                if candidate=='nested_classical':paths+=[folder/'classical/nested.pt']
                bindings={str(x.relative_to(ROOT)):pr.sha(x) for x in paths}
                bindings.update(replay_inputs)
                if record.exists():
                    r=json.loads(record.read_text());assert r['bindings']==bindings and r['sha256']==pr.sha(job)
                    with np.load(job,allow_pickle=False) as z:probs=z['probabilities'];np.testing.assert_array_equal(z['rows'],rows)
                    counts=r['counts']
                else:
                    probs,counts=replay_one(ck,rows,m,windows)
                    job.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(job,rows=rows,probabilities=probs)
                    pr.json_write(record,{'bindings':bindings,'sha256':pr.sha(job),'counts':counts})
                for k in totals:totals[k]+=counts[k]
                # Compare the actual runtime with the independently assembled fitted-cache path.
                nr='reconstructed_v6' if candidate=='nested_classical' else candidate
                components=[]
                for seed in seeds:
                    with np.load(folder/nr/f'seed_{seed}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['rows'],rows);components.append(z['probabilities'])
                with np.load(folder/'classical/covariance_predictions.npz') as z:components.append(z['probabilities'])
                if candidate=='nested_classical':
                    with np.load(folder/'classical/nested_predictions.npz') as z:components.append(z['probabilities'])
                assembled=sum(w*v for w,v in zip(ck['weights'],components))
                np.testing.assert_allclose(probs,assembled,atol=2e-6,rtol=1e-5)
                mismatch=int(np.sum(probs.argmax(1)!=assembled.argmax(1)))
                assert mismatch==0,(candidate,rep,subject,mismatch)
                probabilities[rep,rows]=probs
                print('REPLAY_DONE',candidate,rep,subject,flush=True)
        assert np.isfinite(probabilities).all()
        groups=np.stack([m[k] for k in ['subject','session','interface']],axis=1)
        for rep in range(3):
            official_score=float(group_scores(balanced_accuracy,m['y'],probabilities[rep].argmax(1),groups).mean())
            own,_=pr.cell_scores(m['y'],probabilities[rep].argmax(1),m)
            assert abs(official_score-own.mean())<1e-12
            checks.append({'candidate':candidate,'replicate':rep,'score':official_score})
        path=output/f'{candidate}.npz'
        np.savez_compressed(path,probabilities=probabilities,y_true=m['y'],subject=m['subject'],
                            session=m['session'],interface=m['interface'],rows=np.arange(2655),seed_pairs=p['seed_pairs'])
        files[candidate]=pr.sha(path)
    assert totals=={'calls':39825,'copies':210,'resets':210}
    result={'complete':True,'all_passed':True,'files':files,'plan_sha256':pr.sha(ROOT/'plan.json'),
            'counts':totals,'class_label_mismatches':0,'official_metric_checks':checks,
            'leakage_audit_sha256':pr.sha(ROOT/'verification/cv_leakage_audit.json'),
            'evaluator_sources':official}
    pr.json_write(output/'replay_manifest.json',result)
    pr.json_write(ROOT/'verification/official_replay.json',result)


if __name__=='__main__':replay()
