"""Recompute the complete paired decision from immutable replay predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def calculate(plan_path,predictions_dir,output):
    plan_path=Path(plan_path);predictions_dir=Path(predictions_dir);output=Path(output)
    assert not output.exists(),'Choose a new output directory'
    plan=json.loads(plan_path.read_text())
    manifest=json.loads((predictions_dir/'replay_manifest.json').read_text())
    assert manifest['complete'] and manifest['all_passed']
    assert manifest['plan_sha256']==sha(plan_path)
    assert set(manifest['files'])==set(plan['candidates'])
    summaries={};scores={};data={};keys=None
    for candidate in plan['candidates']:
        p=predictions_dir/f'{candidate}.npz'
        assert sha(p)==manifest['files'][candidate]
        with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        assert d['probabilities'].shape==(3,2655,3)
        assert np.isfinite(d['probabilities']).all()
        np.testing.assert_allclose(d['probabilities'].sum(-1),1.,atol=1e-6)
        assert (d['probabilities']>=0).all()
        np.testing.assert_array_equal(d['rows'],np.arange(2655))
        np.testing.assert_array_equal(d['seed_pairs'],plan['seed_pairs'])
        assert set(d['subject'])==set(plan['public_subjects'])
        if data:
            old=data[plan['baseline']]
            for k in ['rows','subject','session','interface','y_true']:
                np.testing.assert_array_equal(d[k],old[k])
        data[candidate]=d
        ck=sorted(set(zip(d['subject'],d['session'],d['interface'])))
        assert len(ck)==28
        if keys is not None:assert ck==keys
        keys=ck;matrix=np.empty((3,len(keys)));cells=[]
        pred=d['probabilities'].argmax(-1)
        for j,key in enumerate(keys):
            ix=np.flatnonzero((d['subject']==key[0])&(d['session']==key[1])&(d['interface']==key[2]))
            counts=[int(np.sum(d['y_true'][ix]==c)) for c in range(3)];assert all(counts)
            for r in range(3):
                matrix[r,j]=np.mean([np.mean(pred[r,ix][d['y_true'][ix]==c]==c) for c in range(3)])
            cells.append({'subject':str(key[0]),'session':str(key[1]),'interface':str(key[2]),
                          'windows':len(ix),'class_counts':counts,'replicates':matrix[:,j].tolist(),
                          'mean':float(matrix[:,j].mean())})
        replicate=matrix.mean(1);scores[candidate]=matrix
        summaries[candidate]={'mean':float(replicate.mean()),'sample_sd':float(replicate.std(ddof=1)),
                              'replicates':replicate.tolist(),'cells':cells}
    paired={};bootstraps={};baseline=scores[plan['baseline']]
    for candidate in plan['candidates'][1:]:
        diff=scores[candidate]-baseline;delta=diff.mean(0)
        sums=[];counts=[];subjects=[]
        for s in plan['public_subjects']:
            ix=[i for i,k in enumerate(keys) if k[0]==s]
            sums.append(delta[ix].sum());counts.append(len(ix))
            subjects.append({'subject':s,'n_cells':len(ix),'gain':float(delta[ix].mean())})
        rng=np.random.default_rng(plan['bootstrap']['seed'])
        drawn=rng.integers(0,10,size=(plan['bootstrap']['draws'],10))
        boot=np.asarray(sums)[drawn].sum(1)/np.asarray(counts)[drawn].sum(1)
        gain=float(delta.mean());se=float(boot.std(ddof=1));bootstraps[candidate]=boot
        paired[candidate]={'mean_gain':gain,'paired_se':se,'margin':gain-se,'passes':gain>se,
                           'replicate_gains':diff.mean(1).tolist(),
                           'percentile_interval_95':np.percentile(boot,[2.5,97.5]).tolist(),
                           'subjects':subjects}
    passing=[c for c in paired if paired[c]['passes']]
    winner=sorted(passing,key=lambda c:(-paired[c]['margin'],c))[0] if passing else None
    result={'schema':'v7-reconstruction-selection-v1','baseline':plan['baseline'],
            'original_v6_recovered':False,'plan_sha256':sha(plan_path),
            'replay_manifest_sha256':sha(predictions_dir/'replay_manifest.json'),
            'summaries':summaries,'paired':paired,'selected':winner,'null_result':winner is None,
            'contamination':'All ten public participants are adaptively reused development data; P04/P09 have no independent test role.'}
    output.mkdir(parents=True)
    (output/'selection.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(output/'bootstrap_samples.npz',**bootstraps)
    np.savez_compressed(output/'cell_scores.npz',**scores)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();r=calculate(a.plan,a.predictions,a.out)
    print(json.dumps({'selected':r['selected'],'summaries':{k:{q:v[q] for q in ['mean','sample_sd']} for k,v in r['summaries'].items()},'paired':r['paired']},indent=2))
