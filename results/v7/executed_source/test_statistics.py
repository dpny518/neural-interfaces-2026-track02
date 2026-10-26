"""Synthetic regression checks for unequal-cell participant bootstrapping."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import protocol as pr
from analyze import calculate


def run():
    plan=pr.plan();cells=[]
    for s in plan['public_subjects']:
        for session in (['1','2','3'] if s in ['P04','P10'] else ['1']):
            for interface in ['BrainHero','Graz']:cells.append((s,session,interface))
    subject=[];session=[];interface=[];labels=[]
    for i,key in enumerate(cells):
        n=94+(i<23)
        subject.extend([key[0]]*n);session.extend([key[1]]*n);interface.extend([key[2]]*n)
        labels.extend((np.arange(n)%3).tolist())
    labels=np.array(labels);subject=np.array(subject);assert len(labels)==2655
    wrong=np.eye(3)[(labels+1)%3]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'predictions').mkdir();(root/'plan.json').write_bytes((pr.ROOT/'plan.json').read_bytes())
        manifest={'complete':True,'all_passed':True,'plan_sha256':pr.sha(root/'plan.json'),'files':{}}
        for c in plan['candidates']:
            probs=np.repeat(wrong[None],3,axis=0)
            if c=='masked_ssl':probs[:,subject=='P04']=np.eye(3)[labels[subject=='P04']]
            path=root/'predictions'/f'{c}.npz'
            np.savez_compressed(path,probabilities=probs,y_true=labels,subject=subject,
                                session=session,interface=interface,rows=np.arange(2655),seed_pairs=plan['seed_pairs'])
            manifest['files'][c]=pr.sha(path)
        pr.json_write(root/'predictions/replay_manifest.json',manifest)
        result=calculate(root/'plan.json',root/'predictions',root/'analysis')
        q=result['paired']['masked_ssl']
        assert abs(q['mean_gain']-6/28)<1e-14
        assert abs(q['mean_gain']-.1)>.1,'Must not average subject means equally'
        sampled=np.random.default_rng(20261024).integers(0,10,size=(10000,10))
        # Independent formula: only P04 contributes improvement; P04/P10 each have six cells.
        expected=(sampled==3).sum(1)*6/(20+4*((sampled==3)|(sampled==9)).sum(1))
        np.testing.assert_allclose(q['paired_se'],expected.std(ddof=1),atol=1e-14)
        with np.load(root/'analysis/bootstrap_samples.npz') as z:
            np.testing.assert_array_equal(z['masked_ssl'],expected)
        assert result['selected']=='masked_ssl'
        for c in ['nested_classical','expanded_grid','cross_subject_mixup']:
            assert not result['paired'][c]['passes']
        manifest['complete']=False;pr.json_write(root/'predictions/replay_manifest.json',manifest)
        try:calculate(root/'plan.json',root/'predictions',root/'invalid')
        except AssertionError:pass
        else:raise AssertionError('An incomplete panel was accepted')
    pr.json_write(pr.ROOT/'statistics_check.json',{'passed':True,'data':'synthetic only',
                  'checks':['equal-cell rather than equal-subject gain','independent cluster bootstrap formula',
                            'zero-gain rejection','expected adoption decision','incomplete-panel rejection']})
    print('Synthetic paired-statistics checks passed.')


if __name__=='__main__':run()
