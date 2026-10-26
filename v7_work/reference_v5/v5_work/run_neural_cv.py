"""Fixed v5 neural CV: every fitted tensor is isolated to nine participants.

The runner writes probabilities, never selects models or scores a candidate.
Six independently initialized models are fitted per held-out participant:
three predetermined two-member ensemble replicates at fixed epochs 75/20.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'bci_work'),str(ROOT/'v5_work')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from neural_cov import SpatialLogVarianceNet
from temporal_sweep import cell_class_weights
from v5_work.neural_training_v5 import train_model,recipe_definitions
from v5_work.protocol_v5 import (PUBLIC_SUBJECTS,assert_stage_subjects,make_provenance,
    register_training_artifact,check_training_artifacts,sha256_file,write_json_exclusive)


def subset(meta,rows):return {k:v[rows] for k,v in meta.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recipe',required=True,choices=['baseline','augmentation','adversarial','masked_ssl'])
    ap.add_argument('--plan',type=Path,default=ROOT/'v5_work/plan.json')
    ap.add_argument('--cache',type=Path,default=ROOT/'v5_work/cache')
    ap.add_argument('--fold-cache',type=Path,default=ROOT/'v5_work/fold_cache')
    ap.add_argument('--out',type=Path,default=ROOT/'v5_work/neural_results')
    ap.add_argument('--subjects',nargs='+',default=list(PUBLIC_SUBJECTS))
    ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
    torch.set_num_threads(a.threads);threadpool_limits(a.threads)
    plan=json.loads(a.plan.read_text());plan_hash=sha256_file(a.plan)
    assert plan['warmup_test_role_retired'] is True
    assert plan['neural_definitions']==recipe_definitions()
    assert plan['seed_pairs']==[[33,42],[133,142],[233,242]] and plan['fixed_epochs']==[75,20]
    assert len(set(a.subjects))==len(a.subjects) and set(a.subjects)<=set(PUBLIC_SUBJECTS)
    with np.load(a.cache/'metadata.npz') as z:meta={k:z[k] for k in z.files}
    metadata_hash=sha256_file(a.cache/'metadata.npz')
    assert_stage_subjects(meta['subject'],stage='selection')
    assert len(meta['y'])==2655 and set(meta['split'])=={'development'}
    out=a.out/a.recipe;out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();completed=[];all_manifests=[]
    for heldout in a.subjects:
        folder=a.fold_cache/heldout
        checks=json.loads((folder/'cache_checksums.json').read_text())
        assert checks['plan_sha256']==plan_hash
        assert checks['metadata_sha256']==metadata_hash and checks['heldout_subject']==heldout
        for name,digest in checks['files'].items():assert sha256_file(folder/name)==digest,(heldout,name)
        check_training_artifacts([folder/'reference.provenance.json'])
        with np.load(folder/'reference.npz') as z:
            fit=z['fit_rows'];held=z['held_rows'];source=z['source']
        assert np.array_equal(fit,np.flatnonzero(meta['subject']!=heldout))
        assert np.array_equal(held,np.flatnonzero(meta['subject']==heldout))
        fm,hm=subset(meta,fit),subset(meta,held)
        fit_subjects=assert_stage_subjects(fm['subject'],stage='cv_fit',heldout_subject=heldout)
        assert_stage_subjects(hm['subject'],stage='cv_evaluate',heldout_subject=heldout)
        train_cov=np.load(folder/'fit_covariances.npy',mmap_mode='r')
        held_cov=np.load(folder/'held_covariances.npy',mmap_mode='r')
        assert train_cov.shape==(len(fit),14,41,41) and held_cov.shape==(len(held),14,41,41)
        loss_weights=cell_class_weights(fm)
        probabilities=np.empty((3,2,len(held),3),np.float32)
        foldout=out/('hold_'+heldout);foldout.mkdir(exist_ok=True)
        prediction_path=foldout/'predictions.npz';isolation_path=foldout/'isolation_check.json'
        expected_manifests=[foldout/f'seed{seed}_epoch{epoch}.provenance.json'
                            for seeds in plan['seed_pairs'] for seed,epoch in zip(seeds,plan['fixed_epochs'])]
        if prediction_path.exists() and isolation_path.exists():
            saved=json.loads(isolation_path.read_text())
            assert saved['prediction_sha256']==sha256_file(prediction_path)
            audit=check_training_artifacts(expected_manifests)
            for record in audit['artifacts']:
                assert record['heldout_subject']==heldout and record['fitted_subjects']==fit_subjects
                assert record['config']['plan_sha256']==plan_hash and record['config']['recipe']==a.recipe
                assert record['config']['reference_sha256']==checks['files']['reference.npz']
            with np.load(prediction_path) as previous:
                assert np.array_equal(previous['row_indices'],held)
                assert previous['branch_probabilities'].shape==(3,2,len(held),3)
            completed.append(heldout);all_manifests.extend(map(str,expected_manifests))
            print(json.dumps(dict(recipe=a.recipe,heldout_subject=heldout,status='verified_complete_fold_reused')),flush=True)
            del train_cov,held_cov
            continue
        if isolation_path.exists():raise RuntimeError('Completed-fold audit exists without its predictions')
        fold_manifests=[]
        for replicate,seeds in enumerate(plan['seed_pairs']):
            for branch,(seed,epochs) in enumerate(zip(seeds,plan['fixed_epochs'])):
                path=foldout/f'seed{seed}_epoch{epochs}.pt';manifest=path.with_suffix('.provenance.json')
                config=dict(recipe=a.recipe,model_config=plan['neural_definitions']['model'],
                            neural_definitions=plan['neural_definitions'],alignment=plan['neural_alignment'],
                            plan_sha256=plan_hash,reference_sha256=checks['files']['reference.npz'],
                            fixed_epochs=epochs,cosine_horizon=plan['schedule_epochs'])
                if path.exists() or manifest.exists():
                    audit=check_training_artifacts([manifest]);prov=audit['artifacts'][0]
                    assert prov['config']==config and prov['seed']==seed and prov['epoch']==epochs
                    assert prov['heldout_subject']==heldout and prov['fitted_subjects']==fit_subjects
                    checkpoint=torch.load(path,weights_only=True,map_location='cpu')
                    assert checkpoint['heldout_subject']==heldout and checkpoint['training_subjects']==fit_subjects
                    assert np.array_equal(checkpoint['source'].numpy(),source)
                    net=SpatialLogVarianceNet(**checkpoint['model_config']);net.load_state_dict(checkpoint['state_dict']);net.eval()
                else:
                    # Held-out EEG/labels are never supplied to this call.
                    net,history,training_info=train_model(train_cov,fm['y'],loss_weights,fm['subject'],
                        plan['neural_definitions']['model'],seed,epochs,recipe=a.recipe,
                        heldout_subject=heldout,schedule_epochs=plan['schedule_epochs'])
                    prov=make_provenance(stage='cv_fit',fitted_subjects=fm['subject'],heldout_subject=heldout,
                        pretraining_subjects=fm['subject'] if a.recipe=='masked_ssl' else [],
                        normalization_subjects=fm['subject'],reference_subjects=fm['subject'],
                        feature_scale_subjects=fm['subject'] if a.recipe=='masked_ssl' else [],
                        batchnorm_subjects=fm['subject'],checkpoint_selection_subjects=[],
                        seed=seed,epoch=epochs,config=config,row_indices=fit)
                    checkpoint=dict(model_type='spatial_logvariance',model_config=plan['neural_definitions']['model'],
                        state_dict=net.state_dict(),source=torch.from_numpy(source.copy()),
                        alignment_config=plan['neural_alignment'],training_subjects=fit_subjects,
                        heldout_subject=str(heldout),seed=int(seed),epoch=int(epochs),
                        scheduler_T_max=plan['schedule_epochs'],recipe=a.recipe,provenance=prov)
                    torch.save(checkpoint,path);register_training_artifact(path,prov,manifest)
                    write_json_exclusive(path.with_suffix('.training.json'),dict(history=history,training_info=training_info))
                with torch.inference_mode():
                    pred=[]
                    for start in range(0,len(held_cov),64):
                        batch=torch.from_numpy(np.array(held_cov[start:start+64],copy=True))
                        pred.append(net(batch).softmax(-1).numpy())
                    probabilities[replicate,branch]=np.concatenate(pred)
                fold_manifests.append(str(manifest));del net,checkpoint
                print(json.dumps(dict(recipe=a.recipe,heldout_subject=heldout,replicate=replicate,
                    seed=seed,epochs=epochs,seconds=round(time.perf_counter()-started,1))),flush=True)
        np.savez_compressed(prediction_path,branch_probabilities=probabilities,
                            row_indices=held,seed_pairs=np.array(plan['seed_pairs']),subject=hm['subject'])
        audit=check_training_artifacts(fold_manifests)
        audit['prediction_sha256']=sha256_file(prediction_path)
        write_json_exclusive(isolation_path,audit)
        all_manifests.extend(fold_manifests);completed.append(heldout)
        del train_cov,held_cov
    summary=dict(recipe=a.recipe,subjects_completed=completed,plan_sha256=plan_hash,
                 n_models=len(all_manifests),elapsed_seconds=time.perf_counter()-started,
                 selection_performed=False,warmup_test_role_retired=True,manifests=all_manifests)
    name='summary.json' if set(completed)==set(PUBLIC_SUBJECTS) else 'summary_'+'_'.join(completed)+'.json'
    write_json_exclusive(out/name,summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
