"""Conditional post-selection fits, contract checks and auditable deliverables."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
import protocol as pr
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from analyze import calculate
from run_cv import fit_neural_stage
from classical import fit_classical,nested_choice
from verify import checkpoint,replay_one

ROOT=pr.ROOT


def standalone_source():
    sources=[ROOT/'reference_v5/methods_draft/riemann_features.py',
             ROOT/'reference_v5/bci_work/neural_cov.py',
             ROOT/'reference_v5/v5_work/riemann_v5_runtime.py',ROOT/'runtime.py']
    result=['# Reconstructed v6 baseline / fixed v7 experiment.\n# NumPy/PyTorch-only inference.\n']
    for path in sources:
        raw=path.read_text();lines=raw.splitlines(keepends=True);removed=set()
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node,ast.ImportFrom) and node.module in ['__future__','riemann_features','neural_cov','riemann_v5_runtime']:
                removed.update(range(node.lineno-1,node.end_lineno))
        result.append(''.join(line for i,line in enumerate(lines) if i not in removed))
    text='\n'.join(result)
    tree=ast.parse(text)
    allowed={'numpy','torch','pathlib','benchmark_utils'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(x.name.split('.')[0] in allowed for x in node.names)
        elif isinstance(node,ast.ImportFrom):
            assert node.module.split('.')[0] in allowed
    return text


def stage_audit(stage,m,fit):
    folder=ROOT/'postselection'/stage;expected=set(m['subject'][fit]);records=[]
    for path in sorted(folder.rglob('*.pt')):
        d=pr.load_checkpoint(path);r=d['provenance'];rows=np.asarray(r['fitting_rows'],dtype=np.int64)
        actual=set(m['subject'][rows]);excluded=set(r['excluded_subjects'])
        assert actual==set(pr.plan()['public_subjects'])-excluded
        assert actual<=expected
        if r['stage']==stage:assert actual==expected
        else:assert r['stage']=='inner_'+stage and len(actual)==len(expected)-1
        assert not actual&excluded
        np.testing.assert_array_equal(rows,np.flatnonzero(np.isin(m['subject'],list(actual))))
        assert pr.array_sha(rows)==r['fitting_rows_sha256']
        for field in ['fitted_subjects','normalization_subjects','reference_subjects',
                      'feature_scaling_subjects','checkpoint_selection_subjects']:
            assert set(r[field])==actual
        assert set(r['pretraining_subjects'])==(actual if r['recipe']=='masked_ssl' else set())
        records.append({'path':str(path.relative_to(ROOT)),'sha256':pr.sha(path),
                        'fitting_subjects':sorted(actual),'excluded_subjects':sorted(excluded)})
    assert len(records)>=4
    return records


def package(stage,candidate,folder,selection_hash):
    ck=checkpoint(candidate,folder,pr.plan()['final_seed_pair'])
    ck.update(stage=stage,selection_sha256=selection_hash)
    out=ROOT/'packages'/stage;out.mkdir(parents=True,exist_ok=True)
    source=standalone_source()
    if (out/'weights.pt').exists():
        assert (out/'submission.py').read_text()==source
        old=torch.load(out/'weights.pt',map_location='cpu',weights_only=True)
        assert old['selection_sha256']==selection_hash
    else:
        (out/'submission.py').write_text(source)
        torch.save(ck,out/'weights.pt')
    name='submission.zip' if stage=='final' else 'legacy_submission.zip'
    target=ROOT/'deliverables'/name;target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n in ['submission.py','weights.pt']:z.write(out/n,n)
    with zipfile.ZipFile(target) as z:
        assert z.namelist()==['submission.py','weights.pt'] and z.testzip() is None
        for n in z.namelist():assert z.read(n)==(out/n).read_bytes()
    return ck,out,target


def contract(stage,out,zip_path,m,fit,records):
    sys.path.insert(0,str(ROOT/'official'))
    import contract_check
    windows=np.load(ROOT/'cache/windows.npy',mmap_mode='r')
    info=json.loads((ROOT/'cache/dataset_info.json').read_text())
    selected=fit[:12]
    old=contract_check.load_cache
    contract_check.load_cache=lambda *args,**kwargs:(np.array(windows[selected],copy=True),
                                                   pr.subset(m,selected),info)
    report_path=ROOT/'verification'/f'{stage}_contract_check.json'
    try:r=contract_check.check_submission(out,report_path,zip_file=zip_path,kit_dir=ROOT/'official')
    finally:contract_check.load_cache=old
    assert len(r['checks'])==16 and r['all_passed']
    module=contract_check._import_submission(out/'submission.py','_v7_required_'+stage)
    minimal=module.Solver().load_model({'device':'cpu','n_classes':3,'submission_dir':out})
    for n in [0,1,4]:
        pred=copy.deepcopy(minimal).predict(torch.tensor(windows[selected[:n]].copy()))
        assert pred.shape==(n,) and pred.dtype==np.int64
    packaged=torch.load(out/'weights.pt',map_location='cpu',weights_only=True)
    expected=set(m['subject'][fit])
    registered={x['sha256'] for x in records}
    for member in packaged['neural']+packaged['classical']:
        prov=member['provenance'];assert set(prov['fitted_subjects'])==expected
        assert prov['stage']==stage and not set(prov['excluded_subjects'])&expected
    r['checks'].append({'name':'stage_specific_no_heldout_leakage','passed':True,
                        'cv_audit_sha256':pr.sha(ROOT/'verification/cv_leakage_audit.json'),
                        'stage':stage,'stage_artifacts':records,
                        'packaged_member_populations_verified':True})
    r.update(n_checks=17,all_passed=True,stage=stage,runtime_check_fitting_rows=selected.tolist(),
             required_only_metadata_check=True,zip_sha256=pr.sha(zip_path),
             data_used='actual fitting rows and synthetic signals; no diagnostic labels')
    pr.json_write(report_path,r)


def postselection(selection):
    candidate=selection['selected'];assert candidate is not None
    selection_hash=pr.sha(ROOT/'analysis/selection.json')
    freeze=ROOT/'selection_freeze.json'
    binding={'selection_sha256':selection_hash,'plan_sha256':pr.sha(ROOT/'plan.json'),
             'replay_sha256':pr.sha(ROOT/'predictions/replay_manifest.json'),
             'metadata_sha256':pr.sha(ROOT/'cache/metadata.npz'),
             'windows_sha256':pr.sha(ROOT/'cache/windows.npy'),
             'postselection_source_sha256':pr.sha(Path(__file__))}
    if freeze.exists():assert json.loads(freeze.read_text())==binding
    else:pr.json_write(freeze,binding)
    m=pr.metadata();p=pr.plan();cov=np.load(ROOT/'cache/reconstructed_v6_cov.npy',mmap_mode='r')[:,:14]
    models={};packages={}
    for stage,subjects in [('final',p['public_subjects']),('legacy',p['legacy_diagnostic']['training_subjects'])]:
        fit=np.flatnonzero(np.isin(m['subject'],subjects));excluded=sorted(set(p['public_subjects'])-set(subjects))
        folder=ROOT/'postselection'/stage
        recipe='reconstructed_v6' if candidate=='nested_classical' else candidate
        # One fitting-row prediction only; no held-out metric or checkpoint selection.
        fit_neural_stage(m,fit,fit[:1],excluded,recipe,folder/recipe,
                         list(zip(p['final_seed_pair'],p['fixed_epochs'])),'cuda:0',stage=stage)
        fit_classical(cov,m,fit,excluded,'covariance',folder/'classical/covariance.pt',stage=stage)
        if candidate=='nested_classical':
            kind=nested_choice(cov,m,fit,excluded,folder/'nested',stage=stage)
            fit_classical(cov,m,fit,excluded,kind,folder/'classical/nested.pt',stage=stage)
        records=stage_audit(stage,m,fit)
        ck,out,target=package(stage,candidate,folder,selection_hash)
        contract(stage,out,target,m,fit,records)
        models[stage]=ck;packages[stage]=pr.sha(target)
    # Both inference ZIPs and the selected decision are frozen before the one diagnostic pass.
    diagnostic=ROOT/'legacy_diagnostic';diagnostic.mkdir(exist_ok=True)
    receipt=diagnostic/'started.json';result_path=diagnostic/'result.json'
    if result_path.exists():
        r=json.loads(result_path.read_text());assert r['package_hashes']==packages
        return
    with receipt.open('x') as f:
        json.dump({'created_at_utc':datetime.now(timezone.utc).isoformat(),
                   'selection_sha256':selection_hash,'package_hashes':packages,
                   'one_pass':True},f,indent=2)
    rows=np.flatnonzero(np.isin(m['subject'],p['legacy_diagnostic']['evaluation_subjects']))
    assert len(rows)==755
    windows=np.load(ROOT/'cache/windows.npy',mmap_mode='r')
    probs,counts=replay_one(models['legacy'],rows,m,windows)
    scores,keys=pr.cell_scores(m['y'][rows],probs.argmax(1),pr.subset(m,rows))
    assert len(keys)==8 and counts=={'calls':755,'copies':4,'resets':4}
    np.savez_compressed(diagnostic/'predictions.npz',rows=rows,probabilities=probs,y_true=m['y'][rows])
    pr.json_write(result_path,{'score':float(scores.mean()),'cells':scores.tolist(),'keys':keys,
                              'counts':counts,'package_hashes':packages,'selection_sha256':selection_hash,
                              'selection_contaminated':True,'used_for_selection':False,'passes':1})
    for stage,name in [('final','submission.zip'),('legacy','legacy_submission.zip')]:
        assert pr.sha(ROOT/'deliverables'/name)==packages[stage]


def isolated_reanalysis():
    with tempfile.TemporaryDirectory(prefix='ni26-v7-isolated-') as d:
        temp=Path(d);pred=temp/'predictions';pred.mkdir()
        shutil.copy2(ROOT/'analyze.py',temp/'analyze.py');shutil.copy2(ROOT/'plan.json',temp/'plan.json')
        for p in (ROOT/'predictions').glob('*.npz'):shutil.copy2(p,pred/p.name)
        shutil.copy2(ROOT/'predictions/replay_manifest.json',pred/'replay_manifest.json')
        wrapper='''import os,sys,runpy
blocked=os.path.realpath(sys.argv[1]);script=sys.argv[2]
def audit(event,args):
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.path.realpath(os.fsdecode(args[0]))
        if p==blocked or p.startswith(blocked+os.sep):raise PermissionError("Original workspace access denied")
sys.addaudithook(audit)
sys.argv=[script,"--plan","plan.json","--predictions","predictions","--out","analysis"]
runpy.run_path(script,run_name="__main__")
'''
        proc=subprocess.run([sys.executable,'-c',wrapper,str(ROOT),str(temp/'analyze.py')],
                            cwd=temp,capture_output=True,text=True)
        assert proc.returncode==0,proc.stderr
        original=json.loads((ROOT/'analysis/selection.json').read_text())
        recreated=json.loads((temp/'analysis/selection.json').read_text())
        assert recreated==original
        for name in ['bootstrap_samples.npz','cell_scores.npz']:
            with np.load(ROOT/'analysis'/name) as a,np.load(temp/'analysis'/name) as b:
                assert a.files==b.files
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    pr.json_write(ROOT/'verification/isolated_reanalysis.json',{
        'passed':True,'original_workspace_open_access_denied':True,'exact_selection_match':True,
        'exact_cell_and_bootstrap_array_matches':True,'analysis_source_sha256':pr.sha(ROOT/'analyze.py')})


def write_report(selection):
    p=pr.plan();replay=json.loads((ROOT/'predictions/replay_manifest.json').read_text())
    audit=json.loads((ROOT/'verification/cv_leakage_audit.json').read_text())
    lines=['NEURAL INTERFACES 2026 TRACK 2 — V7 / RECONSTRUCTED V6 BASELINE',
           '', 'ORIGINAL V6 ARTIFACTS WERE NOT RECOVERED.',
           'The historical 57.91% +/- 0.58 claim and ce6c36d9... deployment hash were supplied by the user and were not independently verified.',
           'This round measures a newly reconstructed multiscale baseline. It does not establish numerical identity with the lost model.',
           '', 'PROTOCOL',
           'All P01–P10 are adaptively reused public development data under Option A. P04/P09 have no independent test role.',
           'Ten LOSO folds, three fixed seed pairs (33,42), (133,142), (233,242). Fixed branch epochs 75/20, cosine horizon 120; no held-out checkpoint selection.',
           'Each held-out participant is excluded from all fitting, including unlabeled SSL, normalization, reference covariances, feature scales and BatchNorm statistics.',
           'The nested classical replacement uses inner LOSO only within the outer fitting population; every inner model excludes both participants.',
           'Metric: balanced accuracy within each subject/session/interface cell, averaged equally across all 28 cells. SD is sample SD across three complete CV replicates.',
           'Adopt only if the mean paired gain exceeds its participant-cluster bootstrap SE (10,000 draws, seed 20261024). Replicate-paired cell differences are averaged before resampling.',
           'Each participant cluster retains all of its sessions/interfaces. Resamples preserve equal-cell weighting with a ratio of sampled cell sums to sampled cell counts.',
           'Among passes, maximize gain minus SE; exact ties break lexicographically. This is a development rule, not a statistical-significance claim.',
           'Ten participant clusters, overlapping CV fitting populations, adaptive historical reuse and candidate selection all limit external validity.',
           '', 'MEASURED COMPLETE PANEL',
           'Candidate | CV mean +/- SD (%) | Gain (pp) | Paired SE (pp) | Decision']
    for c in p['candidates']:
        s=selection['summaries'][c];q=selection['paired'].get(c)
        line=f"{c} | {100*s['mean']:.4f} +/- {100*s['sample_sd']:.4f}"
        if q:line+=f" | {100*q['mean_gain']:+.4f} | {100*q['paired_se']:.4f} | {'PASS' if q['passes'] else 'REJECT'}"
        else:line+=' | — | — | reference'
        lines.append(line)
    lines+=['','SELECTION',f"Selected: {selection['selected']}" if selection['selected'] else
            'NULL RESULT: no candidate cleared the predeclared bar. No final/legacy candidate fits or new submission ZIPs were made. The historical v6 remains the stated champion, but its original deployable artifact is still missing.',
            '', 'RECONSTRUCTION AND FIXED CANDIDATES',
            'Baseline: 7 spectral bands x early/late/full views; shared 7x16x41 normalized spatial weights; log projected variances plus channel log powers; BatchNorm(momentum=.05), dropout .5; linear logits plus a GELU 128-unit residual head.',
            'Both head branches use normal PyTorch initialization. This exact topology/initialization is an inference fixed before fitting, not recovered v6 source.',
            'Mixture: 45% each neural branch + 10% original temporal Euclidean log-covariance ridge. Neural prior count 16, classical prior 8, EMA floor .02, shrinkage .05, trace clip 10; transform before causal update.',
            'Masked SSL: 30 fitting-only masked log-power reconstruction epochs; fitting-only target scaling; only shared spatial weights retained; same-seed fresh BN and supervised heads.',
            'Nested classical: the lost v6 classical recipe is replaced explicitly. Keep neural weights .45/.45; split the remaining .10 equally between original covariance ridge and an inner-selected log-power linear/RBF ridge member.',
            'Expanded grid: add 8–10,10–13,13–20,20–30 Hz bands and middle two-second view to the baseline grid, for 44 banks. Original classical branch unchanged.',
            'Cross-subject mixup: fitting-minibatch partners from other participants, beta(.2,.2), probability .5, weighted soft targets. No additional masking.',
            'All exact hyperparameters, seeds, grid definitions and inferred details are recorded in plan.json.',
            '', 'ALL PER-CELL RESULTS (mean over three paired replicates; percent)',
            'Subject | Session | Interface | N | Class counts | '+' | '.join(p['candidates'])]
    for i,cell in enumerate(selection['summaries'][p['baseline']]['cells']):
        vals=[f"{100*selection['summaries'][c]['cells'][i]['mean']:.4f}" for c in p['candidates']]
        lines.append(f"{cell['subject']} | {cell['session']} | {cell['interface']} | {cell['windows']} | {cell['class_counts']} | "+' | '.join(vals))
    lines+=['','PER-PARTICIPANT PAIRED GAINS (pp; diagnostic, not equal-subject headline weighting)']
    for c,q in selection['paired'].items():
        lines.append(c+': '+', '.join(f"{s['subject']} {100*s['gain']:+.4f}" for s in q['subjects']))
        lines.append('95% descriptive bootstrap percentile interval (pp): '+str([100*x for x in q['percentile_interval_95']]))
    lines+=['','EXACT FOLDS']
    for s in p['public_subjects']:lines.append('Hold '+s+'; fit '+','.join(x for x in p['public_subjects'] if x!=s))
    lines+=['','VERIFICATION',
            f"Official batch-one replay: {replay['counts']}; zero class-label mismatches; all 15 CV scores match official group_scores(balanced_accuracy).",
            f"Stage-specific audit: {audit['artifact_count']} CV checkpoints/reference artifacts, exact fitting rows, excluded populations, hashes and embedded provenance verified.",
            'The isolated prediction-only reanalysis blocked file access to the original workspace and reproduced every summary, paired statistic, bootstrap array and decision exactly.']
    if selection['selected']:
        for stage,name in [('final','submission.zip'),('legacy','legacy_submission.zip')]:
            r=json.loads((ROOT/'verification'/f'{stage}_contract_check.json').read_text())
            assert r['n_checks']==17 and r['all_passed']
            lines.append(f"{stage}: all 17 contract checks passed; required-only-metadata load/predict also passed. {name} SHA-256: {pr.sha(ROOT/'deliverables'/name)}")
        d=json.loads((ROOT/'legacy_diagnostic/result.json').read_text())
        lines+=['','DEPLOYMENT AND SINGLE LEGACY DIAGNOSTIC',
                'Deployment fit: all P01–P10, seeds33/42, fixed75/20epochs; no new held-out role.',
                'Legacy fit: exactly P01,P02,P03,P06,P07,P10; one P04/P09 pass only after selection and both ZIPs were frozen.',
                f"Legacy diagnostic: {100*d['score']:.4f}% over 755 windows, four sessions and eight cells.",
                'This diagnostic is selection-contaminated and is neither an independent test nor a score of the all-ten deployment model. It did not influence selection.']
    else:lines.append('The 17 submission contract checks are NOT APPLICABLE: no new submission package exists. CV leakage/replay checks were completed separately.')
    lines+=['','DATA AND REPRODUCTION',
            'PROTEUS BCI Bordeaux, NEMAR nm000290 v1.0.0, CC-BY-4.0. Public manifest checksums verified. 112 source runs,84 cued runs,14 sessions,2655 windows.',
            'Preprocessing reuses preserved v5 source: 50/60Hz notches and harmonics,0.1–75Hz bandpass at500Hz, resample120Hz,float32,microvolts; four-second windows start one second after cue. Metadata matches preserved v5 row ordering.',
            'Label mapping: mi=0,sub=1,word=2. Input(1,41,480); only EEG reaches predict. No labels or metadata reach runtime adaptation.',
            'Original reference_v5 sources and extraction manifest remain preserved. Raw EEG and regenerable covariance/alignment caches are omitted from reproduce.zip; regeneration code, checksums, source manifests, all fitted checkpoints and predictions are retained.',
            'Recorded training environment: '+json.dumps(json.loads((ROOT/'environment_neural.json').read_text())),
            'Numerical library/hardware changes can alter fresh training; archived checkpoints preserve this measured run.',
            'Plan SHA-256: '+pr.sha(ROOT/'plan.json'),
            'Implementation freeze SHA-256: '+pr.sha(ROOT/'implementation_freeze.json'),
            'Data DOI: https://doi.org/10.82901/nemar.nm000290',
            'Official evaluator commit: '+p['competition_commit'],
            'Data credit: Dreyer Pauline; Bourdil Manon; Bechon Loic; Kojima Simon; Velut Sebastien; Rimbert Sebastien; Roy Raphaëlle; Lotte Fabien.',
            'No sealed-phase data or leaderboard feedback was used. No competition upload was performed.']
    if (ROOT/'data_transfer.json').exists():
        transfer=json.loads((ROOT/'data_transfer.json').read_text())
        lines.append('Data acquisition acceleration: the Mac used the official content-addressed NEMAR S3 mirror, independently checked every preserved manifest checksum, and ran the unchanged preprocessing source. Prepared arrays were hash-verified on the Dell. Preprocessing environment: '+json.dumps({k:transfer[k] for k in ['python','mne','numpy','scipy']})+'. Full transport provenance is in data_transfer.json.')
    dest=ROOT/'deliverables/report.txt';dest.parent.mkdir(exist_ok=True)
    dest.write_text('\n'.join(lines)+'\n')


def archives():
    dest=ROOT/'deliverables'
    with zipfile.ZipFile(dest/'verification.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted((ROOT/'verification').rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        for name in ['synthetic_check.json','plan.json','plan.sha256','implementation_freeze.json']:
            z.write(ROOT/name,name)
    files=[]
    skip_dirs={'data','packages','deliverables','__pycache__'}
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if p.name.startswith('._') or any(x in skip_dirs for x in rel.parts):continue
        if rel.parts[0]=='cache' and p.suffix not in ['.json','.npz']:continue
        if rel.parts[0]=='cache' and 'runs' in rel.parts:continue
        if p.suffix=='.npy':continue
        if p.name.endswith('.partial'):continue
        files.append(p)
    # Keep the source dataset manifest/license even though raw EEG is omitted.
    for rel in ['reference_v5/data/nm000290/v1.0.0/download_manifest.json',
                'reference_v5/data/nm000290/v1.0.0/dataset_description.json',
                'data/nm000290/v1.0.0/download_verification.json',
                'data/nm000290/v1.0.0/mirror_verification.json']:
        if (ROOT/rel).is_file():files.append(ROOT/rel)
    files=sorted(set(files))
    manifest={str(p.relative_to(ROOT)):pr.sha(p) for p in files}
    with zipfile.ZipFile(dest/'reproduce.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(Path('v7_work')/p.relative_to(ROOT)))
        z.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        z.write(dest/'report.txt','report.txt')
        for name in ['submission.zip','legacy_submission.zip']:
            if (dest/name).exists():z.write(dest/name,name)
    with zipfile.ZipFile(dest/'reproduce.zip') as z:
        assert z.testzip() is None
        for rel,h in manifest.items():
            assert hashlib.sha256(z.read('v7_work/'+rel)).hexdigest()==h
    summary={p.name:{'bytes':p.stat().st_size,'sha256':pr.sha(p)} for p in sorted(dest.iterdir()) if p.is_file()}
    pr.json_write(dest/'SHA256SUMS.json',summary)
    pr.json_write(ROOT/'status.json',{'stage':'complete','selected':json.loads((ROOT/'analysis/selection.json').read_text())['selected'],
                                     'artifacts':summary,'original_v6_recovered':False})


def main():
    threadpool_limits(limits=2);torch.set_num_threads(2);pr.freeze_implementation()
    if not (ROOT/'analysis').exists():calculate(ROOT/'plan.json',ROOT/'predictions',ROOT/'analysis')
    selection=json.loads((ROOT/'analysis/selection.json').read_text())
    assert selection['plan_sha256']==pr.sha(ROOT/'plan.json')
    assert selection['replay_manifest_sha256']==pr.sha(ROOT/'predictions/replay_manifest.json')
    isolated_reanalysis()
    if selection['selected']:postselection(selection)
    else:
        assert not (ROOT/'deliverables/submission.zip').exists()
        assert not (ROOT/'deliverables/legacy_submission.zip').exists()
    write_report(selection);archives()


if __name__=='__main__':main()
