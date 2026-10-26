"""Fit the frozen winning v5 recipe on all public subjects or the legacy six.

This executable refuses to fit anything unless the immutable CV selection
passes the predeclared paired-gain rule. It never evaluates another subject.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'bci_work')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from neural_cov import SpatialLogVarianceNet, fit_source_reference
from train_neural_cov import stream_covariances
from temporal_sweep import cell_class_weights
from v5_work.neural_training_v5 import train_model, recipe_definitions
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, LEGACY_TRAIN_SUBJECTS, assert_stage_subjects,
    verify_selection_freeze, make_provenance, register_training_artifact,
    check_training_artifacts, sha256_file, write_json_exclusive,
)


def verify_postselection_inputs(selection_freeze, plan_path, cache, cv_assembly):
    """Bind post-selection EEG inputs to the exact passing CV development data.

    This guard can also precede the unchanged classical post-selection runner.
    It performs no fitting, inference, scoring, or selection.
    """
    cache, plan_path, cv_assembly = map(Path, (cache, plan_path, cv_assembly))
    frozen = verify_selection_freeze(selection_freeze)
    plan_hash = sha256_file(plan_path)
    assert plan_hash == frozen['plan_sha256']
    selection_path = Path(frozen['selection_path'])
    candidates = []
    if 'v5_work' in selection_path.parts:
        candidates.append(ROOT.joinpath(*selection_path.parts[selection_path.parts.index('v5_work'):]))
    candidates.extend((Path(selection_freeze).resolve().parent / selection_path.name, selection_path))
    selection_path = next((path for path in candidates if path.is_file()), None)
    if selection_path is None:
        raise FileNotFoundError('Cannot locate the immutable passing selection')
    selection = json.loads(selection_path.read_text())
    assert sha256_file(selection_path) == frozen['selection_sha256']
    assert sha256_file(cv_assembly) == selection['assembly_manifest_sha256']
    assembly = json.loads(cv_assembly.read_text())
    assert assembly['complete'] is True and assembly['all_actual_fit_rows_match'] is True
    assert assembly['plan_sha256'] == plan_hash and assembly['n_artifacts'] == 270
    assert sha256_file(cache / 'metadata.npz') == assembly['metadata_sha256']
    assert sha256_file(cache / 'temporal_band_covs.npy') == assembly['raw_covariances_sha256']
    return frozen, plan_hash, assembly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('final', 'legacy'), required=True)
    ap.add_argument('--selection-freeze', type=Path, required=True)
    ap.add_argument('--plan', type=Path, default=ROOT / 'v5_work/plan.json')
    ap.add_argument('--cache', type=Path, default=ROOT / 'v5_work/cache')
    ap.add_argument('--cv-assembly', type=Path, default=ROOT / 'v5_work/cv_predictions/assembly_manifest.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'v5_work/postselection')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    threadpool_limits(args.threads)
    torch.set_num_threads(args.threads)
    frozen, plan_hash, assembly = verify_postselection_inputs(
        args.selection_freeze, args.plan, args.cache, args.cv_assembly)
    plan = json.loads(args.plan.read_text())
    assert plan['neural_definitions'] == recipe_definitions()
    assert plan['final_seed_pair'] == [33, 42]
    assert plan['fixed_epochs'] == [75, 20]
    candidate = frozen['selected_candidate']
    recipe = 'baseline' if candidate == 'riemannian_tangent' else candidate
    assert recipe in ('augmentation', 'adversarial', 'masked_ssl', 'baseline')
    fitting_stage = args.stage + '_fit'
    with np.load(args.cache / 'metadata.npz', allow_pickle=False) as archive:
        metadata = {name: archive[name] for name in archive.files}
    ids = PUBLIC_SUBJECTS if args.stage == 'final' else LEGACY_TRAIN_SUBJECTS
    rows = np.flatnonzero(np.isin(metadata['subject'], ids))
    meta = {name: value[rows] for name, value in metadata.items()}
    subjects = assert_stage_subjects(meta['subject'], stage=fitting_stage,
                                     selection_freeze=args.selection_freeze)
    out = args.out / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('A completed or partial post-selection fit is not overwritten')
    started = time.perf_counter()
    raw_path = args.cache / 'temporal_band_covs.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    fit_raw = np.array(raw[rows], copy=True)
    # The first fitted quantity receives only the asserted stage fitting rows.
    alignment = plan['neural_alignment']
    source = fit_source_reference(fit_raw, meta['stream_id'],
                                  normalization=alignment['normalization'])
    reference_path = out / 'reference.npz'
    np.savez_compressed(reference_path, source=source, fit_rows=rows)
    reference_provenance = make_provenance(
        stage=fitting_stage, fitted_subjects=subjects,
        pretraining_subjects=(), normalization_subjects=subjects,
        reference_subjects=subjects, feature_scale_subjects=(),
        batchnorm_subjects=(), checkpoint_selection_subjects=(),
        row_indices=rows, selection_freeze=args.selection_freeze,
        config=dict(artifact_kind='postselection_neural_source', plan_sha256=plan_hash,
                    metadata_sha256=assembly['metadata_sha256'],
                    raw_covariances_sha256=assembly['raw_covariances_sha256'], alignment=alignment,
                    candidate=candidate, selected_recipe=recipe),
    )
    manifests = [str(register_training_artifact(reference_path, reference_provenance))]
    fit_cov = stream_covariances(fit_raw, meta['stream_id'], source, alignment)
    del raw, fit_raw
    loss_weights = cell_class_weights(meta)
    assert fit_cov.shape == (len(rows), 14, 41, 41)
    for seed, epochs in zip(plan['final_seed_pair'], plan['fixed_epochs']):
        model, history, training_info = train_model(
            fit_cov, meta['y'], loss_weights, meta['subject'],
            plan['neural_definitions']['model'], seed, epochs,
            recipe=recipe, schedule_epochs=plan['schedule_epochs'],
        )
        config = dict(recipe=recipe, candidate=candidate,
                      model_config=plan['neural_definitions']['model'],
                      neural_definitions=plan['neural_definitions'],
                      alignment=alignment, plan_sha256=plan_hash,
                      reference_sha256=sha256_file(reference_path),
                      fixed_epochs=epochs, cosine_horizon=plan['schedule_epochs'])
        provenance = make_provenance(
            stage=fitting_stage, fitted_subjects=subjects,
            pretraining_subjects=subjects if recipe == 'masked_ssl' else (),
            normalization_subjects=subjects, reference_subjects=subjects,
            feature_scale_subjects=subjects if recipe == 'masked_ssl' else (),
            batchnorm_subjects=subjects, checkpoint_selection_subjects=(),
            seed=seed, epoch=epochs, config=config, row_indices=rows,
            selection_freeze=args.selection_freeze,
        )
        checkpoint = dict(
            model_type='spatial_logvariance', model_config=plan['neural_definitions']['model'],
            state_dict=model.state_dict(), source=torch.from_numpy(source.copy()),
            alignment_config=alignment, training_subjects=subjects, heldout_subject=None,
            seed=int(seed), epoch=int(epochs), scheduler_T_max=plan['schedule_epochs'],
            recipe=recipe, provenance=provenance,
        )
        path = out / f'seed{seed}_epoch{epochs}.pt'
        torch.save(checkpoint, path)
        manifests.append(str(register_training_artifact(path, provenance)))
        write_json_exclusive(path.with_suffix('.training.json'),
                             dict(history=history, training_info=training_info))
        # Ordinary deployment architecture and a tensor-only checkpoint.
        restored = torch.load(path, weights_only=True, map_location='cpu')
        check = SpatialLogVarianceNet(**restored['model_config'])
        check.load_state_dict(restored['state_dict'], strict=True)
        del model, checkpoint, restored, check
        print(json.dumps(dict(stage=args.stage, seed=seed, epochs=epochs,
                              elapsed_seconds=time.perf_counter() - started)), flush=True)
    audit = check_training_artifacts(manifests)
    write_json_exclusive(out / 'fit_summary.json', dict(
        stage=fitting_stage, candidate=candidate, recipe=recipe,
        fitting_subjects=subjects, n_fitting_windows=len(rows),
        seed_pair=plan['final_seed_pair'], fixed_epochs=plan['fixed_epochs'],
        plan_sha256=plan_hash, selection_freeze_sha256=sha256_file(args.selection_freeze),
        elapsed_seconds=time.perf_counter() - started, evaluated_windows=0,
        artifacts=audit, manifests=manifests,
    ))


if __name__ == '__main__':
    main()
