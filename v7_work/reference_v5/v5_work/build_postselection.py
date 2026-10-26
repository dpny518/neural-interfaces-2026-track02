"""Package only the frozen, successfully replayed and refitted v5 winner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from v5_work.pack_runtime import build_runtime_source, pack_weights
from v5_work.contract_check_v5 import resolve_recorded_path
from v5_work.protocol_v5 import (
    verify_selection_freeze, check_training_artifacts, assert_stage_subjects,
    sha256_file, write_json_exclusive,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('final', 'legacy'), required=True)
    ap.add_argument('--selection-freeze', type=Path, required=True)
    ap.add_argument('--neural-root', type=Path, default=ROOT / 'v5_work/postselection')
    ap.add_argument('--classical-root', type=Path, default=ROOT / 'v5_work/classical_results')
    ap.add_argument('--cache', type=Path, default=ROOT / 'v5_work/cache')
    ap.add_argument('--replay-report', type=Path,
                    default=ROOT / 'v5_work/cv_streaming_check/streaming_replay_check.json')
    ap.add_argument('--cv-assembly', type=Path,
                    default=ROOT / 'v5_work/cv_predictions/assembly_manifest.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    frozen = verify_selection_freeze(args.selection_freeze)
    candidate = frozen['selected_candidate']
    replay = json.loads(args.replay_report.read_text())
    assert replay['full_panel_certified'] is True
    assert replay['plan_sha256'] == frozen['plan_sha256']
    assert replay['total_predict_calls'] == 39825
    assert replay['exact_predicted_labels'] is True
    selection_path = resolve_recorded_path(frozen['selection_path'], args.selection_freeze)
    selection = json.loads(selection_path.read_text())
    assert sha256_file(selection_path) == frozen['selection_sha256']
    assembly_hash = sha256_file(args.cv_assembly)
    assert assembly_hash == selection['assembly_manifest_sha256']
    assert any(Path(path).name == 'assembly_manifest.json'
               and resolve_recorded_path(path, args.replay_report) == args.cv_assembly.resolve()
               and digest == assembly_hash
               for path, digest in replay['files_unchanged'].items())
    source, source_hashes = build_runtime_source()
    assert hashlib.sha256(source.encode()).hexdigest() == replay['runtime_source_sha256']
    stage = args.stage + '_fit'
    neural_dir = args.neural_root / args.stage
    neural_paths = [neural_dir / 'seed33_epoch75.pt', neural_dir / 'seed42_epoch20.pt']
    classical_dir = args.classical_root / ('full_train' if args.stage == 'final' else 'legacy_six')
    classical_path = classical_dir / ('tangent.pt' if candidate == 'riemannian_tangent' else 'ridge.pt')
    manifest_paths = [neural_dir / 'reference.provenance.json']
    manifest_paths.extend(path.with_suffix('.provenance.json') for path in neural_paths)
    manifest_paths.append(classical_path.with_suffix('.provenance.json'))
    audit = check_training_artifacts(manifest_paths)
    assert audit['n_artifacts'] == 4
    for record in audit['artifacts']:
        assert record['stage'] == stage
        assert record['config']['plan_sha256'] == frozen['plan_sha256']
        artifact_freeze = verify_selection_freeze(record['selection_freeze'])
        assert all(artifact_freeze[key] == frozen[key] for key in
                   ('selection_sha256', 'plan_sha256', 'selected_candidate'))
        assert_stage_subjects(record['fitted_subjects'], stage=stage,
                              selection_freeze=args.selection_freeze)
    neural = [torch.load(path, map_location='cpu', weights_only=True) for path in neural_paths]
    classical = torch.load(classical_path, map_location='cpu', weights_only=True)
    info = json.loads((args.cache / 'dataset_info.json').read_text())
    packed = pack_weights(
        neural, classical, info['ch_names'], candidate=candidate,
        training_subjects=neural[0]['training_subjects'],
        extra_metadata=dict(stage=stage, selection_freeze_sha256=sha256_file(args.selection_freeze),
                            plan_sha256=frozen['plan_sha256'],
                            cv_streaming_verification_sha256=sha256_file(args.replay_report),
                            deployment_seed_pair=[33, 42], fixed_epochs=[75, 20]),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    folder = args.out / 'submission'
    if folder.exists():
        raise FileExistsError('A packaged submission is never overwritten')
    folder.mkdir()
    (folder / 'submission.py').write_text(source)
    torch.save(packed, folder / 'weights.pt')
    archive = args.out / ('submission.zip' if args.stage == 'final' else 'legacy_submission.zip')
    with zipfile.ZipFile(archive, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name in ('submission.py', 'weights.pt'):
            handle.write(folder / name, arcname=name)
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ['submission.py', 'weights.pt']
        assert handle.testzip() is None
    write_json_exclusive(args.out / 'source_manifests.json', dict(manifests=list(map(str, manifest_paths))))
    result = dict(stage=stage, candidate=candidate, training_subjects=packed['training_subjects'],
                  archive=str(archive), bytes=archive.stat().st_size,
                  sha256=sha256_file(archive), runtime_source_sha256=sha256_file(folder / 'submission.py'),
                  runtime_sources=source_hashes, weights_sha256=sha256_file(folder / 'weights.pt'),
                  source_artifacts=audit, selection_freeze_sha256=sha256_file(args.selection_freeze),
                  evaluator_score_used_to_change_package=False)
    write_json_exclusive(args.out / 'build_info.json', result)
    print(json.dumps({key: result[key] for key in ('stage', 'candidate', 'archive', 'bytes', 'sha256')}, indent=2))


if __name__ == '__main__':
    main()
