"""Archive a completed v5 result without raw EEG or regenerable large caches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v5_work.protocol_v5 import sha256_file, write_json_exclusive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'v5_deliverables')
    args = ap.parse_args()
    selection_path = ROOT / 'v5_work/cv_analysis/selection.json'
    selection = json.loads(selection_path.read_text())
    replay_path = ROOT / 'v5_work/cv_streaming_check/streaming_replay_check.json'
    replay = json.loads(replay_path.read_text())
    assert replay['full_panel_certified'] is True
    assert replay['total_predict_calls'] == 39825
    assert replay['plan_sha256'] == selection['plan_sha256']
    assert replay['exact_predicted_labels'] is True
    assembly = json.loads((ROOT / 'v5_work/cv_predictions/assembly_manifest.json').read_text())
    assert assembly['n_artifacts'] == 270 and assembly['complete'] is True
    for recipe in ('baseline', 'augmentation', 'adversarial', 'masked_ssl'):
        summary = json.loads((ROOT / 'v5_work/neural_results' / recipe / 'summary.json').read_text())
        assert summary['n_models'] == 60 and len(summary['subjects_completed']) == 10
    report = args.out / 'report.txt'
    assert report.is_file() and report.stat().st_size > 1000
    adopted = selection['decision']['adopt']
    if adopted:
        contract = json.loads((args.out / 'contract_check.json').read_text())
        assert contract['all_passed'] is True and contract['n_checks'] == 17
        assert (args.out / 'submission.zip').is_file()
        legacy = json.loads((ROOT / 'v5_work/legacy_diagnostic/result.json').read_text())
        assert legacy['passes'] == 1 and legacy['used_for_selection'] is False
        assert legacy['independent_test_estimate'] is False
    else:
        assert not (args.out / 'submission.zip').exists()
        assert not (ROOT / 'v5_work/postselection').exists()
        assert not (ROOT / 'v5_work/legacy_diagnostic/result.json').exists()
    files = {}

    def add(path, name=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        arcname = name or path.relative_to(ROOT).as_posix()
        assert not Path(arcname).is_absolute() and '..' not in Path(arcname).parts
        if arcname in files:
            assert files[arcname] == path
        files[arcname] = path

    add(report, 'report.txt')
    add(args.out / 'report_evidence.json', 'report_evidence.json')
    add(ROOT / 'v5_work/README_reproduce.txt', 'README.txt')
    # Keep all original numerical/training helper source, never old scores.
    for path in sorted((ROOT / 'bci_work').glob('*.py')):
        add(path)
    add(ROOT / 'methods_draft/riemann_features.py')
    for path in sorted((ROOT / 'v5_work').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(ROOT / 'v5_work')
        if relative.parts[0] == 'cache':
            if relative.as_posix() not in {
                'cache/metadata.npz', 'cache/dataset_info.json', 'cache/feature_config_reproducible.json',
            }:
                continue
        if relative.parts[0] == 'fold_cache':
            # Keep only finalized reference evidence. Temporary copies of the
            # regenerable covariance arrays may carry extra filename suffixes.
            if relative.as_posix() != 'fold_cache/cache_checksums.json' and not (
                len(relative.parts) == 3
                and relative.parts[1] in {f'P{i:02d}' for i in range(1, 11)}
                and relative.name in {'reference.npz', 'reference.provenance.json', 'cache_checksums.json'}
            ):
                continue
        if path.suffix in {'.partial', '.pyc'}:
            continue
        add(path)
    public = ROOT / 'data/nm000290/v1.0.0'
    for name in ('download_manifest.json', 'download_verification_all-public-development.json',
                 'dataset_description.json', 'README.md'):
        add(public / name)
    if adopted:
        for name in ('submission.zip', 'contract_check.json', 'build_info.json', 'source_manifests.json'):
            add(args.out / name)
        for name in ('submission.py', 'weights.pt'):
            add(args.out / 'submission' / name)
    # All learned CV artifacts must be present; no regeneration-only claim.
    cv_neural = [name for name in files if name.startswith('v5_work/neural_results/') and name.endswith('.pt')]
    cv_classical = [name for name in files if name.startswith('v5_work/classical_results/hold_') and name.endswith('.pt')]
    references = [name for name in files if name.startswith('v5_work/fold_cache/P') and name.endswith('/reference.npz')]
    assert len(cv_neural) == 240 and len(cv_classical) == 20 and len(references) == 10
    checksums = {name: sha256_file(path) for name, path in sorted(files.items())}
    checksum_text = ''.join(f'{digest}  {name}\n' for name, digest in checksums.items())
    archive = args.out / 'reproduce.zip'
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for name, path in sorted(files.items()):
            handle.write(path, arcname=name)
        handle.writestr('SHA256SUMS.txt', checksum_text)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert set(handle.namelist()) == set(files) | {'SHA256SUMS.txt'}
        for name, digest in checksums.items():
            assert hashlib.sha256(handle.read(name)).hexdigest() == digest, name
    result = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), archive=str(archive),
        bytes=archive.stat().st_size, sha256=sha256_file(archive), n_files=len(files) + 1,
        n_neural_cv_checkpoints=240, n_classical_cv_checkpoints=20, n_covariance_references=10,
        full_streaming_panel_certified=True, adoption=adopted,
        selected_candidate=selection['decision']['selected_candidate'],
        omitted_regenerable_data=['raw EDF recordings', 'processed EEG windows', 'aligned covariance caches'],
        plan_sha256=selection['plan_sha256'], report_sha256=sha256_file(report),
    )
    write_json_exclusive(args.out / 'reproduction_info.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
