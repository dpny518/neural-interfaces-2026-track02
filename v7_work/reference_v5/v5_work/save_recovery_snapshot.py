"""Preserve immutable completed folds without reading or scoring predictions."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'v5_work'
files = {}

def add(path):
    path = Path(path)
    if path.is_file() and '__pycache__' not in path.parts:
        files[path.relative_to(ROOT).as_posix()] = path

for path in WORK.iterdir():
    if path.is_file() and path.suffix in {'.py', '.json', '.txt', '.sha256'}:
        add(path)
for path in (ROOT / 'bci_work').glob('*.py'):
    add(path)
add(ROOT / 'methods_draft/riemann_features.py')
for path in (WORK / 'source_snapshots').rglob('*.py'):
    add(path)
for name in ('metadata.npz', 'dataset_info.json', 'feature_config_reproducible.json'):
    add(WORK / 'cache' / name)
for path in (WORK / 'fold_cache').rglob('*'):
    if path.name in {'reference.npz', 'reference.provenance.json', 'cache_checksums.json'}:
        add(path)
completed = {}
for recipe in ('baseline', 'augmentation', 'adversarial', 'masked_ssl'):
    directory = WORK / 'neural_results' / recipe
    folds = sorted(p.parent for p in directory.glob('hold_*/isolation_check.json'))
    completed[recipe] = [p.name for p in folds]
    for fold in folds:
        for path in fold.iterdir():
            add(path)
    if (directory / 'summary.json').exists():
        add(directory / 'summary.json')
folds = sorted(p.parent for p in (WORK / 'classical_results').glob('hold_*/fit_summary.json'))
completed['classical'] = [p.name for p in folds]
for fold in folds:
    for path in fold.iterdir():
        add(path)
for path in (WORK / 'classical_results').iterdir():
    if path.is_file() and path.suffix in {'.json', '.npz'}:
        add(path)
public = ROOT / 'data/nm000290/v1.0.0'
for name in ('download_manifest.json', 'download_verification_all-public-development.json',
             'dataset_description.json', 'README.md'):
    add(public / name)
out = ROOT / 'v5_recovery'
out.mkdir(exist_ok=True)
archive = out / 'v5_progress_checkpoint.zip'
hashes = {}
record = dict(schema='track2-v5-incomplete-recovery-only-v1',
              created_at_utc=datetime.now(timezone.utc).isoformat(),
              completed_folds=completed, scores_computed=False,
              usable_as_submission=False, selection_permitted=False,
              note='Incomplete run recovery only. Restore completed folds, regenerate omitted public caches, and finish the fixed full panel before analysis.')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
    for name, path in sorted(files.items()):
        data = path.read_bytes()
        assert data == path.read_bytes(), 'Source changed during recovery snapshot: ' + name
        hashes[name] = hashlib.sha256(data).hexdigest()
        handle.writestr(name, data)
    handle.writestr('RECOVERY_STATUS.json', json.dumps(record, indent=2) + '\n')
    handle.writestr('SHA256SUMS.txt', ''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
with zipfile.ZipFile(archive) as handle:
    assert handle.testzip() is None
    for name, digest in hashes.items():
        assert hashlib.sha256(handle.read(name)).hexdigest() == digest
record.update(archive=str(archive), bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), n_files=len(files))
(out / 'recovery_info.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
