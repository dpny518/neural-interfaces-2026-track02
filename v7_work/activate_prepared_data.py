"""Validate the prepared-data transfer and resume the authorized Dell workflow."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
EXPECTED = '8ad650dda971a38301d8404ca64bc649d91631f7d456725391cc567cdf28d184'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__':
    assert str(PROJECT) == '/mnt/18t/neural-interfaces-2026-track02-v7'
    archive = PROJECT / 'prepared_data_transfer.tar.gz'
    assert sha(archive) == EXPECTED
    assert not list((ROOT / 'runs').rglob('*.pt')), 'Fitting already started'
    assert not (ROOT / 'verification/data_transfer_validation.json').exists(), 'Already activated'
    with tempfile.TemporaryDirectory(prefix='incoming-prepared-', dir=PROJECT) as temp:
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                rel = Path(member.name)
                assert rel.parts[0] == 'v7_work' and not rel.is_absolute() and '..' not in rel.parts
                assert member.isfile(), 'Only plain files expected'
            bundle.extractall(temp, filter='data')
        incoming = Path(temp) / 'v7_work'
        record = json.loads((incoming / 'data_transfer.json').read_text())
        for rel, expected in record['files'].items():
            assert sha(incoming / rel) == expected, rel
        assert sha(ROOT / 'prepare_data.py') == record['prepare_data_sha256']
        assert sha(ROOT / 'reference_v5/v5_work/prepare_public.py') == record['prepare_public_sha256']
        for name in ['download_verification.json', 'mirror_verification.json']:
            receipt = json.loads((incoming / 'data/nm000290/v1.0.0' / name).read_text())
            assert receipt['complete'] and len(receipt['files']) == 587
            assert receipt['manifest_sha256'] == sha(ROOT / 'reference_v5/data/nm000290/v1.0.0/download_manifest.json')
        # These are only this round's obsolete downloader and coordinating jobs.
        stopped = []
        for pid, script in [(1566794, 'workflow.py'), (1568237, 'orchestrate.py'), (1568236, 'prepare_data.py')]:
            proc = Path('/proc') / str(pid) / 'cmdline'
            if proc.exists():
                args = proc.read_bytes().decode().split('\0')
                assert any(arg.endswith('v7_work/' + script) for arg in args), (pid, script)
                os.kill(pid, signal.SIGTERM)
                stopped.append({'pid': pid, 'script': script})
        time.sleep(2)
        assert not (ROOT / 'cache/metadata_recovery_check.json').exists()
        # All inputs were validated before publishing the ready marker.
        files = [p for p in incoming.rglob('*') if p.is_file()]
        files.sort(key=lambda p: p.name == 'metadata_recovery_check.json')
        for path in files:
            dest = ROOT / path.relative_to(incoming)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        verification = ROOT / 'verification'
        verification.mkdir(exist_ok=True)
        (verification / 'data_transfer_validation.json').write_text(json.dumps({
            'passed': True, 'archive_sha256': EXPECTED, 'verified_files': record['files'],
            'frozen_preprocessing_source_unchanged': True, 'stopped_own_jobs': stopped,
            'timestamp_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    log = open(PROJECT / 'workflow.log', 'ab', buffering=0)
    log.write(b'\nPREPARED_DATA_TRANSFER_VALIDATED_WORKFLOW_RESTART\n')
    job = subprocess.Popen([str(PROJECT / '.venv/bin/python'), '-u', str(ROOT / 'workflow.py')],
                           cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'workflow_pid': job.pid, 'data_transfer_verified': True}), flush=True)
