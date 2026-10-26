"""Verify the received completed artifacts without modifying their bytes."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'v7_deliverables'


def digest(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    expected = json.loads((DEST / 'SHA256SUMS.json').read_text())
    for name, record in expected.items():
        path = DEST / name
        assert path.stat().st_size == record['bytes'], name
        with path.open('rb') as handle:
            assert digest(handle) == record['sha256'], name
    with zipfile.ZipFile(DEST / 'reproduce.zip') as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names))
        assert not any(Path(n).name in ['submission.zip', 'legacy_submission.zip'] for n in names)
        assert not any('/postselection/' in n for n in names)
        assert sum(n.startswith('v7_work/runs/') and n.endswith('.pt') for n in names) == 460
        assert bundle.read('report.txt') == (DEST / 'report.txt').read_bytes()
        manifest = json.loads(bundle.read('SHA256SUMS.json'))
        for index, (rel, expected_hash) in enumerate(manifest.items(), 1):
            with bundle.open('v7_work/' + rel) as handle:
                assert digest(handle) == expected_hash, rel
            if index % 500 == 0:
                print('VERIFIED_ARCHIVE_MEMBERS', index, len(manifest), flush=True)
        assert bundle.testzip() is None
        selection = json.loads(bundle.read('v7_work/analysis/selection.json'))
        assert selection == json.loads((ROOT/'v7_review/reanalysis/selection.json').read_text())
        assert selection['null_result'] and selection['selected'] is None
        assert hashlib.sha256(bundle.read('v7_work/plan.json')).hexdigest() == '0448fed4aed8eb3b84e399204d1ec87b805f6e7824d3d3b02a41d176cb74950f'
        assert hashlib.sha256(bundle.read('v7_work/implementation_freeze.json')).hexdigest() == '16c85f05472c629b8093721d6fd0d9d451bb0065ab3bac1fe904b110a252a139'
    with zipfile.ZipFile(DEST / 'verification.zip') as bundle:
        assert bundle.testzip() is None
    record = json.loads((DEST / 'local_review.json').read_text())
    record.update(full_reproduce_archive_download_complete=True, all_deliverable_hashes_verified=True,
                  all_zip_crc_checks_passed=True, embedded_manifest_files_verified=len(manifest),
                  archived_cv_artifacts=460, no_new_submission_packages=True,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    (DEST / 'local_review.json').write_text(json.dumps(record, indent=2) + '\n')
    status = {'stage': 'complete', 'selected': None, 'null_result': True,
              'original_v6_recovered': False, 'baseline': 'reconstructed_v6',
              'measured_run_host': 'Dell RTX3090', 'local_alternative_data_cache_used': False,
              'local_deliverables': str(DEST), 'artifacts': expected,
              'delivery_verification': record, 'cloud_storage': json.loads((DEST/'bay_transfer_receipt.json').read_text())}
    (ROOT/'v7_work/status.json').write_text(json.dumps(status, indent=2)+'\n')
    print('ALL_DELIVERABLES_VERIFIED', json.dumps(record), flush=True)
