"""Prepare verified public inputs locally and stage a hashed GPU-host transfer."""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__':
    receipt = ROOT / 'data/nm000290/v1.0.0/mirror_verification.json'
    deadline = time.monotonic() + 4 * 3600
    while not receipt.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Mirror download receipt absent')
        time.sleep(10)
    assert json.loads(receipt.read_text())['complete']
    # This external volume creates AppleDouble ._ files which match the EDF
    # glob in preserved code. Stage only manifest-listed bytes on local APFS;
    # execute the frozen source unchanged and independently recheck every file.
    with tempfile.TemporaryDirectory(prefix='ni26-prepare-') as temp:
        stage = Path(temp)
        for rel in ['prepare_data.py',
                    'reference_v5/data/nm000290/v1.0.0/download_manifest.json',
                    'reference_v5/v5_work/prepare_public.py',
                    'reference_v5/v5_work/cache/metadata.npz']:
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
        datarel = Path('data/nm000290/v1.0.0')
        entries = json.loads((stage / 'reference_v5' / datarel / 'download_manifest.json').read_text())
        for entry in entries:
            dest = stage / datarel / entry['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / datarel / entry['path'], dest)
        subprocess.run([sys.executable, '-u', str(stage / 'prepare_data.py')], check=True)
        shutil.copytree(stage / 'cache', ROOT / 'cache', dirs_exist_ok=True,
                        copy_function=shutil.copyfile)
        shutil.copyfile(stage / datarel / 'download_verification.json',
                        ROOT / datarel / 'download_verification.json')
    import mne
    import numpy as np
    import scipy
    files = [ROOT / 'cache' / n for n in [
        'windows.npy', 'metadata.npz', 'dataset_info.json', 'metadata_recovery_check.json']]
    files += [receipt, receipt.with_name('download_verification.json'),
              ROOT / 'mirror_data.py', ROOT / 'prepare_local.py']
    record = {'purpose': 'Transport-only acceleration; frozen preprocessing unchanged',
              'preprocessing_host': 'Mac, manifest-only APFS staging (AppleDouble sidecars excluded)',
              'python': platform.python_version(),
              'mne': mne.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'prepare_data_sha256': sha(ROOT / 'prepare_data.py'),
              'prepare_public_sha256': sha(ROOT / 'reference_v5/v5_work/prepare_public.py'),
              'files': {str(p.relative_to(ROOT)): sha(p) for p in files}}
    recordpath = ROOT / 'data_transfer.json'
    recordpath.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    files.append(recordpath)
    archive = ROOT / 'prepared_data_transfer.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as output:
        for path in files:
            output.add(path, arcname='v7_work/' + str(path.relative_to(ROOT)), recursive=False)
    (ROOT / 'prepared_data_transfer.sha256').write_text(sha(archive) + '\n')
    print('PREPARED_TRANSFER_READY', archive.stat().st_size, sha(archive), flush=True)
