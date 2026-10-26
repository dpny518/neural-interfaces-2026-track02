"""Recover the exact versioned public inputs; never select models here."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def download():
    manifest = ROOT / 'reference_v5/data/nm000290/v1.0.0/download_manifest.json'
    entries = json.loads(manifest.read_text())
    dest = ROOT / 'data/nm000290/v1.0.0'
    dest.mkdir(parents=True, exist_ok=True)
    def checksum(raw, algorithm):
        if algorithm == 'git':
            return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        return hashlib.new(algorithm, raw).hexdigest()
    def fetch(e):
        rel = Path(e['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        def valid(raw):
            return len(raw) == e['size'] and checksum(raw, e['checksum_algorithm']) == e['checksum']
        if p.is_file() and valid(p.read_bytes()):
            return {'path': str(rel), 'bytes': e['size'], 'verified': True}
        tmp = p.with_name(p.name + '.partial')
        for attempt in range(12):
            try:
                offset = tmp.stat().st_size if tmp.exists() else 0
                if offset >= e['size']:
                    raw = tmp.read_bytes()
                    if valid(raw):
                        os.replace(tmp,p)
                        return {'path':str(rel),'bytes':e['size'],'verified':True}
                    tmp.write_bytes(b'');offset=0
                headers={'User-Agent':'NeuralInterfaces-reproducible-training/1.0'}
                if offset:headers['Range']=f'bytes={offset}-'
                request=urllib.request.Request(e['url'],headers=headers)
                with urllib.request.urlopen(request, timeout=45) as response:
                    append=offset>0 and response.status==206
                    if append:
                        assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    with tmp.open('ab' if append else 'wb') as output:
                        while True:
                            block=response.read(65536)
                            if not block:break
                            output.write(block)
                raw = tmp.read_bytes()
                assert valid(raw), 'Manifest checksum mismatch: ' + str(rel)
                os.replace(tmp, p)
                return {'path': str(rel), 'bytes': e['size'], 'verified': True}
            except Exception as error:
                print(json.dumps({'retry':str(rel),'attempt':attempt+1,'error':repr(error),
                                  'partial_bytes':tmp.stat().st_size if tmp.exists() else 0}),flush=True)
                if attempt == 11:
                    raise
                time.sleep(min(2 ** attempt,30))
    records = []
    errors=[]
    print('DOWNLOAD_START',len(entries),'manifest entries',flush=True)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,e):e for e in entries}
        for future in cf.as_completed(futures):
            try:records.append(future.result())
            except Exception as error:
                errors.append({'path':futures[future]['path'],'error':repr(error)})
                print('DOWNLOAD_ERROR',json.dumps(errors[-1]),flush=True)
            if len(records) % 20 == 0:
                print('download', len(records), '/', len(entries),'verified_bytes',sum(r['bytes'] for r in records),flush=True)
    if errors:
        (dest/'download_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
        raise RuntimeError(f'{len(errors)} downloads failed; rerun to resume verified/partial files')
    (dest / 'download_verification.json').write_text(json.dumps({
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'files': records, 'complete': True}, indent=2) + '\n')
    return dest


if __name__ == '__main__':
    dest = download()
    subprocess.run([sys.executable, str(ROOT / 'reference_v5/v5_work/prepare_public.py'),
                    '--data-dir', str(dest), '--output-dir', str(ROOT / 'cache'),
                    '--workers', '3'], check=True)
    import numpy as np
    with np.load(ROOT / 'cache/metadata.npz', allow_pickle=False) as fresh, np.load(
            ROOT / 'reference_v5/v5_work/cache/metadata.npz', allow_pickle=False) as old:
        checked = []
        for key in old.files:
            if key == 'source_file':
                continue
            # Historical snapshots may use the explicit BIDS prefix.
            a, b = fresh[key], old[key]
            if key == 'subject':
                a = np.char.replace(a.astype(str), 'sub-', '')
                b = np.char.replace(b.astype(str), 'sub-', '')
            np.testing.assert_array_equal(a, b, err_msg=key)
            checked.append(key)
    (ROOT / 'cache/metadata_recovery_check.json').write_text(json.dumps(
        {'passed': True, 'checked_fields': checked, 'windows': 2655}, indent=2) + '\n')
    print('DATA_READY', flush=True)
