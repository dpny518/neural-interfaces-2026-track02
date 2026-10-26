"""Download manifest-identical public bytes via NEMAR's official S3 mirror.

This acquisition helper does not change preprocessing or the frozen experiment.
Every published file is checked against the preserved v5 download manifest.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import os
import ssl
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'data/nm000290/v1.0.0'


def digest(raw, algorithm):
    if algorithm == 'git':
        return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    return hashlib.new(algorithm, raw).hexdigest()


def main():
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    manifest = ROOT / 'reference_v5/data/nm000290/v1.0.0/download_manifest.json'
    entries = json.loads(manifest.read_text())
    def fetch(e):
        rel = Path(e['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        path = DEST / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        def valid(raw):
            return len(raw) == e['size'] and digest(raw, e['checksum_algorithm']) == e['checksum']
        if path.exists() and valid(path.read_bytes()):
            return {'path': str(rel), 'bytes': e['size'], 'reused': True}
        url = e['url']
        if e['checksum_algorithm'] == 'sha256':
            key = f"SHA256E-s{e['size']}--{e['checksum']}{rel.suffix}"
            url = 'https://nemar.s3.us-east-2.amazonaws.com/nm000290/objects/' + key
        partial = path.with_name(path.name + '.mirror.partial')
        for attempt in range(16):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= e['size']:
                    if valid(partial.read_bytes()):
                        os.replace(partial, path)
                        return {'path': str(rel), 'bytes': e['size'], 'url': url}
                    partial.write_bytes(b'')
                    offset = 0
                headers = {'User-Agent': 'NeuralInterfaces-reproducible-training/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=45, context=context) as response:
                    append = offset > 0 and response.status == 206
                    if append:
                        assert response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                    with partial.open('ab' if append else 'wb') as output:
                        while True:
                            block = response.read(65536)
                            if not block:
                                break
                            output.write(block)
                assert valid(partial.read_bytes()), 'manifest checksum mismatch'
                os.replace(partial, path)
                return {'path': str(rel), 'bytes': e['size'], 'url': url}
            except Exception as error:
                print(json.dumps({'retry': str(rel), 'attempt': attempt + 1, 'error': repr(error)}), flush=True)
                if attempt == 15:
                    raise
                time.sleep(min(2 ** attempt, 20))
    records = []
    errors = []
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch, e): e for e in reversed(entries)}
        for future in cf.as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({'path': futures[future]['path'], 'error': repr(error)})
            if len(records) % 20 == 0:
                print(json.dumps({'verified_files': len(records), 'total': len(entries),
                                  'verified_bytes': sum(r['bytes'] for r in records),
                                  'elapsed_s': round(time.time() - start, 1), 'errors': errors}), flush=True)
    result = {'complete': not errors and len(records) == len(entries), 'errors': errors,
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'files': records, 'elapsed_s': time.time() - start}
    (DEST / 'mirror_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['complete'], errors
    print('MIRROR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
