"""Resume parallel ranges and verify the full completed archive before publishing."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SHA = '7a101ef47340cbb4d2b9a331329cd35b61ca1fb8476c5aef8aece8632cb980ff'
SIZE = 1269782187
CHUNK = 8 * 1024 * 1024
URL = 'http://100.71.130.36:18767/' + SHA
PARTS = ROOT / 'v7_transfer_chunks'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(index):
    lo = index * CHUNK
    hi = min(SIZE, lo + CHUNK) - 1
    path = PARTS / f'{index:04d}.part'
    for attempt in range(20):
        offset = path.stat().st_size if path.exists() else 0
        assert offset <= hi - lo + 1
        if offset == hi - lo + 1:
            return
        start = lo + offset
        try:
            request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{hi}'})
            with OPENER.open(request, timeout=40) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {start}-{hi}/{SIZE}'
                assert response.headers['X-Archive-SHA256'] == SHA
                with path.open('ab') as handle:
                    while True:
                        data = response.read(65536)
                        if not data:
                            break
                        handle.write(data)
            assert path.stat().st_size == hi - lo + 1
            return
        except Exception as error:
            print(json.dumps({'chunk': index, 'attempt': attempt + 1, 'error': repr(error)}), flush=True)
            if attempt == 19:
                raise
            time.sleep(min(2 ** attempt, 15))


if __name__ == '__main__':
    PARTS.mkdir(exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for done, future in enumerate(cf.as_completed(futures), 1):
            future.result()
            print(json.dumps({'completed_chunks': done, 'total_chunks': count,
                              'elapsed_s': round(time.time() - started, 1),
                              'bytes_received': sum(p.stat().st_size for p in PARTS.glob('*.part'))}), flush=True)
    target = ROOT / 'v7_deliverables/reproduce.verified.zip'
    digest = hashlib.sha256()
    with target.open('wb') as output:
        for index in range(count):
            with (PARTS / f'{index:04d}.part').open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
                    output.write(block)
    assert target.stat().st_size == SIZE and digest.hexdigest() == SHA
    print('COMPLETE_ARCHIVE_SHA256_VERIFIED', SHA, flush=True)
