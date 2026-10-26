"""Private Bay transfer using the existing project's CLI and in-memory auth."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import os
import runpy
import shlex
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
LOCAL_CLI = Path('/Volumes/Untitled/Development/procen/cloudflare-storage/cli/bay')
REMOTE = '/mnt/18t/neural-interfaces-2026-track02-v7'
FOLDER = 'transfers/neural-interfaces-2026-track02/v7-20261025'
EXPECTED_SHA = '7a101ef47340cbb4d2b9a331329cd35b61ca1fb8476c5aef8aece8632cb980ff'
EXPECTED_SIZE = 1269782187


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def client():
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
    except ImportError:
        pass
    cli = LOCAL_CLI if LOCAL_CLI.exists() else HERE / 'bay_cli.py'
    module = runpy.run_path(str(cli), run_name='bay_client')
    config = module['load_config']()
    module['require_token'](config)
    return module, config


def launch():
    bay, config = client()
    payload = {'config': config, 'source': Path(__file__).read_text(), 'cli': LOCAL_CLI.read_text()}
    code = """import json,os,sys,subprocess
from pathlib import Path
p=json.load(sys.stdin);r=Path('/mnt/18t/neural-interfaces-2026-track02-v7')
(r/'bay_transfer.py').write_text(p['source']);(r/'bay_cli.py').write_text(p['cli'])
env=dict(os.environ,BAY_URL=p['config']['url'],BAY_TOKEN=p['config']['token'])
log=open(r/'bay_upload.log','ab',buffering=0)
job=subprocess.Popen([str(r/'.venv/bin/python'),'-u',str(r/'bay_transfer.py'),'upload'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'bay_upload_pid':job.pid,'folder':'transfers/neural-interfaces-2026-track02/v7-20261025','credential_persisted':False}))
"""
    # Authentication travels only in the encrypted SSH stdin and child memory.
    # It is never placed in a command line, file, log, or reproduction archive.
    result = subprocess.run(['ssh', '-S', '/tmp/ni26-ssh-control-v3',
                             '-o', 'ConnectTimeout=20', 'dell@100.71.130.36',
                             'python3 -c ' + shlex.quote(code)],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=120)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def upload():
    bay, config = client()
    folder = bay['resolve_folder'](config, FOLDER)
    path = HERE / 'bay_upload_state.json'
    state = json.loads(path.read_text()) if path.exists() else {'folder': FOLDER, 'files': {}}
    def save():
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, path)
    def request(method, url, **kwargs):
        for attempt in range(8):
            try:
                return bay['request'](config, method, url, **kwargs)
            except (SystemExit, OSError):
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 20))
    source = HERE / 'v7_work/deliverables'
    for name in ['report.txt', 'verification.zip', 'SHA256SUMS.json', 'reproduce.zip']:
        local = source / name
        size, digest = local.stat().st_size, sha(local)
        if name == 'reproduce.zip':
            assert size == EXPECTED_SIZE and digest == EXPECTED_SHA
        if name not in state['files']:
            started = bay['request'](config, 'POST', '/api/uploads',
                                     body={'parentId': folder, 'name': name, 'size': size})
            state['files'][name] = {**started, 'bytes': size, 'sha256': digest, 'parts': {}}
            save()
        entry = state['files'][name]
        assert entry['sha256'] == digest and entry['bytes'] == size
        if entry.get('complete'):
            continue
        uid = entry['id']
        if entry['mode'] == 'single':
            node = request('PUT', f'/api/uploads/{uid}', data=local.read_bytes())
        else:
            part_size = entry['partSize']
            count = (size + part_size - 1) // part_size
            def part(number):
                with local.open('rb') as handle:
                    handle.seek((number - 1) * part_size)
                    data = handle.read(part_size)
                response = request('PUT', f'/api/uploads/{uid}/parts/{number}', data=data)
                return number, {'partNumber': response['partNumber'], 'etag': response['etag']}
            todo = [n for n in range(1, count + 1) if str(n) not in entry['parts']]
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                for future in cf.as_completed([pool.submit(part, n) for n in todo]):
                    number, record = future.result()
                    entry['parts'][str(number)] = record
                    save()
                    print(json.dumps({'file': name, 'uploaded_parts': len(entry['parts']), 'total_parts': count}), flush=True)
            node = request('POST', f'/api/uploads/{uid}/complete', body={'parts': list(entry['parts'].values())})
        entry.update(complete=True, node=node)
        save()
        print('UPLOADED', name, size, digest, flush=True)
    state['complete'] = True
    save()
    print('BAY_TRANSFER_UPLOAD_COMPLETE', flush=True)


def download():
    bay, config = client()
    node = bay['resolve_file'](config, FOLDER + '/reproduce.zip')
    url = config['url'] + f'/api/nodes/{node}/content?download=1'
    headers = {'User-Agent': bay['USER_AGENT'], **bay['auth_headers'](config['token'])}
    root = HERE.parent
    parts = root / 'v7_transfer_chunks'
    parts.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    count = (EXPECTED_SIZE + chunk_size - 1) // chunk_size
    def fetch(index):
        lo = index * chunk_size
        hi = min(EXPECTED_SIZE, lo + chunk_size) - 1
        path = parts / f'{index:04d}.part'
        for attempt in range(12):
            offset = path.stat().st_size if path.exists() else 0
            assert offset <= hi - lo + 1
            if offset == hi - lo + 1:
                return
            start = lo + offset
            try:
                req = urllib.request.Request(url, headers={**headers, 'Range': f'bytes={start}-{hi}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{hi}/{EXPECTED_SIZE}'
                    with path.open('ab') as output:
                        for block in iter(lambda: response.read(65536), b''):
                            output.write(block)
                assert path.stat().st_size == hi - lo + 1
                return
            except (OSError, AssertionError):
                if attempt == 11:
                    raise
                time.sleep(min(2 ** attempt, 15))
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for done, future in enumerate(cf.as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
            future.result()
            print(json.dumps({'downloaded_parts': done, 'total_parts': count, 'elapsed_s': round(time.time()-started, 1)}), flush=True)
    dest = root / 'v7_deliverables/reproduce.verified.zip'
    with dest.open('wb') as output:
        for index in range(count):
            with (parts / f'{index:04d}.part').open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    output.write(block)
    assert dest.stat().st_size == EXPECTED_SIZE and sha(dest) == EXPECTED_SHA
    os.replace(dest, dest.with_name('reproduce.zip'))
    record = {'folder': FOLDER, 'node_id': node, 'bytes': EXPECTED_SIZE, 'sha256': EXPECTED_SHA,
              'download_verified': True, 'public_share_created': False}
    (root/'v7_deliverables/bay_transfer_receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print('BAY_DOWNLOAD_SHA256_VERIFIED', EXPECTED_SHA, flush=True)


def receive():
    bay, config = client()
    folder = bay['resolve_folder'](config, FOLDER)
    deadline = time.monotonic() + 1800
    while True:
        listing = bay['request'](config, 'GET', '/api/nodes?parent=' + folder)
        found = [n for n in listing['nodes'] if n['kind'] == 'file' and n['name'] == 'reproduce.zip']
        if found:
            assert len(found) == 1 and found[0]['size'] == EXPECTED_SIZE
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Complete Bay archive is not yet available')
        time.sleep(15)
    download()
    subprocess.run([sys.executable, '-u', str(HERE/'final_delivery_check.py')], check=True)


def probe():
    bay, config = client()
    node = bay['resolve_file'](config, FOLDER + '/reproduce.zip')
    headers = {'User-Agent': bay['USER_AGENT'], **bay['auth_headers'](config['token'])}
    url = config['url'] + f'/api/nodes/{node}/content?download=1'
    for request_range in ['bytes=1245184-8388607', None]:
        h = dict(headers)
        if request_range:
            h['Range'] = request_range
        req = urllib.request.Request(url+'&v7_probe='+str(time.time_ns()), headers=h)
        with urllib.request.urlopen(req, timeout=60) as response:
            block = response.read(4096)
            print(json.dumps({'requested': request_range, 'status': response.status,
                              'content_range': response.headers.get('Content-Range'),
                              'content_length': response.headers.get('Content-Length'),
                              'first4096_sha256': hashlib.sha256(block).hexdigest()}), flush=True)


def full_download():
    bay, config = client()
    node = bay['resolve_file'](config, FOLDER + '/reproduce.zip')
    headers = {'User-Agent': bay['USER_AGENT'], **bay['auth_headers'](config['token'])}
    root = HERE.parent
    target = root/'v7_deliverables/reproduce.bay.partial'
    for attempt in range(4):
        try:
            url = config['url'] + f'/api/nodes/{node}/content?download=1&v7_full={EXPECTED_SHA}&attempt={attempt}'
            req = urllib.request.Request(url, headers=headers)
            count = 0
            digest = hashlib.sha256()
            started = time.time()
            with urllib.request.urlopen(req, timeout=90) as response:
                assert response.status == 200
                assert int(response.headers['Content-Length']) == EXPECTED_SIZE
                with target.open('wb') as output:
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        count += len(block)
                        if count % (64*1024*1024) == 0:
                            print(json.dumps({'bytes_received':count,'total':EXPECTED_SIZE,
                                              'elapsed_s':round(time.time()-started,1)}),flush=True)
            assert count == EXPECTED_SIZE and digest.hexdigest() == EXPECTED_SHA
            os.replace(target,target.with_name('reproduce.zip'))
            record={'folder':FOLDER,'node_id':node,'bytes':EXPECTED_SIZE,'sha256':EXPECTED_SHA,
                    'download_verified':True,'public_share_created':False,'transfer_mode':'HTTP 200 full response',
                    'range_endpoint_observation':'Content-Range was bytes NaN-1269782186/1269782187; ranged responses were rejected'}
            (root/'v7_deliverables/bay_transfer_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
            print('BAY_DOWNLOAD_SHA256_VERIFIED',EXPECTED_SHA,flush=True)
            subprocess.run([sys.executable,'-u',str(HERE/'final_delivery_check.py')],check=True)
            return
        except (OSError,AssertionError) as error:
            print('FULL_DOWNLOAD_RETRY',attempt+1,type(error).__name__,flush=True)
            if attempt==3:
                raise
            time.sleep(5)


def verified_parts_download():
    bay, config = client()
    root = HERE.parent
    manifest = json.loads((root/'v7_deliverables/reproduce_chunk_hashes.json').read_text())
    assert manifest['sha256'] == EXPECTED_SHA and manifest['bytes'] == EXPECTED_SIZE
    assert sum(c['size'] for c in manifest['chunks']) == EXPECTED_SIZE
    node = bay['resolve_file'](config,FOLDER+'/reproduce.zip')
    base = config['url']+f'/api/nodes/{node}/content?download=1'
    headers = {'User-Agent':bay['USER_AGENT'],**bay['auth_headers'](config['token'])}
    parts = root/'v7_bay_verified_parts'
    parts.mkdir(exist_ok=True)
    started=time.time()
    def get(index):
        entry=manifest['chunks'][index]
        path=parts/f'{index:04d}.part'
        if path.exists() and path.stat().st_size==entry['size'] and sha(path)==entry['sha256']:
            return
        lo=index*manifest['chunk_size'];hi=lo+entry['size']-1
        for attempt in range(30):
            try:
                req=urllib.request.Request(base+f'&v7_part={index}&attempt={attempt}',
                                           headers={**headers,'Range':f'bytes={lo}-{hi}'})
                with urllib.request.urlopen(req,timeout=60) as response:
                    assert response.status==206
                    assert int(response.headers['Content-Length'])==entry['size']
                    data=response.read(entry['size']+1)
                # Bay's range header is malformed. Accept bytes only when they
                # match the exact segment hash computed on the original host.
                assert len(data)==entry['size']
                assert hashlib.sha256(data).hexdigest()==entry['sha256']
                temporary=path.with_suffix('.partial')
                temporary.write_bytes(data);os.replace(temporary,path)
                return
            except (OSError,AssertionError) as error:
                print(json.dumps({'retry_part':index,'attempt':attempt+1,'error_type':type(error).__name__}),flush=True)
                if attempt==29:raise
                time.sleep(min(2**attempt,15))
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        tasks=[pool.submit(get,i) for i in range(len(manifest['chunks']))]
        for done,future in enumerate(cf.as_completed(tasks),1):
            future.result()
            print(json.dumps({'verified_parts':done,'total_parts':len(tasks),
                              'elapsed_s':round(time.time()-started,1)}),flush=True)
    target=root/'v7_deliverables/reproduce.verified.zip'
    with target.open('wb') as output:
        for index in range(len(manifest['chunks'])):
            output.write((parts/f'{index:04d}.part').read_bytes())
    assert target.stat().st_size==EXPECTED_SIZE and sha(target)==EXPECTED_SHA
    os.replace(target,target.with_name('reproduce.zip'))
    receipt={'folder':FOLDER,'node_id':node,'bytes':EXPECTED_SIZE,'sha256':EXPECTED_SHA,
             'download_verified':True,'public_share_created':False,
             'transfer_mode':'Each 4MiB segment verified against original-host SHA-256, then full archive hash',
             'verified_segments':len(manifest['chunks']),
             'range_endpoint_observation':'Malformed Content-Range header; segment content validated cryptographically'}
    (root/'v7_deliverables/bay_transfer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('BAY_DOWNLOAD_SHA256_VERIFIED',EXPECTED_SHA,flush=True)
    subprocess.run([sys.executable,'-u',str(HERE/'final_delivery_check.py')],check=True)


if __name__ == '__main__':
    if sys.argv[1]=='verified-parts':
        sys.stdout=open(HERE/'bay_verified_download.log','a',buffering=1)
        sys.stderr=sys.stdout
    {'launch': launch, 'upload': upload, 'download': download, 'receive': receive, 'probe': probe,
     'full-download':full_download,'verified-parts':verified_parts_download}[sys.argv[1]]()
