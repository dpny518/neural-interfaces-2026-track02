"""Test interrupted/resumed downloads without contacting the internet."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch
import prepare_data as module


def run():
    bodies={'resume.bin':b'resumable-data-'*100,'ignore-range.bin':b'full-response-'*80,
            'existing.bin':b'already-verified'}
    calls=[];lock=threading.Lock();first=True
    class Response:
        def __init__(self,body,status=200,offset=0,interrupt=False):
            self.body=io.BytesIO(body);self.status=status;self.interrupt=interrupt;self.reads=0
            self.headers={'Content-Range':f'bytes {offset}-{offset+len(body)-1}/{offset+len(body)}'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):
            self.reads+=1
            if self.interrupt:
                if self.reads==1:return self.body.read(37)
                raise TimeoutError('synthetic interrupted transfer')
            return self.body.read(n)
    def urlopen(request,timeout):
        nonlocal first
        name=request.full_url.rsplit('/',1)[1];headers=dict(request.header_items())
        offset=int(headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
        with lock:
            calls.append((name,offset))
            interrupted=name=='resume.bin' and first
            if interrupted:first=False
        if name=='ignore-range.bin':return Response(bodies[name])
        return Response(bodies[name][offset:],206 if offset else 200,offset,interrupted)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);manifest=root/'reference_v5/data/nm000290/v1.0.0/download_manifest.json'
        manifest.parent.mkdir(parents=True)
        entries=[{'path':k,'size':len(v),'checksum_algorithm':'sha256',
                  'checksum':hashlib.sha256(v).hexdigest(),'url':'https://example.invalid/'+k} for k,v in bodies.items()]
        manifest.write_text(json.dumps(entries))
        dest=root/'data/nm000290/v1.0.0';dest.mkdir(parents=True)
        (dest/'ignore-range.bin.partial').write_bytes(b'incorrect-old-prefix')
        (dest/'existing.bin').write_bytes(bodies['existing.bin'])
        with patch.object(module,'ROOT',root),patch.object(module.urllib.request,'urlopen',urlopen),patch.object(module.time,'sleep',lambda _:None):
            module.download()
        for name,body in bodies.items():assert (dest/name).read_bytes()==body
        assert ('resume.bin',37) in calls
        assert not any(n=='existing.bin' for n,_ in calls)
        assert json.loads((dest/'download_verification.json').read_text())['complete']
    print('Interrupted-transfer resume, ignored-Range restart and existing-checksum reuse passed.')


if __name__=='__main__':run()
