"""Fetch hash-pinned official evaluator source, not a changing main branch."""
import hashlib
import json
from pathlib import Path
import urllib.request
import ssl

ROOT=Path(__file__).resolve().parent


def fetch():
    spec=json.loads((ROOT/'reference_v5/v5_work/reference_sources.json').read_text())
    dest=ROOT/'official'
    records={}
    for name,expected in {**spec['files'], 'benchmark_utils/data.py':None}.items():
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raw=p.read_bytes()
        else:
            url=f"https://raw.githubusercontent.com/neural-interfaces26/2026-competition/{spec['competition_commit']}/{name}"
            try:
                import certifi
                context=ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context=ssl.create_default_context()
            with urllib.request.urlopen(url,timeout=90,context=context) as r:raw=r.read()
        actual=hashlib.sha256(raw).hexdigest()
        if expected is not None:assert actual==expected,name
        p.write_bytes(raw);records[name]=actual
    (dest/'benchmark_utils/__init__.py').touch(exist_ok=True)
    (dest/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    return dest


if __name__=='__main__':print(fetch())
