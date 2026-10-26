from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
for sub in ['bci_work', 'methods_draft', 'v5_work']:
    sys.path.insert(0, str(ROOT / 'reference_v5' / sub))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(raw)
    os.replace(tmp, path)


def plan():
    p = ROOT / 'plan.json'
    assert sha(p) == (ROOT / 'plan.sha256').read_text().strip(), 'Plan changed'
    return json.loads(p.read_text())


def metadata():
    with np.load(ROOT / 'cache/metadata.npz', allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    assert len(m['y']) == 2655
    assert set(m['subject']) == set(plan()['public_subjects'])
    assert set(m['split']) == {'development'}
    return m


def subset(m, rows):
    return {k: v[rows] for k, v in m.items()}


def fitting_record(m, rows, excluded, stage, recipe, **extra):
    rows = np.asarray(rows, dtype=np.int64)
    actual = sorted(set(m['subject'][rows]))
    assert len(rows) > 0 and len(rows) == len(np.unique(rows))
    assert not set(actual) & set(excluded), 'Excluded subject reached fitting'
    fields = ['fitted_subjects', 'normalization_subjects', 'reference_subjects',
              'feature_scaling_subjects', 'checkpoint_selection_subjects']
    result = {k: actual for k in fields}
    result.update(stage=stage, recipe=recipe, excluded_subjects=sorted(excluded),
                  fitting_rows=rows.tolist(), fitting_rows_sha256=array_sha(rows),
                  fitting_metadata_sha256=sha(ROOT / 'cache/metadata.npz'),
                  plan_sha256=sha(ROOT / 'plan.json'),
                  source_freeze_sha256=sha(ROOT / 'implementation_freeze.json'),
                  pretraining_subjects=actual if recipe == 'masked_ssl' else [],
                  checkpoint_policy='fixed epoch/no held-out checkpoint decisions', **extra)
    return result


def save_checkpoint(path, data, provenance):
    import torch
    path = Path(path)
    assert not path.exists(), 'Refusing to overwrite fitted checkpoint: ' + str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(json.dumps(provenance))
    data = {**data, 'provenance': provenance}
    tmp = path.with_suffix('.partial')
    torch.save(data, tmp)
    os.replace(tmp, path)
    json_write(path.with_suffix('.provenance.json'), {**provenance, 'artifact_sha256': sha(path)})


def load_checkpoint(path):
    import torch
    sidecar = Path(path).with_suffix('.provenance.json')
    record = json.loads(sidecar.read_text())
    assert sha(path) == record.pop('artifact_sha256')
    assert record['plan_sha256'] == sha(ROOT / 'plan.json')
    assert record['source_freeze_sha256'] == sha(ROOT / 'implementation_freeze.json')
    assert record['fitting_metadata_sha256'] == sha(ROOT / 'cache/metadata.npz')
    data = torch.load(path, map_location='cpu', weights_only=True)
    assert data['provenance'] == record
    return data


def freeze_implementation():
    files = ['protocol.py', 'runtime.py', 'training.py', 'classical.py', 'run_cv.py',
             'prepare_data.py', 'reference_v5/extraction_manifest.json']
    records = {f: sha(ROOT / f) for f in files}
    # Check every preserved source, not merely its manifest's bytes.
    ref = json.loads((ROOT / 'reference_v5/extraction_manifest.json').read_text())
    for e in ref['files']:
        assert sha(ROOT / 'reference_v5' / e['path']) == e['sha256']
    p = ROOT / 'implementation_freeze.json'
    if p.exists():
        assert json.loads(p.read_text())['files'] == records, 'Fitting source changed'
    else:
        json_write(p, {'files': records, 'plan_sha256': sha(ROOT / 'plan.json'),
                       'created_at_utc': datetime.now(timezone.utc).isoformat()})


def cell_scores(y, prediction, m):
    keys = sorted(set(zip(m['subject'], m['session'], m['interface'])))
    scores = []
    for key in keys:
        ix = np.flatnonzero((m['subject'] == key[0]) & (m['session'] == key[1]) & (m['interface'] == key[2]))
        assert all(np.any(y[ix] == c) for c in range(3))
        scores.append(float(np.mean([np.mean(prediction[ix][y[ix] == c] == c) for c in range(3)])))
    return np.asarray(scores), keys
