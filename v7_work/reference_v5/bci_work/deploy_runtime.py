"""Runtime appended to standalone submission.py by build_submission.py.

Requires NumPy and PyTorch only. Supporting numeric functions and network
definitions are embedded above this source, never imported from the workspace.
"""
from pathlib import Path

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:
    class CompetSolver:
        """Local fallback; the official worker supplies the real base class."""
        pass


def _as_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _softmax(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


class _LinearMember:
    def __init__(self, data):
        self.data = data
        self.family = data['family']
        self.views = data['views']
        self.feature_indices = (None if data.get('feature_indices') is None
                                else _as_array(data['feature_indices']).astype(np.int64))
        self.scaler_mean = _as_array(data['scaler_mean'])
        self.scaler_scale = _as_array(data['scaler_scale'])
        self.coef = _as_array(data['coef'])
        self.intercept = _as_array(data['intercept'])
        self.temperature = float(data.get('temperature', 1.))
        if self.family == 'logcenter':
            self.transformer = CausalVectorCenter(_as_array(data['source_mean']), **data['center_kwargs'])
        elif self.family == 'euclidean':
            source = {key: _as_array(data['source_'+key]) for key in ('cov','logcov','logpower')}
            source['shrinkage'] = float(data['shrinkage'])
            self.transformer = CausalCovarianceFeatures(source, **data['center_kwargs'])
        else:
            raise ValueError('Unknown numeric member family')

    def reset_state(self):
        self.transformer.reset_state()

    def predict_one(self, cov, feature_cache):
        if self.family == 'euclidean':
            feat = self.transformer.transform_one(cov)
            if self.feature_indices is not None:
                feat = feat[self.feature_indices]
        else:
            key = (tuple((tuple(v['crop']),v.get('car',False),v.get('zscore',False)) for v in self.views),
                   self.data.get('raw_kind','logcov'), float(self.data['shrinkage']))
            if key not in feature_cache:
                feature_cache[key] = static_features(cov, self.data.get('raw_kind','logcov'), self.data['shrinkage']).astype(np.float32)
            feat = feature_cache[key]
            if self.feature_indices is not None:
                feat = feat[self.feature_indices]
            feat = self.transformer.transform_one(feat)
        feat = (feat-self.scaler_mean) / self.scaler_scale
        logits = feat @ self.coef + self.intercept
        return _softmax(logits, self.temperature)


class _NeuralCovMember:
    def __init__(self, data, device):
        self.data = data
        self.views = data['views']
        self.device = device
        ckpt = data['checkpoint']
        self.net = SpatialLogVarianceNet(**ckpt['model_config'])
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.to(device).eval()
        self.aligner = CausalCovarianceAligner(_as_array(ckpt['source']), **ckpt['alignment_config'])
        self.temperature = float(data.get('temperature', 1.))

    def reset_state(self):
        self.aligner.reset_state()
        self.net.eval()

    def predict_one(self, cov, feature_cache):
        aligned = self.aligner.transform_one(cov)
        x = torch.as_tensor(aligned[None], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits = self.net(x)[0].detach().cpu().numpy()
        return _softmax(logits, self.temperature)


class _RawMember:
    def __init__(self, data, device):
        ckpt = data['checkpoint']
        self.views = []
        self.device = device
        self.temperature = float(data.get('temperature', 1.))
        self.net = make_raw_model(ckpt['model_type'], ckpt['model_config'])
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.to(device).eval()

    def reset_state(self):
        self.net.eval()

    def predict_one(self, x, feature_cache):
        with torch.inference_mode():
            logits = self.net(torch.as_tensor(x[None],dtype=torch.float32,device=self.device))[0].detach().cpu().numpy()
        return _softmax(logits, self.temperature)


class CausalBCIDecoder:
    """All online state belongs to this instance and is reset per session.

    Processing an arbitrary batch is exactly a sequence of single-window
    calls. No labels, recording IDs, session IDs, or future windows are used.
    Arrays and torch modules are ordinary deepcopy-safe in-memory objects.
    """
    def __init__(self, checkpoint, meta):
        self.device = torch.device(meta.get('device','cpu'))
        self.n_chans = int(meta.get('n_chans',41))
        self.n_times = int(meta.get('n_times',480))
        if (self.n_chans,self.n_times,int(meta.get('n_classes',3))) != (41,480,3):
            raise ValueError('This checkpoint is trained for 41 channels, 480 samples and 3 classes')
        if 'sfreq' in meta and not np.isclose(float(meta['sfreq']),120.):
            raise ValueError('This checkpoint requires 120 Hz preprocessed input')
        expected = checkpoint['ch_names']
        supplied = meta.get('ch_names')
        self.channel_indices = None
        if supplied is not None:
            supplied = list(supplied)
            if len(supplied) != len(expected) or set(supplied) != set(expected):
                raise ValueError('Input EEG channel names do not match the trained montage')
            self.channel_indices = np.array([supplied.index(name) for name in expected],dtype=np.int64)
        self.members=[]
        self.member_types=[]
        for record in checkpoint['members']:
            kind=record['type']
            if kind=='linear':
                member=_LinearMember(record)
            elif kind=='neural_cov':
                member=_NeuralCovMember(record,self.device)
            elif kind=='raw':
                member=_RawMember(record,self.device)
            else:
                raise ValueError('Unsupported checkpoint member')
            self.members.append(member)
            self.member_types.append(kind)
        self.mix_weights=np.array([float(m['weight']) for m in checkpoint['members']],dtype=np.float64)
        if not len(self.members) or not np.isfinite(self.mix_weights).all() or np.any(self.mix_weights<0) or self.mix_weights.sum()<=0:
            raise ValueError('Invalid ensemble weights')
        self.mix_weights/=self.mix_weights.sum()
        self.reset_state()

    def reset_state(self):
        for member in self.members:
            member.reset_state()
        self.n_seen=0

    def predict_proba(self, X):
        if isinstance(X,torch.Tensor):
            x=X.detach().to(device='cpu',dtype=torch.float32).numpy()
        else:
            x=np.asarray(X,dtype=np.float32)
        if x.ndim!=3 or tuple(x.shape[1:])!=(self.n_chans,self.n_times):
            raise ValueError('Expected input shape (B,41,480)')
        if not np.isfinite(x).all():
            raise ValueError('EEG input must be finite')
        if self.channel_indices is not None:
            x=x[:,self.channel_indices,:]
        answer=np.empty((len(x),3),dtype=np.float64)
        for i,window in enumerate(x):
            cov_cache={}
            feature_cache={}
            probs=[]
            for kind,member in zip(self.member_types,self.members):
                if kind=='raw':
                    probs.append(member.predict_one(window,feature_cache))
                    continue
                banks=[]
                for view in member.views:
                    key=(tuple(view['crop']),view.get('car',False),view.get('zscore',False))
                    if key not in cov_cache:
                        start,stop=view['crop']
                        segment=window[:,start:stop]
                        if view.get('zscore',False):
                            segment=segment-segment.mean(axis=-1,keepdims=True)
                            segment=segment/np.maximum(segment.std(axis=-1,keepdims=True),1e-6)
                        cov_cache[key]=bank_covariances(segment,common_average=view.get('car',False))
                    banks.append(cov_cache[key])
                cov=banks[0] if len(banks)==1 else np.concatenate(banks,axis=0)
                probs.append(member.predict_one(cov,feature_cache))
            answer[i]=self.mix_weights@np.stack(probs)
            self.n_seen+=1
        return answer

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


class Solver(CompetSolver):
    name='Causal temporal filter-bank ensemble'
    requirements=[]
    def load_model(self, meta):
        path=Path(meta['submission_dir'])/'weights.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('schema_version')!=1:
            raise ValueError('Unsupported weights format')
        return CausalBCIDecoder(checkpoint,meta)
