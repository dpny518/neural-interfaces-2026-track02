"""Reconstructed multiscale architecture; NumPy/PyTorch-only inference."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from riemann_features import bank_covariances, CausalCovarianceFeatures
from neural_cov import shrink, invsqrt
from riemann_v5_runtime import V5AlignedCovarianceStream


def softmax(z, temperature=1.):
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def arr(x):
    return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x).copy()


def normalized_covariance(c, broadband_indices):
    c = np.asarray(c, dtype=np.float64)
    scale = np.trace(c[..., broadband_indices, :, :], axis1=-2, axis2=-1).mean(axis=-1) / c.shape[-1]
    return c / np.maximum(scale, 1e-10)[..., None, None, None]


class MultiscaleAligner:
    def __init__(self, source, broadband_indices, prior_count=16., ema_rate=.02,
                 shrinkage=.05, update_clip=10., normalization='broadband', strength=1.):
        assert normalization == 'broadband' and strength == 1.
        self.source = arr(source).astype(np.float64)
        self.broadband_indices = list(broadband_indices)
        self.prior_count, self.ema_rate = prior_count, ema_rate
        self.shrinkage, self.update_clip = shrinkage, update_clip
        self.reset_state()

    def reset_state(self):
        self.running_cov = self.source.copy()
        self.n_seen = 0

    def transform_one(self, cov):
        current = normalized_covariance(cov, self.broadband_indices)
        white = invsqrt(shrink(self.running_cov, self.shrinkage))
        result = white @ shrink(current, self.shrinkage) @ np.swapaxes(white, -1, -2)
        rate = min(1., max(self.ema_rate, 1. / (self.prior_count + self.n_seen + 1.)))
        old = np.maximum(np.trace(self.running_cov, axis1=-2, axis2=-1), 1e-12)
        new = np.maximum(np.trace(current, axis1=-2, axis2=-1), 1e-12)
        ratio = new / old
        target = current * (np.clip(ratio, 1/self.update_clip, self.update_clip) / ratio)[:, None, None]
        self.running_cov += rate * (target - self.running_cov)
        self.n_seen += 1
        return result.astype(np.float32)


class MultiscaleNet(nn.Module):
    def __init__(self, n_bands=7, n_views=3, n_chans=41, n_filters=16, hidden=128, dropout=.5):
        super().__init__()
        self.n_bands, self.n_views = n_bands, n_views
        self.spatial_weight = nn.Parameter(F.normalize(torch.randn(n_bands, n_filters, n_chans), dim=-1))
        dim = n_views * n_bands * (n_filters + n_chans)
        self.normalizer = nn.BatchNorm1d(dim, momentum=.05)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, 3)
        self.hidden = nn.Linear(dim, hidden)
        self.residual = nn.Linear(hidden, 3)

    def features(self, cov):
        n = len(cov)
        c = cov.reshape(n, self.n_views, self.n_bands, 41, 41)
        w = F.normalize(self.spatial_weight, dim=-1)
        projected = torch.matmul(w[None, None], c)
        var = (projected * w[None, None]).sum(-1).clamp_min(1e-6).log()
        diag = c.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6).log()
        return torch.cat([var, diag], dim=-1).flatten(1)

    def forward(self, cov):
        z = self.dropout(self.normalizer(self.features(cov)))
        return self.linear(z) + self.residual(self.dropout(F.gelu(self.hidden(z))))


class NeuralMember:
    def __init__(self, data, device='cpu'):
        self.net = MultiscaleNet(**data['config']).to(device)
        self.net.load_state_dict(data['state_dict'])
        self.net.eval()
        self.device = device
        self.aligner = MultiscaleAligner(data['source'], data['broadband_indices'], **data['alignment'])

    def reset_state(self):
        self.aligner.reset_state()
        self.net.eval()

    def predict_one(self, cov):
        x = self.aligner.transform_one(cov)
        with torch.inference_mode():
            z = self.net(torch.as_tensor(x[None], device=self.device))[0].cpu().numpy()
        return softmax(z)


class ClassicalMember:
    def __init__(self, data):
        self.kind = data['kind']
        source = {k: arr(data['source'][k]) for k in ['cov', 'logcov', 'logpower']}
        source['shrinkage'] = .05
        kwargs = dict(kind='logcov', alignment='euclidean', strength=1., prior_count=8.,
                      ema_rate=.02, shrinkage=.05, update_clip=10.)
        self.stream = (CausalCovarianceFeatures(source, **kwargs) if self.kind == 'covariance'
                       else V5AlignedCovarianceStream(source, **kwargs))
        self.mean, self.scale = arr(data['mean']), arr(data['scale'])
        self.coef, self.bias = arr(data['coef']), arr(data['bias'])
        self.temperature = float(data['temperature'])
        if self.kind == 'rbf':
            self.centers = arr(data['centers'])
            self.gamma = float(data['gamma'])

    def reset_state(self):
        self.stream.reset_state()

    def predict_one(self, cov):
        if self.kind == 'covariance':
            f = self.stream.transform_one(cov)
        else:
            aligned = self.stream.transform_covariance_one(cov)
            f = np.log(np.maximum(np.diagonal(aligned, axis1=-2, axis2=-1), 1e-12)).flatten()
        f = np.asarray(f, dtype=np.float32)
        z = (f - self.mean) / self.scale
        if self.kind == 'rbf':
            z = np.exp(-self.gamma * np.sum((self.centers - z) ** 2, axis=1))
        return softmax(z @ self.coef + self.bias, self.temperature)


class Decoder:
    def __init__(self, checkpoint, meta):
        self.device = meta.get('device', 'cpu')
        if int(meta.get('n_classes', 3)) != 3:
            raise ValueError('Expected three classes')
        for key, value in [('n_chans',41),('n_times',480),('sfreq',120)]:
            if key in meta and float(meta[key]) != value:
                raise ValueError('Unexpected '+key)
        self.checkpoint = checkpoint
        self.bands, self.views = checkpoint['bands'], checkpoint['views']
        self.channel_indices = None
        if 'ch_names' in meta:
            names = list(meta['ch_names']); expected = checkpoint['ch_names']
            if len(names) != 41 or set(names) != set(expected):
                raise ValueError('Unexpected channel names')
            self.channel_indices = [names.index(s) for s in expected]
        self.neural = [NeuralMember(d, self.device) for d in checkpoint['neural']]
        self.classical = [ClassicalMember(d) for d in checkpoint['classical']]
        self.weights = arr(checkpoint['weights']).astype(np.float64)
        assert len(self.weights) == len(self.neural)+len(self.classical)
        np.testing.assert_allclose(self.weights.sum(), 1.)
        self.reset_state()

    def reset_state(self):
        for member in self.neural + self.classical:
            member.reset_state()
        self.n_seen = 0

    def predict_proba(self, X):
        x = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        if x.ndim != 3 or x.shape[1:] != (41,480) or not np.isfinite(x).all():
            raise ValueError('Expected finite (B,41,480) EEG')
        x = x.astype(np.float32, copy=False)
        if self.channel_indices is not None:
            x = x[:, self.channel_indices]
        result = np.empty((len(x), 3), dtype=np.float64)
        for i, window in enumerate(x):
            cov = np.concatenate([bank_covariances(window[:,lo:hi], bands=self.bands) for lo,hi in self.views])
            if self.bands == [[1,4],[4,8],[8,12],[12,18],[18,26],[26,40],[4,40]]:
                classic_cov = cov[:14]
            else:
                classic_cov = np.concatenate([bank_covariances(window[:,:240]),bank_covariances(window[:,240:])])
            probabilities = [m.predict_one(cov) for m in self.neural]
            probabilities += [m.predict_one(classic_cov) for m in self.classical]
            result[i] = self.weights @ np.stack(probabilities)
            self.n_seen += 1
        return result

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:
    class CompetSolver:
        pass


class Solver(CompetSolver):
    name = 'Reconstructed multiscale covariance ensemble v7'
    requirements = []

    def load_model(self, meta):
        data = torch.load(Path(meta['submission_dir'])/'weights.pt', map_location='cpu', weights_only=True)
        return Decoder(data, meta)
