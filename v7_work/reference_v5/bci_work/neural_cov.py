"""Compact covariance-input spatial-filter neural decoder.

This is an FBCNet-style, learned spatial log-variance model. It operates on
window-local filter-bank covariances rather than replaying temporal convolutions
during training. ``CausalCovarianceAligner`` references only already-seen
windows and resets all target-domain state at each participant/session.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize_covariances(cov, mode="broadband"):
    c = np.asarray(cov, dtype=np.float64)
    if mode == "broadband":
        # Every block of seven entries ends with 4--40 Hz. For temporal-bank
        # inputs average their broadbands, preserving early/late contrasts.
        # Normalize global gain while retaining relative band-power changes.
        broad = c[..., np.arange(6, c.shape[-3], 7), :, :]
        scale = np.trace(broad, axis1=-2, axis2=-1).mean(axis=-1) / c.shape[-1]
        c = c / np.maximum(scale, 1e-10)[..., None, None, None]
    elif mode == "band":
        scale = np.trace(c, axis1=-2, axis2=-1) / c.shape[-1]
        c = c / np.maximum(scale, 1e-10)[..., None, None]
    elif mode != "none":
        raise ValueError("Unknown normalization: " + str(mode))
    return c


def shrink(c, amount):
    c = (c + np.swapaxes(c, -1, -2)) * .5
    scale = np.maximum(np.trace(c, axis1=-2, axis2=-1) / c.shape[-1], 1e-10)
    return (1. - amount) * c + (amount * scale + 1e-10)[..., None, None] * np.eye(c.shape[-1])


def invsqrt(c):
    vals, vecs = np.linalg.eigh((c + np.swapaxes(c, -1, -2)) * .5)
    vals = np.maximum(vals, np.maximum(vals.mean(axis=-1, keepdims=True) * 1e-7, 1e-12))
    return (vecs * (vals ** -.5)[..., None, :]) @ np.swapaxes(vecs, -1, -2)


def fit_source_reference(covs, groups, normalization="broadband"):
    """Equal-session arithmetic covariance prior; pass training data only."""
    c = normalize_covariances(covs, normalization)
    groups = np.asarray(groups)
    return np.mean([c[groups == group].mean(axis=0) for group in np.unique(groups)], axis=0)


class CausalCovarianceAligner:
    def __init__(self, source, normalization="broadband", strength=1.,
                 prior_count=16., ema_rate=.02, shrinkage=.05, update_clip=10.):
        self.source = np.asarray(source, dtype=np.float64).copy()
        self.normalization = normalization
        self.strength = float(strength)
        self.prior_count = float(prior_count)
        self.ema_rate = float(ema_rate)
        self.shrinkage = float(shrinkage)
        self.update_clip = float(update_clip)
        self.fixed_whitener = invsqrt(shrink(self.source, self.shrinkage))
        self.reset_state()

    def reset_state(self):
        self.running_cov = self.source.copy()
        self.n_seen = 0

    def transform_one(self, cov):
        current = normalize_covariances(cov, self.normalization)
        if self.strength == 0.:
            whitener = self.fixed_whitener
        else:
            reference = (1. - self.strength) * self.source + self.strength * self.running_cov
            whitener = invsqrt(shrink(reference, self.shrinkage))
        aligned = whitener @ shrink(current, self.shrinkage) @ np.swapaxes(whitener, -1, -2)
        # Update after forming the current feature: no self-reference and no
        # future-window input to target-domain normalization.
        rate = min(1., max(self.ema_rate, 1. / (self.prior_count + self.n_seen + 1.)))
        old_trace = np.maximum(np.trace(self.running_cov, axis1=-2, axis2=-1), 1e-12)
        new_trace = np.maximum(np.trace(current, axis1=-2, axis2=-1), 1e-12)
        ratio = new_trace / old_trace
        target = current * (np.clip(ratio, 1. / self.update_clip, self.update_clip) / ratio)[:, None, None]
        self.running_cov += rate * (target - self.running_cov)
        self.n_seen += 1
        return aligned.astype(np.float32)

    def transform(self, covs):
        return np.stack([self.transform_one(c) for c in covs])


class SpatialLogVarianceNet(nn.Module):
    """Learn band-specific spatial filters and a regularized classifier.

    Inputs are covariances of shape [batch, bands, channels, channels]. Each
    spatial filter produces log(w C w^T). Optional channel log-power features
    expose the stable topographic view alongside learned spatial projections.
    BatchNorm uses training-only running statistics at inference.
    """
    def __init__(self, n_bands=7, n_chans=41, n_filters=16, n_classes=3,
                 dropout=.5, include_diagonal=True, hidden=0):
        super().__init__()
        self.n_bands = int(n_bands)
        self.n_chans = int(n_chans)
        self.n_filters = int(n_filters)
        self.include_diagonal = bool(include_diagonal)
        weights = torch.randn(n_bands, n_filters, n_chans)
        weights = F.normalize(weights, p=2, dim=-1)
        self.spatial_weight = nn.Parameter(weights)
        n_features = n_bands * (n_filters + (n_chans if include_diagonal else 0))
        self.normalizer = nn.BatchNorm1d(n_features, momentum=.05)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(n_features, hidden) if hidden else None
        self.classifier = nn.Linear(hidden if hidden else n_features, n_classes)
        self.config = dict(n_bands=n_bands, n_chans=n_chans, n_filters=n_filters,
                           n_classes=n_classes, dropout=dropout,
                           include_diagonal=include_diagonal, hidden=hidden)

    def features(self, covs):
        w = F.normalize(self.spatial_weight, p=2, dim=-1)
        wc = torch.matmul(w.unsqueeze(0), covs)
        variance = (wc * w.unsqueeze(0)).sum(-1).clamp_min(1e-6)
        features = [variance.log()]
        if self.include_diagonal:
            features.append(torch.diagonal(covs, dim1=-2, dim2=-1).clamp_min(1e-6).log())
        return torch.cat(features, dim=-1).flatten(1)

    def forward(self, covs):
        z = self.normalizer(self.features(covs))
        z = self.dropout(z)
        if self.hidden is not None:
            z = self.dropout(F.gelu(self.hidden(z)))
        return self.classifier(z)


class NeuralCovarianceDecoder:
    """Inference wrapper loading a selected training checkpoint dictionary.

    ``predict_proba_from_cov`` accepts the unnormalized spectral covariance
    bank. For temporal14-band checkpoints callers concatenate early/late
    two-second banks before invoking this method. All adaptation is local to
    this object and fully restored by ``reset_state``.
    """
    def __init__(self, checkpoint, device="cpu"):
        self.device = str(device)
        self.net = SpatialLogVarianceNet(**checkpoint["model_config"])
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.to(self.device).eval()
        source = checkpoint["source"]
        if torch.is_tensor(source):
            source = source.detach().cpu().numpy()
        self.aligner = CausalCovarianceAligner(source, **checkpoint["alignment_config"])

    def reset_state(self):
        self.aligner.reset_state()
        self.net.eval()

    def predict_proba_from_cov(self, covariances):
        covariances = np.asarray(covariances)
        expected = (self.net.n_bands, self.net.n_chans, self.net.n_chans)
        if covariances.ndim != 4 or tuple(covariances.shape[1:]) != expected:
            raise ValueError(f"Expected (B,{expected}) covariance array")
        if len(covariances) == 0:
            return np.empty((0, 3), dtype=np.float32)
        features = self.aligner.transform(covariances)
        with torch.inference_mode():
            inputs = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            return torch.softmax(self.net(inputs), -1).cpu().numpy()

    def predict_from_cov(self, covariances):
        return self.predict_proba_from_cov(covariances).argmax(1).astype(np.int64)

