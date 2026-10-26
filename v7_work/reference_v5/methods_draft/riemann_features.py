"""Small, NumPy-only causal filter-bank features for 41 x 480 EEG windows.

This module deliberately does not reproduce or alter the official EEG
preprocessing. Its input is already-preprocessed, microvolt-valued windows.
FFT filtering is wholly inside each supplied four-second window.

Caching API
-----------
    C = bank_covariances(X)  # (windows, bands, channels, channels), float32
    source = estimate_source(C[train_indices], groups=train_session_ids)
    stream = CausalCovarianceFeatures(source, alignment="logcenter")
    F = stream.transform(C_one_session)  # recording order; updates after each row
    stream.reset_state()               # mandatory at a session boundary

Do not estimate ``source`` or tune parameters using validation/test windows.
Those windows may update a fresh stream only in arrival order. ``transform``
uses the same sequential implementation as ``transform_one``, so batch size
cannot change the resulting features.
"""

from __future__ import annotations

import numpy as np


DEFAULT_BANDS = ((1., 4.), (4., 8.), (8., 12.), (12., 18.),
                 (18., 26.), (26., 40.), (4., 40.))


def _symmetric(a):
    return (a + np.swapaxes(a, -1, -2)) * .5


def bank_covariances(X, bands=DEFAULT_BANDS, sfreq=120., taper=True,
                     common_average=False, batch_size=32):
    """Compute real spectral covariances without a scipy inference dependency.

    Bands are half-open ``[low, high)`` in Hz. Covariances integrate band power,
    rather than averaging spectral bins. Hann-window power loss is corrected.
    Returned matrices are positive semidefinite; apply shrinkage downstream.
    The mean of each channel is removed inside each window. A common-average
    projection is optional and requires nonzero downstream shrinkage.
    """
    X = np.asarray(X)
    single = X.ndim == 2
    if single:
        X = X[None]
    if X.ndim != 3:
        raise ValueError("Expected (B, channels, time) or (channels, time)")
    if not np.isfinite(X).all():
        raise ValueError("EEG windows contain nonfinite values")
    n, channels, n_times = X.shape
    if n_times < 4:
        raise ValueError("Too few time samples")
    bands = np.asarray(bands, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[1] != 2:
        raise ValueError("bands must have shape (n_bands, 2)")
    freq = np.fft.rfftfreq(n_times, 1. / sfreq)
    masks = [(freq >= lo) & (freq < hi) for lo, hi in bands]
    if any(not m.any() for m in masks):
        raise ValueError("A frequency band has no FFT bins")
    window = np.hanning(n_times) if taper else np.ones(n_times)
    one_sided_weight = np.full(len(freq), 2., dtype=np.float64)
    one_sided_weight[0] = 1.
    if n_times % 2 == 0:
        one_sided_weight[-1] = 1.
    scale = np.sqrt(one_sided_weight / (n_times * np.sum(window ** 2)))
    out = np.empty((n, len(bands), channels, channels), dtype=np.float32)
    for start in range(0, n, batch_size):
        x = X[start:start + batch_size].astype(np.float64, copy=True)
        x -= x.mean(axis=-1, keepdims=True)
        if common_average:
            x -= x.mean(axis=-2, keepdims=True)
        spec = np.fft.rfft(x * window, axis=-1) * scale
        for band, mask in enumerate(masks):
            z = spec[..., mask]
            r, i = z.real, z.imag
            out[start:start + len(x), band] = (
                r @ np.swapaxes(r, -1, -2) +
                i @ np.swapaxes(i, -1, -2))
    return out[0] if single else out


def shrink_covariances(cov, shrinkage=.05, floor=1e-10):
    """Trace-relative diagonal shrinkage; accepts arbitrary leading axes."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one")
    cov = _symmetric(np.asarray(cov, dtype=np.float64))
    channels = cov.shape[-1]
    scale = np.maximum(np.trace(cov, axis1=-2, axis2=-1) / channels, floor)
    return ((1. - shrinkage) * cov +
            (shrinkage * scale + floor)[..., None, None] * np.eye(channels))


def spd_power(cov, power, floor=1e-8):
    """Stable symmetric matrix powers with a trace-relative eigenvalue floor."""
    cov = _symmetric(np.asarray(cov, dtype=np.float64))
    vals, vecs = np.linalg.eigh(cov)
    minimum = np.maximum(np.mean(vals, axis=-1, keepdims=True) * floor, 1e-12)
    vals = np.maximum(vals, minimum)
    transformed = np.log(vals) if power == "log" else vals ** power
    return (vecs * transformed[..., None, :]) @ np.swapaxes(vecs, -1, -2)


def sym_to_vec(matrix):
    """Isometric upper-triangle vectorization; band-major flattened output."""
    matrix = np.asarray(matrix)
    rows, cols = np.triu_indices(matrix.shape[-1])
    weights = np.where(rows == cols, 1., np.sqrt(2.))
    vector = matrix[..., rows, cols] * weights
    return vector.reshape(*matrix.shape[:-3], -1)


def static_features(cov, kind="logcov", shrinkage=.05):
    """Return log-power or matrix-log features for cached bank covariances.

    ``relativepower`` subtracts the within-window channel mean log-power from
    every band-channel feature. It retains between-band differences.
    ``logcorrelation`` removes channel scales independently in each band and
    therefore discards absolute band-power information by design.
    """
    c = shrink_covariances(cov, shrinkage)
    if kind in ("logpower", "relativepower"):
        power = np.log(np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-12))
        if kind == "relativepower":
            power = power - np.mean(power, axis=(-2, -1), keepdims=True)
        return power.reshape(*c.shape[:-3], -1)
    if kind == "logcorrelation":
        d = np.sqrt(np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-12))
        c = c / (d[..., :, None] * d[..., None, :])
    elif kind != "logcov":
        raise ValueError("Unknown kind: " + str(kind))
    return sym_to_vec(spd_power(c, "log"))


def estimate_source(cov, groups=None, shrinkage=.05):
    """Estimate equal-session source priors using training windows only.

    Arithmetic covariance and log-covariance means are both retained so the
    same cache can compare Euclidean whitening and log-Euclidean centering.
    Group balancing prevents long sessions from dominating the source prior.
    ``groups`` must uniquely identify sessions, e.g. participant/session keys.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 4 or len(c) == 0:
        raise ValueError("Expected nonempty (windows, bands, channels, channels)")
    logs = spd_power(shrink_covariances(c, shrinkage), "log")
    logpowers = np.log(np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-12))
    if groups is None:
        weights = np.full(len(c), 1. / len(c))
    else:
        groups = np.asarray(groups)
        if len(groups) != len(c):
            raise ValueError("One group id is required per training window")
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = 1. / (len(counts) * counts[inverse])
    return {
        "cov": np.einsum("n,nbij->bij", weights, c),
        "logcov": np.einsum("n,nbij->bij", weights, logs),
        "logpower": np.einsum("n,nbc->bc", weights, logpowers),
        "shrinkage": float(shrinkage),
    }


class CausalCovarianceFeatures:
    """Source-anchored online features whose entire state is session-local.

    ``strength`` interpolates between a fixed source reference (0) and the
    adapted session reference (1). ``prior_count`` controls startup speed.
    For every already-seen window the state is updated with
    ``rate=max(ema_rate, 1/(prior_count + n_seen + 1))``. Set ``ema_rate=0``
    for an exact cumulative source-pseudocount mean. No current-window data
    enters its own alignment reference.

    Covariance-state update norms are trace-clipped relative to the previous
    reference to limit isolated artifacts, without altering the current
    window's features. Log-state update differences are clipped in their
    eigenvalues (default max factor ten) for the same reason. Disabling this
    guard uses ``update_clip=None``.
    """

    def __init__(self, source, kind="logcov", alignment="logcenter",
                 strength=1., prior_count=16., ema_rate=.02,
                 shrinkage=None, band_indices=None, update_clip=10.):
        self.source = {k: np.asarray(v).copy() if k != "shrinkage" else float(v)
                       for k, v in source.items()}
        self.kind = kind
        self.alignment = alignment
        self.strength = float(strength)
        self.prior_count = float(prior_count)
        self.ema_rate = float(ema_rate)
        self.shrinkage = (float(source["shrinkage"]) if shrinkage is None
                          else float(shrinkage))
        self.band_indices = (None if band_indices is None
                             else np.asarray(band_indices, dtype=int))
        self.update_clip = update_clip
        if not 0 <= self.strength <= 1 or not 0 <= self.ema_rate <= 1:
            raise ValueError("strength and ema_rate must be in [0, 1]")
        if self.prior_count < 1:
            raise ValueError("prior_count must be >= 1")
        if alignment not in ("none", "logcenter", "euclidean"):
            raise ValueError("Unknown alignment: " + str(alignment))
        if kind not in ("logcov", "logpower", "relativepower", "logcorrelation"):
            raise ValueError("Unknown feature kind: " + str(kind))
        self.reset_state()

    def reset_state(self):
        self.running_cov = self.source["cov"].copy()
        self.running_logcov = self.source["logcov"].copy()
        self.running_logpower = self.source["logpower"].copy()
        self.n_seen = 0

    def _mix(self, source, state):
        return (1. - self.strength) * source + self.strength * state

    def _update(self, c, logs, logpower):
        rate = min(1., max(self.ema_rate, 1. / (self.prior_count + self.n_seen + 1.)))
        target_cov = c
        delta_log = logs - self.running_logcov
        delta_power = logpower - self.running_logpower
        if self.update_clip is not None:
            limit = float(self.update_clip)
            if limit <= 1.:
                raise ValueError("update_clip must be > 1 or None")
            previous = np.maximum(np.trace(self.running_cov, axis1=-2, axis2=-1), 1e-12)
            current = np.maximum(np.trace(c, axis1=-2, axis2=-1), 1e-12)
            ratio = current / previous
            target_cov = c * (np.clip(ratio, 1. / limit, limit) / ratio)[:, None, None]
            eigen, vectors = np.linalg.eigh(_symmetric(delta_log))
            eigen = np.clip(eigen, -np.log(limit), np.log(limit))
            delta_log = (vectors * eigen[:, None, :]) @ np.swapaxes(vectors, -1, -2)
            delta_power = np.clip(delta_power, -np.log(limit), np.log(limit))
        self.running_cov += rate * (target_cov - self.running_cov)
        self.running_logcov += rate * delta_log
        self.running_logpower += rate * delta_power
        self.n_seen += 1

    def transform_one(self, cov, update=True):
        c = np.asarray(cov, dtype=np.float64)
        if c.shape != self.source["cov"].shape:
            raise ValueError("Covariance shape differs from source prior")
        raw_power = np.log(np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-12))
        shrunk = shrink_covariances(c, self.shrinkage)
        logs = spd_power(shrunk, "log")
        selected = slice(None) if self.band_indices is None else self.band_indices
        if self.alignment == "euclidean":
            reference = self._mix(self.source["cov"], self.running_cov)
            whitener = spd_power(shrink_covariances(reference, self.shrinkage), -.5)
            aligned = whitener @ shrunk @ np.swapaxes(whitener, -1, -2)
            feature = static_features(aligned[selected], self.kind, shrinkage=0.)
        elif self.alignment == "logcenter":
            if self.kind in ("logpower", "relativepower"):
                feature = raw_power - self._mix(self.source["logpower"], self.running_logpower)
                if self.kind == "relativepower":
                    feature -= feature.mean()
                feature = feature[selected].reshape(-1)
            elif self.kind == "logcorrelation":
                # Correlation is a distinct window-local view; center its
                # log features explicitly downstream if that is desired.
                feature = static_features(c[selected], self.kind, self.shrinkage)
            else:
                aligned_log = logs - self._mix(self.source["logcov"], self.running_logcov)
                feature = sym_to_vec(aligned_log[selected])
        else:
            feature = static_features(c[selected], self.kind, self.shrinkage)
        if update:
            self._update(c, logs, raw_power)
        return feature.astype(np.float32)

    def transform(self, covariances, update=True):
        c = np.asarray(covariances)
        if c.ndim == 3:
            return self.transform_one(c, update=update)
        if c.ndim != 4:
            raise ValueError("Expected one covariance bank or a sequence")
        if not len(c):
            bands = len(self.source["cov"]) if self.band_indices is None else len(self.band_indices)
            ch = self.source["cov"].shape[-1]
            dim = bands * (ch if "power" in self.kind else ch * (ch + 1) // 2)
            return np.empty((0, dim), dtype=np.float32)
        return np.stack([self.transform_one(x, update=update) for x in c])


def transform_sessions(covariances, groups, source, **kwargs):
    """Simulate separate causal streams; retain the original row ordering."""
    c = np.asarray(covariances)
    groups = np.asarray(groups)
    if len(c) != len(groups):
        raise ValueError("One session key required per window")
    result = None
    for key in np.unique(groups):
        indices = np.flatnonzero(groups == key)
        transformer = CausalCovarianceFeatures(source, **kwargs)
        features = transformer.transform(c[indices])
        if result is None:
            result = np.empty((len(c), features.shape[1]), dtype=np.float32)
        result[indices] = features
    if result is None:
        raise ValueError("No sessions supplied")
    return result


class CausalVectorCenter:
    """Fast cached log-feature centering, suitable for hyperparameter sweeps.

    This is exactly log-Euclidean centering when the input vectors are
    ``static_features(C, 'logcov')`` and clipping is disabled. A separate
    instance is required per stream. ``source_mean`` is training-derived.
    Optional ``update_scale`` clips coordinate-wise update residuals to that
    nonnegative scale; it should also be computed from training data only.
    ``strength=0`` is a useful fixed-reference control.
    """

    def __init__(self, source_mean, strength=1., prior_count=16., ema_rate=.02,
                 update_scale=None):
        self.source_mean = np.asarray(source_mean, dtype=np.float64).copy()
        self.strength = float(strength)
        self.prior_count = float(prior_count)
        self.ema_rate = float(ema_rate)
        self.update_scale = (None if update_scale is None else
                             np.asarray(update_scale, dtype=np.float64).copy())
        if not 0 <= self.strength <= 1 or not 0 <= self.ema_rate <= 1:
            raise ValueError("strength and ema_rate must lie in [0, 1]")
        if self.prior_count < 1:
            raise ValueError("prior_count must be >= 1")
        if self.update_scale is not None and np.any(self.update_scale < 0):
            raise ValueError("update_scale must be nonnegative")
        self.reset_state()

    def reset_state(self):
        self.running_mean = self.source_mean.copy()
        self.n_seen = 0

    def transform_one(self, feature, update=True):
        feature = np.asarray(feature, dtype=np.float64)
        if feature.shape != self.source_mean.shape:
            raise ValueError("Feature shape differs from source mean")
        reference = ((1. - self.strength) * self.source_mean +
                     self.strength * self.running_mean)
        out = feature - reference
        if update:
            rate = min(1., max(self.ema_rate, 1. / (self.prior_count + self.n_seen + 1.)))
            delta = feature - self.running_mean
            if self.update_scale is not None:
                delta = np.clip(delta, -self.update_scale, self.update_scale)
            self.running_mean += rate * delta
            self.n_seen += 1
        return out.astype(np.float32)

    def transform(self, features, update=True):
        features = np.asarray(features)
        if features.ndim == 1:
            return self.transform_one(features, update=update)
        if features.ndim != 2:
            raise ValueError("Expected one feature row or a feature matrix")
        if not len(features):
            return np.empty((0, len(self.source_mean)), dtype=np.float32)
        return np.stack([self.transform_one(f, update=update) for f in features])


def mean_by_group(features, groups):
    """Equal-session feature mean, for training-only source prior estimation."""
    features, groups = np.asarray(features), np.asarray(groups)
    return np.mean([np.mean(features[groups == g], axis=0, dtype=np.float64)
                    for g in np.unique(groups)], axis=0)


def center_feature_sessions(features, groups, source_mean, **kwargs):
    """Fast causal transform of precomputed feature rows, reset per session."""
    features, groups = np.asarray(features), np.asarray(groups)
    if len(features) != len(groups):
        raise ValueError("One session key is required per feature row")
    result = np.empty(features.shape, dtype=np.float32)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        transform = CausalVectorCenter(source_mean, **kwargs)
        result[indices] = transform.transform(features[indices])
    return result
