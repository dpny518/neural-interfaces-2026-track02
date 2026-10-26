"""NumPy deployment helpers for the fixed v5 Riemannian tangent ablation.

The containing submission embeds riemann_features.py before this file.
The tangent origin is fitted on source-training windows only. Nothing here
estimates a geometric mean from an evaluation session or changes that origin.
"""
from __future__ import annotations

import numpy as np
from riemann_features import (
    CausalCovarianceFeatures, shrink_covariances, spd_power, sym_to_vec,
)


class V5AlignedCovarianceStream(CausalCovarianceFeatures):
    """Expose precisely the covariance used by v3's Euclidean log features."""

    def transform_covariance_one(self, covariance):
        if self.alignment != "euclidean" or self.band_indices is not None:
            raise ValueError("V5 tangent branch requires full-bank Euclidean alignment")
        c = np.asarray(covariance, dtype=np.float64)
        if c.shape != self.source["cov"].shape:
            raise ValueError("Covariance shape differs from training source")
        raw_power = np.log(np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-12))
        shrunk = shrink_covariances(c, self.shrinkage)
        logs = spd_power(shrunk, "log")
        reference = self._mix(self.source["cov"], self.running_cov)
        whitener = spd_power(shrink_covariances(reference, self.shrinkage), -.5)
        aligned = whitener @ shrunk @ np.swapaxes(whitener, -1, -2)
        # Match static_features(..., shrinkage=0) in the original v3 branch.
        aligned = shrink_covariances(aligned, 0.)
        self._update(c, logs, raw_power)
        return aligned


def v5_tangent_features(aligned_covariances, tangent_inverse_sqrt):
    c = np.asarray(aligned_covariances, dtype=np.float64)
    white = np.asarray(tangent_inverse_sqrt, dtype=np.float64)
    tangent = spd_power(white @ c @ np.swapaxes(white, -1, -2), "log")
    return sym_to_vec(tangent).astype(np.float32)


def _v5_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


class V5TangentMember:
    """A deepcopy-safe causal member with a frozen source-training tangent map."""

    def __init__(self, data):
        self.views = data["views"]
        source = {key: _v5_array(data["source_" + key])
                  for key in ("cov", "logcov", "logpower")}
        source["shrinkage"] = float(data["shrinkage"])
        self.transformer = V5AlignedCovarianceStream(source, **data["center_kwargs"])
        self.tangent_inverse_sqrt = _v5_array(data["tangent_inverse_sqrt"])
        self.scaler_mean = _v5_array(data["scaler_mean"])
        self.scaler_scale = _v5_array(data["scaler_scale"])
        self.coef = _v5_array(data["coef"])
        self.intercept = _v5_array(data["intercept"])
        self.temperature = float(data["temperature"])

    def reset_state(self):
        self.transformer.reset_state()

    def predict_one(self, covariance, feature_cache=None):
        aligned = self.transformer.transform_covariance_one(covariance)
        feature = v5_tangent_features(aligned, self.tangent_inverse_sqrt)
        logits = ((feature - self.scaler_mean) / self.scaler_scale) @ self.coef + self.intercept
        logits = np.asarray(logits, dtype=np.float64) / self.temperature
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()
