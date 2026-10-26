"""Synthetic-only numerical audit; no public EEG, labels or scores are read."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bci_work"), str(ROOT / "methods_draft")]
from v5_work.classical_v5 import SPEC, riemannian_mean
from v5_work.riemann_v5_runtime import V5AlignedCovarianceStream, v5_tangent_features
from riemann_features import CausalCovarianceFeatures, estimate_source, spd_power, sym_to_vec


def main():
    rng = np.random.default_rng(20261023)
    x = rng.normal(size=(9, 3, 5, 5))
    covariances = x @ np.swapaxes(x, -1, -2) + np.eye(5) * .4
    source = estimate_source(covariances, np.repeat(np.arange(3), 3))
    old = CausalCovarianceFeatures(source, **SPEC["center_kwargs"])
    new = V5AlignedCovarianceStream(source, **SPEC["center_kwargs"])
    for covariance in covariances:
        expected = old.transform_one(covariance)
        aligned = new.transform_covariance_one(covariance)
        actual = sym_to_vec(spd_power(aligned, "log")).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        identity = np.broadcast_to(np.eye(5), (3, 5, 5))
        np.testing.assert_array_equal(v5_tangent_features(aligned, identity), expected)
        for key in ("running_cov", "running_logcov", "running_logpower"):
            np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
    assert old.n_seen == new.n_seen == 9
    values = np.array([[1., 2., 3.], [4., 3., 2.], [7., 5., 1.]])
    matrices = np.array([np.diag(value) for value in values])
    weights = np.array([.2, .3, .5])
    mean, diagnostic = riemannian_mean(matrices, weights)
    expected = np.diag(np.exp(weights @ np.log(values)))
    np.testing.assert_allclose(mean, expected, rtol=1e-9, atol=1e-9)
    transform = np.array([[1.2, .2, .1], [.1, 1.4, .3], [.2, .1, .9]])
    changed = np.array([transform @ matrix @ transform.T for matrix in matrices])
    second, second_diagnostic = riemannian_mean(changed, weights)
    np.testing.assert_allclose(second, transform @ expected @ transform.T, rtol=1e-8, atol=1e-8)
    result = dict(all_passed=True, synthetic_only=True, v3_features_bitwise_identical=True,
                  session_state_bitwise_identical=True, identity_tangent_features_match=True,
                  commuting_geometric_mean=True, affine_mean_equivariance=True,
                  mean_solver_diagnostics=[diagnostic, second_diagnostic])
    destination = ROOT / "v5_work/classical_math_check.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "mean_solver_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
