"""Synthetic-only correctness checks for the fixed v5 neural recipes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "bci_work"), str(ROOT)]

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from crossvalidate_neural import fit_fixed
from neural_cov import SpatialLogVarianceNet
from v5_work.neural_training_v5 import MODEL_CONFIG, mask_covariances, train_model


def main():
    torch.set_num_threads(1)
    threadpool_limits(1)
    rng = np.random.default_rng(20261023)
    factors = rng.normal(size=(18, 14, 41, 8)).astype(np.float32)
    covariances = factors @ factors.swapaxes(-1, -2) / 8. + np.eye(41, dtype=np.float32)[None, None] * .3
    covariances = covariances.astype(np.float32)
    original_input = covariances.copy()
    labels = np.tile(np.arange(3), 6).astype(np.int64)
    weights = np.linspace(.8, 1.2, 18, dtype=np.float32)
    subjects = np.repeat(["P01", "P02", "P03"], 6)
    checks = []

    reference, _ = fit_fixed(covariances, labels, weights, MODEL_CONFIG,
                             seed=33, epochs=3, schedule_epochs=120)
    baseline, _, info = train_model(covariances, labels, weights, subjects,
                                    MODEL_CONFIG, 33, 3, heldout_subject="P04")
    assert set(reference.state_dict()) == set(baseline.state_dict())
    assert all(torch.equal(reference.state_dict()[k], v) for k, v in baseline.state_dict().items())
    checks.append(dict(name="baseline_bitwise_equal_to_v3_loop", passed=True,
                       n_state_tensors=len(baseline.state_dict()), synthetic_epochs=3))

    masked, targets = mask_covariances(torch.from_numpy(covariances[:4]),
                                      np.random.default_rng(17), .5, .5)
    assert masked.shape == (4, 14, 41, 41) and targets.shape == (4, 574)
    assert torch.allclose(masked, masked.transpose(-1, -2), rtol=0., atol=0.)
    smallest = float(np.linalg.eigvalsh(masked.numpy()).min())
    assert smallest >= -1e-5
    # Every channel replacement is shared across all banks; band masks across
    # the two halves are identical (broadband banks are never band-masked).
    identity = np.eye(41, dtype=np.float32)
    bank_mask = np.all(masked.numpy() == identity, axis=(-1, -2))
    assert np.array_equal(bank_mask[:, :7], bank_mask[:, 7:])
    assert not bank_mask[:, [6, 13]].any()
    checks.append(dict(name="mask_shapes_symmetry_PSD_and_paired_halves", passed=True,
                       minimum_eigenvalue=smallest))

    for recipe in ("augmentation", "adversarial", "masked_ssl"):
        net, history, info = train_model(covariances, labels, weights, subjects,
                                         MODEL_CONFIG, 42, 2, recipe=recipe,
                                         heldout_subject="P04")
        assert type(net) is SpatialLogVarianceNet
        assert set(net.state_dict()) == set(baseline.state_dict())
        assert all(torch.isfinite(t).all() for t in net.state_dict().values())
        with torch.inference_mode():
            logits = net(torch.from_numpy(covariances[:2]))
        assert logits.shape == (2, 3) and torch.isfinite(logits).all()
        assert info["fitted_subjects"] == ["P01", "P02", "P03"]
        assert info["deployment_auxiliary_heads"] is False
        if recipe == "masked_ssl":
            assert info["pretraining_subjects"] == ["P01", "P02", "P03"]
            assert info["pretraining"]["labels_used"] is False
            assert info["pretraining"]["retained_parameters"] == ["spatial_weight"]
            target = torch.diagonal(torch.from_numpy(covariances), dim1=-2, dim2=-1).clamp_min(1e-6).log().flatten(1)
            np.testing.assert_array_equal(info["pretraining"]["target_center"], target.mean(0).numpy())
            # Only the two supervised batches remain in BatchNorm after the
            # reset; the 30 pretraining batches do not persist in its count.
            assert int(net.normalizer.num_batches_tracked) == 2
        checks.append(dict(name=recipe + "_finite_forward_and_no_auxiliary_deployment_head",
                           passed=True, supervised_synthetic_epochs=2))

    for recipe in ("baseline", "augmentation", "adversarial", "masked_ssl"):
        try:
            train_model(covariances, labels, weights, subjects, MODEL_CONFIG,
                        33, 1, recipe=recipe, heldout_subject="P02")
        except ValueError as exc:
            assert "Held-out participant" in str(exc)
        else:
            raise AssertionError("A held-out fitting subject was accepted")
    checks.append(dict(name="heldout_rejected_before_any_recipe_fit", passed=True, recipes=4))
    assert np.array_equal(original_input, covariances)
    checks.append(dict(name="fitting_input_immutable", passed=True))
    result = dict(all_passed=True, data_used="synthetic SPD covariances only; no EEG or CV outcomes",
                  n_checks=len(checks), checks=checks)
    output = ROOT / "v5_work/neural_training_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
