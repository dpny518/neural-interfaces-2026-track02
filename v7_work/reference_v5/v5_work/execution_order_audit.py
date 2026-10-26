"""Synthetic check that earlier jobs/RNG state cannot change a fixed recipe fit.

No EEG, held-out predictions or scores are loaded. This checks a scheduling
change, not a model candidate, at the same two-thread setting as the CV jobs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from v5_work.neural_training_v5 import MODEL_CONFIG, train_model
from v5_work.protocol_v5 import sha256_file, write_json_exclusive


def main():
    torch.set_num_threads(2)
    threadpool_limits(2)
    generator = np.random.default_rng(20261024)
    factor = generator.standard_normal((18, 14, 41, 10)).astype(np.float32)
    covariances = factor @ np.swapaxes(factor, -1, -2) / 10 + np.eye(41, dtype=np.float32) * .5
    labels = np.arange(18, dtype=np.int64) % 3
    weights = np.ones(18, dtype=np.float32)
    subjects = np.repeat([f"P{i:02d}" for i in range(2, 11)], 2)

    first, first_history, first_info = train_model(
        covariances, labels, weights, subjects, MODEL_CONFIG,
        seed=33, epochs=2, recipe="augmentation", heldout_subject="P01")
    torch.manual_seed(123456)
    torch.randn(10001)
    np.random.seed(987654)
    np.random.random(10001)
    # An intervening, independently initialized fit emulates another prior
    # fold/job in the long-running process, using only synthetic fit values.
    intervening, _, _ = train_model(
        covariances, labels, weights, subjects, MODEL_CONFIG,
        seed=7331, epochs=1, recipe="baseline", heldout_subject="P01")
    del intervening
    second, second_history, second_info = train_model(
        covariances, labels, weights, subjects, MODEL_CONFIG,
        seed=33, epochs=2, recipe="augmentation", heldout_subject="P01")
    equal = {name: bool(torch.equal(value, second.state_dict()[name]))
             for name, value in first.state_dict().items()}
    result = dict(all_passed=all(equal.values()) and first_history == second_history
                  and first_info == second_info,
                  check="fixed augmentation fit independent of previous jobs/global RNG state",
                  tensor_equal=equal, seed=33, synthetic_epochs=2, threads=2,
                  n_synthetic_examples=18, real_EEG_loaded=False, scores_computed=False,
                  intervention="poison both global RNGs and fit a different seeded baseline model",
                  train_model_source_sha256=sha256_file(ROOT / "v5_work/neural_training_v5.py"),
                  audit_source_sha256=sha256_file(Path(__file__)))
    write_json_exclusive(ROOT / "v5_work/execution_order_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise RuntimeError("Fixed training result depends on previous jobs/RNG state")


if __name__ == "__main__":
    main()
