"""Completion-only bindings for the fixed v5 classical LOSO run.

This does not fit, score, select, or modify a model. It refuses an incomplete
run and records hashes at completion; it does not claim they were captured
before fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, check_training_artifacts, json_sha256, sha256_file,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/classical_results")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan_hash = sha256_file(args.plan)
    if plan_hash != args.expected_plan_sha256:
        raise RuntimeError("Frozen plan hash changed")
    summary_path = args.out / "cv_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("complete") is not True or summary.get("stage") != "cv":
        raise RuntimeError("Completion bindings require all CV folds")
    if summary.get("plan_sha256") != plan_hash:
        raise RuntimeError("Classical fit used a different plan")
    expected = set(PUBLIC_SUBJECTS)
    if {row["fold"] for row in summary["folds"]} != expected or len(summary["folds"]) != 10:
        raise RuntimeError("Classical run did not complete each of the ten subjects once")
    metadata_path = args.cache / "metadata.npz"
    prediction_path = args.out / "cv_predictions.npz"
    with np.load(metadata_path, allow_pickle=False) as archive:
        subjects = archive["subject"].copy()
    rows = np.arange(len(subjects), dtype=np.int64)
    with np.load(prediction_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["row_indices"], rows)
        np.testing.assert_array_equal(archive["subjects"], subjects)
        for branch in ("ridge", "tangent"):
            probabilities = archive[branch + "_probabilities"]
            if probabilities.shape != (len(rows), 3):
                raise RuntimeError("Invalid complete OOF probability shape")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise RuntimeError("OOF probabilities are invalid")
            np.testing.assert_allclose(probabilities.sum(1), 1., rtol=0., atol=1e-12)
    manifests = []
    for heldout in PUBLIC_SUBJECTS:
        for branch in ("ridge", "tangent"):
            manifest = args.out / ("hold_" + heldout) / (branch + ".provenance.json")
            evidence = json.loads(manifest.read_text())["provenance"]
            if evidence["config"]["plan_sha256"] != plan_hash:
                raise RuntimeError("A branch artifact has a different frozen plan")
            if evidence["row_indices_sha256"] != json_sha256(rows[subjects != heldout].tolist()):
                raise RuntimeError("A branch fitting row set differs from its LOSO fold")
            manifests.append(manifest)
    provenance = check_training_artifacts(manifests)
    record = dict(
        schema="track2-v5-classical-completion-integrity-v1",
        stage="cv", complete=True,
        integrity_checked_at_completion=True,
        integrity_observation="Hashes checked only after the complete fixed 10-fold run",
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_hash,
        metadata_sha256=sha256_file(metadata_path),
        raw_covariances_sha256=sha256_file(args.cache / "temporal_band_covs.npy"),
        predictions_sha256=sha256_file(prediction_path),
        summary_sha256=sha256_file(summary_path),
        row_indices_sha256=json_sha256(rows.tolist()),
        n_rows=len(rows), folds=list(PUBLIC_SUBJECTS),
        code_sha256={name: sha256_file(ROOT / "v5_work" / name) for name in (
            "classical_v5.py", "riemann_v5_runtime.py", "finalize_classical_integrity.py")},
        no_heldout_leakage=provenance,
    )
    destination = args.out / "cv_integrity.json"
    with destination.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(complete=True, n_rows=len(rows), n_artifacts=len(manifests),
                          integrity_path=str(destination), plan_sha256=plan_hash)))


if __name__ == "__main__":
    main()
