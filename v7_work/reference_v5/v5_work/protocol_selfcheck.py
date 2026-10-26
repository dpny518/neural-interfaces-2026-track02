"""Synthetic negative controls for stage provenance and paired statistics.

This file neither opens EEG nor evaluates a model. It tests failure modes that
would otherwise permit held-out-subject leakage or a mislabeled final fit.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, SUBJECT_FIELDS, LeakageError, assert_stage_subjects,
    make_provenance, validate_provenance, register_training_artifact,
    check_training_artifacts, freeze_selection, sha256_file,
    claim_legacy_diagnostic_once,
)
from v5_work.analyze_cv import paired_subject_cluster


def main():
    checks = []

    def passed(name, action):
        action()
        checks.append({"name": name, "passed": True})

    def rejected(name, action, exception=LeakageError):
        try:
            action()
        except exception:
            checks.append({"name": name, "passed": True})
        else:
            raise AssertionError(f"Expected rejection: {name}")

    for heldout in PUBLIC_SUBJECTS:
        fitted = [s for s in PUBLIC_SUBJECTS if s != heldout]
        passed("valid_cv_fit_" + heldout,
               lambda f=fitted, h=heldout: assert_stage_subjects(f, stage="cv_fit", heldout_subject=h))
        rejected("heldout_in_fit_" + heldout,
                 lambda h=heldout: assert_stage_subjects(PUBLIC_SUBJECTS, stage="cv_fit", heldout_subject=h))
    fitted = [s for s in PUBLIC_SUBJECTS if s != "P04"]
    valid = make_provenance(stage="cv_fit", fitted_subjects=fitted, heldout_subject="P04",
                            pretraining_subjects=fitted, batchnorm_subjects=fitted,
                            config={"fixed_epoch": 75}, seed=33, epoch=75, row_indices=range(100))
    passed("valid_fold_provenance", lambda: validate_provenance(valid))
    for field in SUBJECT_FIELDS:
        bad = copy.deepcopy(valid)
        bad[field] = list(PUBLIC_SUBJECTS)
        rejected("heldout_in_" + field, lambda b=bad: validate_provenance(b))
    bad = copy.deepcopy(valid)
    bad["config"]["fixed_epoch"] = 20
    rejected("configuration_hash_binding", lambda: validate_provenance(bad))
    rejected("nonpublic_subject", lambda: assert_stage_subjects(["P99"], stage="cv_evaluate", heldout_subject="P04"))
    rejected("final_fit_requires_passing_freeze", lambda: assert_stage_subjects(PUBLIC_SUBJECTS, stage="final_fit"))
    with tempfile.TemporaryDirectory(prefix="v5-protocol-selfcheck-") as temporary:
        directory = Path(temporary)
        artifact = directory / "checkpoint.bin"
        artifact.write_bytes(b"synthetic checkpoint, not model weights")
        manifest = register_training_artifact(artifact, valid)
        passed("bound_artifact_valid", lambda: check_training_artifacts([manifest]))
        artifact.write_bytes(b"changed")
        rejected("changed_artifact_rejected", lambda: check_training_artifacts([manifest]))
        artifact.write_bytes(b"synthetic checkpoint, not model weights")
        rejected("registered_manifest_not_overwritten", lambda: register_training_artifact(artifact, valid), FileExistsError)
        plan = directory / "plan.json"
        plan.write_text(json.dumps({"warmup_test_role_retired": True}))
        selection = directory / "selection.json"
        selection.write_text(json.dumps({"decision": {"adopt": False, "selected_candidate": None},
                                         "warmup_test_role_retired": True, "plan_sha256": sha256_file(plan)}))
        freeze = directory / "freeze.json"
        rejected("null_result_cannot_freeze", lambda: freeze_selection(selection_path=selection, plan_path=plan, freeze_path=freeze))
        passed_selection = {"decision": {"adopt": True, "selected_candidate": "synthetic"},
                            "warmup_test_role_retired": True, "plan_sha256": sha256_file(plan),
                            "candidates": {"synthetic": {"paired_comparison": {
                                "mean_paired_gain": .02, "paired_gain_subject_cluster_se": .01,
                                "passes_adoption_rule": True}}}}
        forged = copy.deepcopy(passed_selection)
        forged["candidates"]["synthetic"]["paired_comparison"]["mean_paired_gain"] = .005
        selection.write_text(json.dumps(forged))
        rejected("adopt_boolean_without_passing_gain_rejected", lambda: freeze_selection(selection_path=selection, plan_path=plan, freeze_path=freeze))
        selection.write_text(json.dumps(passed_selection))
        freeze_selection(selection_path=selection, plan_path=plan, freeze_path=freeze)
        passed("all_public_final_fit_after_freeze", lambda: assert_stage_subjects(PUBLIC_SUBJECTS, stage="final_fit", selection_freeze=freeze))
        final = make_provenance(stage="final_fit", fitted_subjects=PUBLIC_SUBJECTS, selection_freeze=freeze)
        passed("all_public_final_provenance", lambda: validate_provenance(final))
        receipt = directory / "diagnostic_receipt.json"
        passed("legacy_diagnostic_claim", lambda: claim_legacy_diagnostic_once(selection_freeze=freeze, model_path=artifact, receipt_path=receipt))
        rejected("legacy_diagnostic_claim_not_repeatable", lambda: claim_legacy_diagnostic_once(selection_freeze=freeze, model_path=artifact, receipt_path=receipt), FileExistsError)
        selection.write_text("{}")
        rejected("selection_immutable_after_freeze", lambda: assert_stage_subjects(PUBLIC_SUBJECTS, stage="final_fit", selection_freeze=freeze))

    keys = []
    for subject in PUBLIC_SUBJECTS:
        for session in range(1, 4 if subject in {"P04", "P10"} else 2):
            for interface in ("BrainHero", "Graz"):
                keys.append((subject, str(session), interface))
    assert len(keys) == 28
    baseline = np.full((3, 28), .5)
    zero, _ = paired_subject_cluster(baseline, baseline, keys)
    assert zero["mean_paired_gain"] == 0 and zero["paired_gain_subject_cluster_se"] == 0
    assert not zero["passes_adoption_rule"]
    checks.append({"name": "zero_gain_fails_strict_gate", "passed": True})
    constant, _ = paired_subject_cluster(baseline + .02, baseline, keys)
    assert np.isclose(constant["mean_paired_gain"], .02) and constant["passes_adoption_rule"]
    checks.append({"name": "constant_subject_gain_passes", "passed": True})
    modified = baseline.copy()
    for column, key in enumerate(keys):
        if key[0] == "P04":
            modified[:, column] += .1
    weighted, boot = paired_subject_cluster(modified, baseline, keys)
    assert np.isclose(weighted["mean_paired_gain"], .1 * 6 / 28)
    repeated, boot2 = paired_subject_cluster(modified, baseline, keys)
    assert np.array_equal(boot, boot2) and weighted == repeated
    checks.append({"name": "official_cell_weighting_preserved", "passed": True})
    checks.append({"name": "bootstrap_fixed_seed_reproducible", "passed": True})
    result = {"all_passed": True, "n_checks": len(checks), "checks": checks,
              "data_used": "synthetic metadata/checkpoint bytes/scores only; no EEG, model fitting or measured outcomes"}
    (ROOT / "v5_work/protocol_selfcheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_passed": True, "n_checks": len(checks)}))


if __name__ == "__main__":
    main()
