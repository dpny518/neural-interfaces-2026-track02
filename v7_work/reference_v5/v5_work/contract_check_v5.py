"""The unchanged original 16 runtime checks plus v5 stage-specific isolation.

This wrapper is conditional on a passing, immutable CV selection. It checks
the complete 270-artifact CV evidence and the selected post-selection fit(s),
then binds each packaged learned member to its registered source checkpoint.
The original runtime checker is not modified. Its cache-loader symbol is
temporarily supplied with twelve real windows from the current stage's fitting
population; no evaluation labels or old warm-up split loader are used.

For a null result this executable is not run: there is no v5 submission and
the runtime checks are inapplicable. The completed CV isolation audit remains
valid independently of submission construction.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bci_work")]
import contract_check as original_contract
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, LEGACY_TRAIN_SUBJECTS, SEED_PAIRS, SUBJECT_FIELDS,
    LeakageError, assert_stage_subjects, check_training_artifacts,
    json_sha256, resolve_manifest_artifact, sha256_file, validate_provenance,
    verify_selection_freeze,
)
from v5_work.assemble_cv_predictions import audit_signature, FROZEN_PLAN_SHA256

FITTING_FIELD_ALIASES = {
    **{field: field for field in SUBJECT_FIELDS},
    "training_subjects": "fitted_subjects",
    "source_prior_subjects": "reference_subjects",
    "normalization_fit_subjects": "normalization_subjects",
}


def require(condition, message):
    if not condition:
        raise LeakageError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_recorded_path(value, anchor=None):
    """Rebase archived evidence without changing its hash-bound contents."""
    path = Path(value)
    candidates = []
    if anchor is not None:
        anchor = Path(anchor).resolve()
        if not path.is_absolute():
            candidates.append(anchor.parent / path)
        candidates.append(anchor.parent / path.name)
    for prefix in ("v5_work", "v5_deliverables"):
        if prefix in path.parts:
            candidates.append(ROOT.joinpath(*path.parts[path.parts.index(prefix):]))
    candidates.extend((ROOT / path, path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise LeakageError("Cannot locate archived evidence: " + str(value))


def expected_populated_fields(record):
    config = record["config"]
    populated = {"fitted_subjects", "normalization_subjects", "reference_subjects"}
    if "branch" in config:
        require(config["branch"] in {"ridge", "tangent"}, "Unknown classical branch")
        populated.add("feature_scale_subjects")
    elif "recipe" in config:
        require(config["recipe"] in {"baseline", "augmentation", "adversarial", "masked_ssl"}, "Unknown neural recipe")
        populated.add("batchnorm_subjects")
        if config["recipe"] == "masked_ssl":
            populated.update(("pretraining_subjects", "feature_scale_subjects"))
    else:
        require(config.get("artifact_kind") in {
            "neural_fold_source_and_row_partition", "postselection_neural_source"}, "Unknown fitted artifact kind")
    return populated


def validate_actual_fit_record(record, subjects, *, plan_hash, selection_freeze):
    stage, heldout = record["stage"], record["heldout_subject"]
    if stage == "cv_fit":
        rows = np.flatnonzero(subjects != heldout)
    else:
        require(stage in {"final_fit", "legacy_fit"}, "Unexpected fitting stage")
        ids = PUBLIC_SUBJECTS if stage == "final_fit" else LEGACY_TRAIN_SUBJECTS
        rows = np.flatnonzero(np.isin(subjects, ids))
    fitting = assert_stage_subjects(subjects[rows], stage=stage, heldout_subject=heldout,
                                   selection_freeze=selection_freeze if stage != "cv_fit" else None)
    require(record["n_fitting_rows"] == len(rows)
            and record["row_indices_sha256"] == json_sha256(rows.tolist()), "Artifact actual fitting rows differ")
    require(record["config"]["plan_sha256"] == plan_hash, "Artifact belongs to another frozen plan")
    for field in SUBJECT_FIELDS:
        wanted = fitting if field in expected_populated_fields(record) else []
        require(record[field] == wanted, "Artifact fitting population differs: " + field)
    return rows


def inspect_embedded_subjects(payload, *, stage, fitting, registered_provenances,
                              expected=None, path="checkpoint"):
    """Verify semantic fitting metadata, including all seven provenance fields."""
    checked = []
    if isinstance(payload, dict):
        if "provenance" in payload:
            expected = validate_provenance(payload["provenance"])
            require(expected["stage"] == stage, "Embedded stage differs: " + path)
            require(expected in registered_provenances, "Embedded provenance is not a registered source fit: " + path)
        for key, value in payload.items():
            here = path + "." + str(key)
            if key in FITTING_FIELD_ALIASES:
                field = FITTING_FIELD_ALIASES[key]
                wanted = expected[field] if expected is not None else fitting
                actual = sorted(map(str, value))
                require(actual == wanted, "Embedded fitting metadata differs: " + here)
                checked.append(dict(field=here, subjects=actual))
            elif key == "heldout_subject":
                wanted = expected["heldout_subject"] if expected is not None else None
                require(value == wanted, "Embedded held-out participant differs: " + here)
            elif isinstance(value, (dict, list, tuple)):
                checked.extend(inspect_embedded_subjects(
                    value, stage=stage, fitting=fitting, registered_provenances=registered_provenances,
                    expected=expected, path=here))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            if isinstance(value, (dict, list, tuple)):
                checked.extend(inspect_embedded_subjects(
                    value, stage=stage, fitting=fitting, registered_provenances=registered_provenances,
                    expected=expected, path=f"{path}[{index}]"))
    return checked


def tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(tree_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def load_packaged_weights(submission_dir, zip_file=None):
    submission_dir = Path(submission_dir)
    archive = submission_dir if submission_dir.is_file() else zip_file
    if archive is not None:
        with zipfile.ZipFile(archive) as package:
            return torch.load(io.BytesIO(package.read("weights.pt")), weights_only=True, map_location="cpu")
    return torch.load(submission_dir / "weights.pt", weights_only=True, map_location="cpu")


def validate_stage_artifact_groups(records, *, cv):
    groups = {}
    for record in records:
        key = record["heldout_subject"] if cv else record["stage"]
        groups.setdefault(key, []).append(record)
    require(set(groups) == set(PUBLIC_SUBJECTS) if cv else bool(groups), "Artifact stage/fold groups are incomplete")
    for key, values in groups.items():
        references = [v for v in values if "artifact_kind" in v["config"]]
        neural = [v for v in values if "recipe" in v["config"]]
        classical = [v for v in values if "branch" in v["config"]]
        require(len(references) == 1 and len(values) == (27 if cv else 4), "Incomplete reference/neural/classical group")
        require(len(classical) == (2 if cv else 1) and len(neural) == (24 if cv else 2), "Unexpected artifact group count")
        if cv:
            wanted = {(recipe, seed, epoch) for recipe in ("baseline", "augmentation", "adversarial", "masked_ssl")
                      for pair in SEED_PAIRS for seed, epoch in zip(pair, (75, 20))}
            require({(v["config"]["recipe"], v["seed"], v["epoch"]) for v in neural} == wanted, "CV recipe/seed/epoch coverage differs")
            require({v["config"]["branch"] for v in classical} == {"ridge", "tangent"}, "CV classical coverage differs")
        else:
            require({(v["seed"], v["epoch"]) for v in neural} == {(33, 75), (42, 20)}, "Final seeds or epochs differ")
    return groups


def check_v5(submission_dir, output, *, stage, selection_freeze, cv_audit, manifests,
             cache_dir=ROOT / "v5_work/cache", zip_file=None, kit_dir=None):
    require(stage in {"final_fit", "legacy_fit"}, "Contract checks require an explicit post-selection stage")
    output, cache_dir, cv_audit = Path(output), Path(cache_dir), Path(cv_audit)
    selection_freeze = resolve_recorded_path(selection_freeze)
    frozen = verify_selection_freeze(selection_freeze)
    require(frozen["plan_sha256"] == FROZEN_PLAN_SHA256, "Post-selection plan differs")
    selection_path = resolve_recorded_path(frozen["selection_path"], selection_freeze)
    selection = read_json(selection_path)
    assembly_path = cv_audit.parent / "assembly_manifest.json"
    assembly = read_json(assembly_path)
    require(assembly["complete"] is True and assembly["plan_sha256"] == frozen["plan_sha256"], "CV assembly is incomplete")
    require(sha256_file(assembly_path) == selection["assembly_manifest_sha256"], "Selection used another CV assembly")
    require(sha256_file(cv_audit) == assembly["stage_specific_audit"]["sha256"], "Completed CV audit changed")
    saved_cv = read_json(cv_audit)
    require(saved_cv["n_artifacts"] == 270 and saved_cv["all_actual_fit_rows_match"] is True, "Incomplete CV audit")
    cv_manifests = [resolve_recorded_path(record["manifest_path"], cv_audit) for record in saved_cv["artifacts"]]
    current_cv = check_training_artifacts(cv_manifests)
    require(audit_signature(current_cv) == audit_signature(saved_cv), "Actual CV artifacts differ from completed audit")
    validate_stage_artifact_groups(current_cv["artifacts"], cv=True)
    with np.load(cache_dir / "metadata.npz", allow_pickle=False) as archive:
        subjects = np.array(archive["subject"], copy=True)
    require(subjects.shape == (2655,), "Expected complete public metadata")
    require(sha256_file(cache_dir / "metadata.npz") == assembly["metadata_sha256"], "Metadata differs from CV fitting evidence")
    post_manifests = list(dict.fromkeys(resolve_recorded_path(value) for value in manifests))
    require(post_manifests, "Provide registered post-selection fitting artifacts")
    current_post = check_training_artifacts(post_manifests)
    require(all(record["stage"] in {"final_fit", "legacy_fit"} for record in current_post["artifacts"]),
            "Post-selection manifest list contains another fitting stage")
    post_groups = validate_stage_artifact_groups(current_post["artifacts"], cv=False)
    require(stage in post_groups, "No complete fitting evidence for the packaged stage")
    all_records = current_cv["artifacts"] + current_post["artifacts"]
    require(len({record["artifact_path"] for record in all_records}) == len(all_records), "Duplicate learned artifact")
    embedded_checks = []
    tensors_by_stage = {}
    for record in all_records:
        rows = validate_actual_fit_record(record, subjects, plan_hash=frozen["plan_sha256"], selection_freeze=selection_freeze)
        if "artifact_kind" in record["config"]:
            require(record["config"]["metadata_sha256"] == assembly["metadata_sha256"]
                    and record["config"]["raw_covariances_sha256"] == assembly["raw_covariances_sha256"],
                    "Fitted covariance source belongs to different public input data")
        artifact = Path(record["artifact_path"])
        expected = read_json(record["manifest_path"])["provenance"]
        if artifact.suffix == ".npz":
            with np.load(artifact, allow_pickle=False) as archive:
                require(np.array_equal(archive["fit_rows"], rows), "Reference contains different fitting indices")
                if record["stage"] == "cv_fit":
                    require(np.array_equal(archive["held_rows"], np.flatnonzero(subjects == record["heldout_subject"])),
                            "CV reference contains different held-out indices")
                source = np.array(archive["source"], copy=True)
            require(source.shape == (14, 41, 41) and np.isfinite(source).all(), "Invalid fitted source covariance")
            if record["stage"] != "cv_fit":
                tensors_by_stage.setdefault(record["stage"], {})["reference"] = source
            embedded_checks.append(dict(artifact=str(artifact), actual_row_indices_match=True))
        else:
            require(artifact.suffix == ".pt", "Unexpected learned artifact extension")
            payload = torch.load(artifact, weights_only=True, map_location="cpu")
            require(payload["provenance"] == expected, "Embedded artifact provenance differs from its registered record")
            fields = inspect_embedded_subjects(payload, stage=record["stage"], fitting=record["fitted_subjects"],
                                               registered_provenances=[expected], expected=expected, path=artifact.name)
            require(all(any(item["field"].endswith("." + field) for item in fields) for field in SUBJECT_FIELDS),
                    "Checkpoint omitted an embedded fitting-provenance field")
            embedded_checks.append(dict(artifact=str(artifact), checked_subject_fields=fields))
            if record["stage"] != "cv_fit":
                key = (record["config"].get("recipe"), record["seed"], record["epoch"]) if "recipe" in record["config"] else record["config"]["branch"]
                tensors_by_stage.setdefault(record["stage"], {})[key] = payload
    candidate = frozen["selected_candidate"]
    recipe = "baseline" if candidate == "riemannian_tangent" else candidate
    branch = "tangent" if candidate == "riemannian_tangent" else "ridge"
    fitting = list(PUBLIC_SUBJECTS if stage == "final_fit" else LEGACY_TRAIN_SUBJECTS)
    current = tensors_by_stage[stage]
    require(set(current) == {"reference", branch, (recipe, 33, 75), (recipe, 42, 20)}, "Selected package source set differs")
    for seed, epoch in ((33, 75), (42, 20)):
        require(np.array_equal(current[(recipe, seed, epoch)]["source"].numpy(), current["reference"]),
                "Post-selection neural source differs from registered reference")
    packed = load_packaged_weights(submission_dir, zip_file)
    require(packed["schema_version"] == 1 and packed["candidate"] == candidate
            and packed["stage"] == stage and packed["training_subjects"] == fitting,
            "Packaged stage/candidate/fitting participants differ")
    require(packed["warmup_test_role_retired"] is True
            and packed["selection_freeze_sha256"] == sha256_file(selection_freeze), "Packaged selection freeze differs")
    require(packed["dataset"] == "nm000290" and packed["version"] == "1.0.0", "Packaged data source differs")
    require(len(packed["members"]) == 3, "Unexpected ensemble member count")
    for index, (seed, epoch) in enumerate(((33, 75), (42, 20))):
        member = packed["members"][index]
        require(member["type"] == "neural_cov" and member["weight"] == .45 and member["temperature"] == 1.,
                "Packaged neural mixture differs from the frozen plan")
        require(member["views"] == [{"crop": [0, 240]}, {"crop": [240, 480]}], "Packaged temporal views differ")
        require(tree_equal(member["checkpoint"], current[(recipe, seed, epoch)]), "Packaged neural tensors differ from registered checkpoint")
    classical_member = dict(packed["members"][2])
    require(classical_member.pop("weight") == .1, "Packaged classical ensemble weight differs")
    require(tree_equal(classical_member, current[branch]), "Packaged classical tensors differ from registered checkpoint")
    stage_provenances = [read_json(record["manifest_path"])["provenance"] for record in post_groups[stage]]
    packaged_fields = inspect_embedded_subjects(packed, stage=stage, fitting=fitting,
                                               registered_provenances=stage_provenances, path="packaged_weights")

    # Only the existing checker's data source is adapted. Its original 16
    # packaging, load/predict, deepcopy/reset, and causal-prefix checks are kept.
    selected_rows = np.flatnonzero(np.isin(subjects, fitting))
    assert_stage_subjects(subjects[selected_rows], stage=stage, selection_freeze=selection_freeze)
    contract_rows = selected_rows[:12]
    require(len(contract_rows) == 12, "Runtime checks require twelve real fitting windows")

    def fitting_cache_loader(cache_dir_ignored=None, split=None, mmap_mode="r"):
        require(split == "train", "Original runtime checker requested an unexpected split")
        windows = np.load(cache_dir / "windows.npy", mmap_mode="r", allow_pickle=False)
        require(windows.shape == (2655, 41, 480), "Unexpected input window cache shape")
        info = read_json(cache_dir / "dataset_info.json")
        return np.array(windows[contract_rows], copy=True), {"subject": subjects[contract_rows]}, info

    original_loader = original_contract.load_cache
    try:
        original_contract.load_cache = fitting_cache_loader
        report = original_contract.check_submission(submission_dir, output, zip_file=zip_file, kit_dir=kit_dir)
    finally:
        original_contract.load_cache = original_loader
    require(len(report["checks"]) == 16, "Expected all 16 original runtime checks; supply the final ZIP")
    isolation_check = dict(name="stage_specific_no_heldout_leakage", passed=True,
                           stage=stage, fitting_subjects=fitting, n_cv_artifacts=270,
                           n_postselection_artifacts=len(current_post["artifacts"]), n_artifacts=len(all_records),
                           all_seven_fields_validated=True, all_actual_fit_rows_match=True,
                           packaged_tensors_equal_registered_sources=True,
                           cv_audit_sha256=sha256_file(cv_audit), selection_freeze_sha256=sha256_file(selection_freeze),
                           artifact_checks=all_records, embedded_source_metadata=embedded_checks,
                           packaged_subject_metadata=packaged_fields,
                           explanation="Every CV fold excludes its own held-out participant in all fitting fields. All10 final fitting and original6 legacy fitting occur only after the immutable passing selection.")
    report["checks"].append(isolation_check)
    report.update(all_passed=all(check["passed"] for check in report["checks"]), n_checks=17,
                  stage=stage, data_used="Twelve current-stage fitting windows plus synthetic signals; no evaluation labels",
                  runtime_check_fitting_rows=contract_rows.tolist(), fitting_subjects=fitting,
                  warmup_test_role_retired=True, legacy_diagnostic_used_for_selection=False)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(all_passed=report["all_passed"], n_checks=17, stage=stage,
                          n_cv_artifacts=270, n_postselection_artifacts=len(current_post["artifacts"]),
                          output=str(output))), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("final_fit", "legacy_fit"), required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--cv-audit", type=Path, default=ROOT / "v5_work/cv_predictions/stage_specific_leakage_check.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--zip-file", type=Path)
    parser.add_argument("--kit-dir", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--manifest-list", type=Path)
    args = parser.parse_args()
    values = list(args.manifest)
    if args.manifest_list is not None:
        listed = read_json(args.manifest_list)
        values.extend(listed["manifests"] if isinstance(listed, dict) else listed)
    if not values:
        parser.error("Provide registered post-selection references, two neural checkpoints and the selected classical branch")
    check_v5(args.submission_dir, args.output, stage=args.stage, selection_freeze=args.selection_freeze,
             cv_audit=args.cv_audit, manifests=values, cache_dir=args.cache, zip_file=args.zip_file,
             kit_dir=args.kit_dir)
