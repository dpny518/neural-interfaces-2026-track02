"""Verify the complete fixed CV panel and assemble probabilities, without scoring.

Every learned artifact is bound to actual fitting row indices and the frozen
plan. The held-out participant must be absent from all seven fitting-provenance
fields. Predictions are assembled only after all ten folds, all four neural
recipes, and both deterministic classical branches have completed.

This script does not choose a seed, compute an argmax, calculate accuracy, or
change any fitting/adaptation setting. Analysis is a separate explicit step.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, SEED_PAIRS, SUBJECT_FIELDS, assert_stage_subjects,
    check_training_artifacts, json_sha256, sha256_file, write_json_exclusive,
)

FROZEN_PLAN_SHA256 = "ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59"
RECIPES = ("baseline", "augmentation", "adversarial", "masked_ssl")
CANDIDATES = ("baseline_v3", "augmentation", "adversarial", "masked_ssl", "riemannian_tangent")
N_ROWS = 2655


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def input_hash(path, ledger):
    path = Path(path).resolve()
    key = str(path)
    if key not in ledger:
        ledger[key] = sha256_file(path)
    return ledger[key]


def assert_probability_array(value, shape, label):
    require(value.shape == shape, f"{label}: expected shape {shape}, got {value.shape}")
    require(np.isfinite(value).all(), f"{label}: nonfinite probabilities")
    require((value >= -1e-7).all() and (value <= 1 + 1e-7).all(), f"{label}: invalid probability range")
    require(np.allclose(value.sum(axis=-1), 1., atol=1e-5, rtol=0), f"{label}: probabilities do not sum to one")


def assert_provenance(record, *, heldout, fit_rows, expected_fields, config, seed, epoch):
    """Bind a validated artifact's provenance to the actual current row split."""
    require(record["stage"] == "cv_fit" and record["heldout_subject"] == heldout,
            "Artifact does not describe the requested held-out CV fold")
    fit_subjects = sorted(set(PUBLIC_SUBJECTS) - {heldout})
    require(record["fitted_subjects"] == fit_subjects, "Unexpected fitting participants")
    for field in SUBJECT_FIELDS:
        wanted = fit_subjects if field in expected_fields else []
        require(record[field] == wanted, f"{heldout}: unexpected {field}")
    require(record["n_fitting_rows"] == len(fit_rows), "Fitting row count differs")
    require(record["row_indices_sha256"] == json_sha256(fit_rows.tolist()), "Actual fitting rows differ from provenance")
    require(record["config"] == config, "Artifact recipe differs from the frozen plan")
    require(record["seed"] == seed and record["epoch"] == epoch, "Unplanned seed or checkpoint epoch")
    require(record["selection_freeze"] is None, "CV artifact incorrectly claims a post-selection fit")


def audit_signature(audit):
    """Ignore relocatable paths, preserving every other provenance assertion."""
    require(audit.get("all_passed") is True, "An upstream artifact audit did not pass")
    require(audit.get("check") == "stage_specific_no_heldout_leakage", "Unexpected upstream audit")
    require(audit.get("warmup_test_role_retired") is True, "Old test role was not explicitly retired")
    records = audit["artifacts"]
    require(audit["n_artifacts"] == len(records), "Upstream audit count mismatch")
    normalized = [{k: v for k, v in item.items() if k not in {"manifest_path", "artifact_path"}}
                  for item in records]
    require(len({x["manifest_sha256"] for x in normalized}) == len(normalized), "Duplicate artifact manifest")
    return sorted(normalized, key=lambda x: x["manifest_sha256"])


def preflight(plan_path, cache, fold_cache, neural, classical):
    required = [plan_path, cache / "metadata.npz", cache / "temporal_band_covs.npy",
                fold_cache / "cache_checksums.json", classical / "cv_predictions.npz",
                classical / "cv_summary.json", classical / "cv_integrity.json"]
    for heldout in PUBLIC_SUBJECTS:
        required.extend(fold_cache / heldout / name for name in (
            "cache_checksums.json", "reference.npz", "reference.provenance.json",
            "fit_covariances.npy", "held_covariances.npy"))
        for recipe in RECIPES:
            folder = neural / recipe / ("hold_" + heldout)
            required.extend(folder / name for name in ("predictions.npz", "isolation_check.json"))
            for pair in SEED_PAIRS:
                for seed, epoch in zip(pair, (75, 20)):
                    required.extend(folder / f"seed{seed}_epoch{epoch}{suffix}" for suffix in (
                        ".pt", ".provenance.json", ".training.json"))
        for branch in ("ridge", "tangent"):
            required.extend(classical / ("hold_" + heldout) / (branch + suffix)
                            for suffix in (".pt", ".provenance.json"))
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Complete-panel barrier: missing files (first 15): " + repr(missing[:15])
            + f"; total missing={len(missing)}")


def load_metadata(path):
    fields = ("y", "subject", "session", "interface", "stream_id", "record_id", "onset", "split")
    with np.load(path, allow_pickle=False) as archive:
        meta = {key: np.array(archive[key], copy=True) for key in fields}
    require(all(value.shape == (N_ROWS,) for value in meta.values()), "Metadata row count/shape differs")
    require(np.issubdtype(meta["y"].dtype, np.integer) and np.isin(meta["y"], [0, 1, 2]).all(), "Unexpected labels")
    assert_stage_subjects(meta["subject"], stage="selection")
    require(set(meta["split"]) == {"development"}, "V5 development pool has obsolete split restrictions")
    require(set(meta["interface"]) == {"Graz", "BrainHero"}, "Unexpected interface labels")
    streams = np.unique(meta["stream_id"])
    cells = sorted(set(zip(meta["subject"], meta["session"], meta["interface"])))
    require(len(streams) == 14 and len(cells) == 28, "Expected 14 sessions and 28 scoring cells")
    for stream in streams:
        rows = np.flatnonzero(meta["stream_id"] == stream)
        require(len(set(zip(meta["subject"][rows], meta["session"][rows]))) == 1, "Stream crosses participant/session")
        require(np.all(np.diff(rows) == 1), "A session's rows are not contiguous")
        order = list(zip(meta["record_id"][rows].tolist(), meta["onset"][rows].tolist()))
        require(order == sorted(order) and len(set(order)) == len(order), "Stream order is not chronological")
    return meta, cells


def assemble(*, plan_path, cache, fold_cache, neural, classical, out):
    plan_path, cache, fold_cache, neural, classical, out = map(
        lambda p: Path(p).resolve(), (plan_path, cache, fold_cache, neural, classical, out))
    preflight(plan_path, cache, fold_cache, neural, classical)
    require(not out.exists(), "Refusing to replace an existing assembled panel")
    torch.set_num_threads(1)
    ledger = {}
    plan_hash = input_hash(plan_path, ledger)
    require(plan_hash == FROZEN_PLAN_SHA256, "The plan frozen before fitting has changed")
    plan = read_json(plan_path)
    require(plan["candidates"] == list(CANDIDATES), "Candidate panel changed")
    require(plan["seed_pairs"] == [list(x) for x in SEED_PAIRS], "Seed pairs changed")
    require(plan["fixed_epochs"] == [75, 20] and plan["schedule_epochs"] == 120, "Checkpoint schedule changed")
    require(plan["ensemble_weights"] == [.45, .45, .1], "Ensemble weights changed")
    meta, cells = load_metadata(cache / "metadata.npz")
    metadata_hash = input_hash(cache / "metadata.npz", ledger)
    raw_hash = input_hash(cache / "temporal_band_covs.npy", ledger)
    all_cache = read_json(fold_cache / "cache_checksums.json")
    input_hash(fold_cache / "cache_checksums.json", ledger)
    require(all_cache["all_passed"] is True and all_cache["n_folds"] == 10, "Fold cache is incomplete")
    require(all_cache["plan_sha256"] == plan_hash and all_cache["metadata_sha256"] == metadata_hash
            and all_cache["raw_covariances_sha256"] == raw_hash, "Fold cache belongs to different input data/plan")
    require(all_cache["labels_loaded"] is False, "Covariance reference fitting loaded labels")
    require(set(all_cache["folds"]) == set(PUBLIC_SUBJECTS), "Missing covariance-cache fold")
    for relative, digest in all_cache["files"].items():
        require(input_hash(fold_cache / relative, ledger) == digest, f"Changed fold-cache input: {relative}")
    branches = {recipe: np.full((3, 2, N_ROWS, 3), np.nan, np.float64) for recipe in RECIPES}
    filled = {recipe: np.zeros(N_ROWS, np.int64) for recipe in RECIPES}
    all_manifests, embedded_checks, reference_details = [], [], []

    for heldout in PUBLIC_SUBJECTS:
        folder = fold_cache / heldout
        fit = np.flatnonzero(meta["subject"] != heldout)
        held = np.flatnonzero(meta["subject"] == heldout)
        fit_subjects = assert_stage_subjects(meta["subject"][fit], stage="cv_fit", heldout_subject=heldout)
        assert_stage_subjects(meta["subject"][held], stage="cv_evaluate", heldout_subject=heldout)
        checks = read_json(folder / "cache_checksums.json")
        require(checks == all_cache["folds"][heldout], "Per-fold and global cache records differ")
        require(checks["heldout_subject"] == heldout and checks["fit_subjects"] == fit_subjects,
                "Fold cache has different fitting participants")
        require(checks["n_fit"] == len(fit) and checks["n_held"] == len(held), "Fold cache row counts differ")
        require(checks["alignment"] == plan["neural_alignment"] and checks["causal_session_transform"] is True,
                "Causal adaptation setting differs")
        for name, digest in checks["files"].items():
            require(input_hash(folder / name, ledger) == digest, f"Changed {heldout}/{name}")
        with np.load(folder / "reference.npz", allow_pickle=False) as archive:
            require(np.array_equal(archive["fit_rows"], fit) and np.array_equal(archive["held_rows"], held),
                    "Fold reference row partition differs from actual metadata")
            source = np.array(archive["source"], copy=True)
        require(source.shape == (14, 41, 41) and np.isfinite(source).all(), "Invalid covariance source")
        for name, rows in (("fit_covariances.npy", fit), ("held_covariances.npy", held)):
            value = np.load(folder / name, allow_pickle=False, mmap_mode="r")
            require(value.shape == (len(rows), 14, 41, 41) and value.dtype == np.float32, "Wrong aligned covariance cache")
            del value
        reference_manifest = folder / "reference.provenance.json"
        reference_audit = check_training_artifacts([reference_manifest])
        ref_config = dict(artifact_kind="neural_fold_source_and_row_partition", plan_sha256=plan_hash,
                          alignment=plan["neural_alignment"],
                          source_fit="equal-session arithmetic mean of training-only normalized covariance",
                          metadata_sha256=metadata_hash, raw_covariances_sha256=raw_hash,
                          labels_loaded=False, feature_scaler_fitted=False,
                          normalization="fixed window-local broadband trace plus training-only covariance prior")
        assert_provenance(reference_audit["artifacts"][0], heldout=heldout, fit_rows=fit,
                          expected_fields={"fitted_subjects", "normalization_subjects", "reference_subjects"},
                          config=ref_config, seed=0, epoch=None)
        all_manifests.append(reference_manifest)
        reference_details.append(dict(heldout_subject=heldout, fit_subjects=fit_subjects,
                                      n_fit=len(fit), n_held=len(held),
                                      actual_fit_rows_sha256=json_sha256(fit.tolist()),
                                      source_reference_sha256=checks["files"]["reference.npz"]))
        for recipe in RECIPES:
            destination = neural / recipe / ("hold_" + heldout)
            manifests = [destination / f"seed{seed}_epoch{epoch}.provenance.json"
                         for pair in SEED_PAIRS for seed, epoch in zip(pair, (75, 20))]
            audit = check_training_artifacts(manifests)
            saved_audit = read_json(destination / "isolation_check.json")
            input_hash(destination / "isolation_check.json", ledger)
            require(audit_signature(audit) == audit_signature(saved_audit), "Neural completion audit changed")
            prediction_path = destination / "predictions.npz"
            require(saved_audit["prediction_sha256"] == input_hash(prediction_path, ledger), "Neural predictions changed")
            for index, (seed, epoch) in enumerate((s, e) for pair in SEED_PAIRS for s, e in zip(pair, (75, 20))):
                record = audit["artifacts"][index]
                config = dict(recipe=recipe, model_config=plan["neural_definitions"]["model"],
                              neural_definitions=plan["neural_definitions"], alignment=plan["neural_alignment"],
                              plan_sha256=plan_hash, reference_sha256=checks["files"]["reference.npz"],
                              fixed_epochs=epoch, cosine_horizon=plan["schedule_epochs"])
                populated = {"fitted_subjects", "normalization_subjects", "reference_subjects", "batchnorm_subjects"}
                if recipe == "masked_ssl":
                    populated.update(("pretraining_subjects", "feature_scale_subjects"))
                assert_provenance(record, heldout=heldout, fit_rows=fit, expected_fields=populated,
                                  config=config, seed=seed, epoch=epoch)
                artifact = destination / f"seed{seed}_epoch{epoch}.pt"
                checkpoint = torch.load(artifact, weights_only=True, map_location="cpu")
                manifest = read_json(manifests[index])
                require(checkpoint["provenance"] == manifest["provenance"], "Embedded neural provenance differs")
                require(checkpoint["training_subjects"] == fit_subjects and checkpoint["heldout_subject"] == heldout,
                        "Embedded neural fitting participant set differs")
                require(checkpoint["seed"] == seed and checkpoint["epoch"] == epoch
                        and checkpoint["scheduler_T_max"] == 120 and checkpoint["recipe"] == recipe,
                        "Embedded neural recipe/checkpoint differs")
                require(checkpoint["model_type"] == "spatial_logvariance"
                        and checkpoint["model_config"] == plan["neural_definitions"]["model"]
                        and checkpoint["alignment_config"] == plan["neural_alignment"], "Embedded neural architecture/adaptation differs")
                require(np.array_equal(checkpoint["source"].numpy(), source), "Neural source tensor differs from isolated fold reference")
                require(all(torch.isfinite(value).all().item() for value in checkpoint["state_dict"].values()),
                        "Neural learned tensors contain nonfinite values")
                training_path = artifact.with_suffix(".training.json")
                training = read_json(training_path)
                input_hash(training_path, ledger)
                info = training["training_info"]
                require(info["fitted_subjects"] == fit_subjects and info["normalization_subjects"] == fit_subjects
                        and info["pretraining_subjects"] == (fit_subjects if recipe == "masked_ssl" else [])
                        and info["heldout_subject"] == heldout, "Training log population differs")
                require(info["seed"] == seed and info["epochs"] == epoch and info["recipe"] == recipe
                        and info["n_fitting_windows"] == len(fit)
                        and [entry["epoch"] for entry in training["history"]] == list(range(15, epoch + 1, 15)),
                        "Training log differs from fixed fitting schedule")
                embedded_checks.append(dict(artifact=str(artifact), stage="cv_fit", heldout_subject=heldout,
                                            all_seven_fields_match=True, source_tensor_matches=True,
                                            actual_fitting_rows_match=True))
                del checkpoint
            with np.load(prediction_path, allow_pickle=False) as archive:
                probability = np.array(archive["branch_probabilities"], dtype=np.float64, copy=True)
                assert_probability_array(probability, (3, 2, len(held), 3), str(prediction_path))
                require(np.array_equal(archive["row_indices"], held), "Neural prediction row order differs")
                require(np.array_equal(archive["subject"], meta["subject"][held]), "Neural prediction participants differ")
                require(np.array_equal(archive["seed_pairs"], np.array(SEED_PAIRS)), "Neural prediction seed order differs")
            branches[recipe][:, :, held, :] = probability
            filled[recipe][held] += 1
            all_manifests.extend(manifests)
    require(all((counts == 1).all() for counts in filled.values()), "OOF rows are omitted or duplicated")

    integrity_path = classical / "cv_integrity.json"
    integrity = read_json(integrity_path)
    input_hash(integrity_path, ledger)
    require(integrity["complete"] is True and integrity["integrity_checked_at_completion"] is True,
            "Classical completion integrity record is missing")
    for field, wanted in (("plan_sha256", plan_hash), ("metadata_sha256", metadata_hash),
                          ("raw_covariances_sha256", raw_hash),
                          ("predictions_sha256", input_hash(classical / "cv_predictions.npz", ledger)),
                          ("summary_sha256", input_hash(classical / "cv_summary.json", ledger))):
        require(integrity[field] == wanted, "Classical integrity mismatch: " + field)
    summary = read_json(classical / "cv_summary.json")
    require(summary["stage"] == "cv" and summary["complete"] is True
            and summary["plan_sha256"] == plan_hash and summary["warmup_test_role_retired"] is True,
            "Classical CV summary is incomplete or belongs to another plan")
    require(sorted(item["fold"] for item in summary["folds"]) == list(PUBLIC_SUBJECTS), "Classical folds differ")
    classical_manifests = [classical / ("hold_" + heldout) / (branch + ".provenance.json")
                          for heldout in PUBLIC_SUBJECTS for branch in ("ridge", "tangent")]
    classical_audit = check_training_artifacts(classical_manifests)
    require(audit_signature(classical_audit) == audit_signature(summary["no_heldout_leakage"]),
            "Classical completion audit differs from actual artifacts")
    for index, (heldout, branch) in enumerate((h, b) for h in PUBLIC_SUBJECTS for b in ("ridge", "tangent")):
        fit = np.flatnonzero(meta["subject"] != heldout)
        config = dict(plan["classical_definition"], branch=branch, plan_sha256=plan_hash)
        assert_provenance(classical_audit["artifacts"][index], heldout=heldout, fit_rows=fit,
                          expected_fields={"fitted_subjects", "normalization_subjects", "reference_subjects", "feature_scale_subjects"},
                          config=config, seed=0, epoch=None)
        artifact = classical / ("hold_" + heldout) / (branch + ".pt")
        checkpoint = torch.load(artifact, weights_only=True, map_location="cpu")
        require(checkpoint["provenance"] == read_json(classical_manifests[index])["provenance"],
                "Embedded classical provenance differs")
        require(checkpoint["type"] == ("linear" if branch == "ridge" else "riemannian_tangent"), "Classical branch type differs")
        require(checkpoint["center_kwargs"] == plan["classical_definition"]["center_kwargs"]
                and checkpoint["temperature"] == .125 and checkpoint["shrinkage"] == .05,
                "Classical adaptation/calibration differs")
        require(checkpoint["views"] == [{"crop": [0, 240]}, {"crop": [240, 480]}], "Classical temporal views differ")
        for field in ("scaler_mean", "scaler_scale", "coef", "intercept", "source_cov", "source_logcov", "source_logpower"):
            require(torch.isfinite(checkpoint[field]).all().item(), "Nonfinite classical learned tensor: " + field)
        if branch == "tangent":
            require(torch.isfinite(checkpoint["tangent_origin"]).all().item()
                    and torch.isfinite(checkpoint["tangent_inverse_sqrt"]).all().item(), "Nonfinite tangent origin")
        embedded_checks.append(dict(artifact=str(artifact), stage="cv_fit", heldout_subject=heldout,
                                    all_seven_fields_match=True, actual_fitting_rows_match=True))
        del checkpoint
    all_manifests.extend(classical_manifests)
    with np.load(classical / "cv_predictions.npz", allow_pickle=False) as archive:
        require(np.array_equal(archive["row_indices"], np.arange(N_ROWS)), "Classical OOF row order differs")
        require(np.array_equal(archive["subjects"], meta["subject"]), "Classical OOF participant IDs differ")
        ridge = np.array(archive["ridge_probabilities"], np.float64, copy=True)
        tangent = np.array(archive["tangent_probabilities"], np.float64, copy=True)
    assert_probability_array(ridge, (N_ROWS, 3), "ridge")
    assert_probability_array(tangent, (N_ROWS, 3), "tangent")

    audit = check_training_artifacts(all_manifests)
    require(audit["n_artifacts"] == 270, "Expected 10 references, 240 neural fits, and 20 classical fits")
    require(len({x["artifact_path"] for x in audit["artifacts"]}) == 270, "Duplicate artifact path")
    for record in audit["artifacts"]:
        ledger[record["artifact_path"]] = record["artifact_sha256"]
        ledger[record["manifest_path"]] = record["manifest_sha256"]
    audit.update(schema="track2-v5-complete-cv-isolation-audit-v1", plan_sha256=plan_hash,
                 n_reference_artifacts=10, n_neural_artifacts=240, n_classical_artifacts=20,
                 fitting_fields_checked=list(SUBJECT_FIELDS), actual_row_partitions=reference_details,
                 embedded_checkpoint_checks=embedded_checks, all_actual_fit_rows_match=True,
                 scoring_performed=False,
                 interpretation="Stage-specific CV isolation; P04/P09 excluded from fitting only in their own held-out folds. The original runtime checks are separate.")
    mixtures = {("baseline_v3" if recipe == "baseline" else recipe):
                .45 * probability[:, 0] + .45 * probability[:, 1] + .1 * ridge[None]
                for recipe, probability in branches.items()}
    mixtures["riemannian_tangent"] = .45 * branches["baseline"][:, 0] + .45 * branches["baseline"][:, 1] + .1 * tangent[None]
    for name, value in mixtures.items():
        assert_probability_array(value, (3, N_ROWS, 3), name)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + "-assembling-", dir=out.parent))
    try:
        outputs = {}
        for name in CANDIDATES:
            target = staging / (name + ".npz")
            np.savez_compressed(target, probabilities=mixtures[name], y_true=meta["y"],
                                subject=meta["subject"], session=meta["session"], interface=meta["interface"],
                                row_indices=np.arange(N_ROWS), seed_pairs=np.array(SEED_PAIRS))
            outputs[name] = dict(path=str(out / target.name), sha256=sha256_file(target))
        audit_path = staging / "stage_specific_leakage_check.json"
        write_json_exclusive(audit_path, audit)
        manifest = dict(schema="track2-v5-complete-probability-assembly-v1", complete=True,
                        assembled_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_hash,
                        metadata_sha256=metadata_hash, raw_covariances_sha256=raw_hash,
                        candidates=list(CANDIDATES), seed_pairs=[list(x) for x in SEED_PAIRS],
                        n_rows=N_ROWS, n_subjects=10, n_sessions=14, n_cells=len(cells),
                        n_artifacts=270, n_neural_checkpoints=240, n_classical_checkpoints=20,
                        n_covariance_references=10, all_actual_fit_rows_match=True,
                        warmup_test_role_retired=True, scoring_performed=False, selection_performed=False,
                        arithmetic="Fixed .45/.45/.10 probability mixture in float64; three separate paired replicates; deterministic classical branch shared across replicates",
                        inputs_sha256=ledger, prediction_files=outputs,
                        stage_specific_audit=dict(path=str(out / audit_path.name), sha256=sha256_file(audit_path)),
                        code_sha256={str(path): sha256_file(path) for path in (
                            Path(__file__), ROOT / "v5_work/protocol_v5.py", ROOT / "v5_work/analyze_cv.py")})
        write_json_exclusive(staging / "assembly_manifest.json", manifest)
        require(not out.exists(), "Output panel appeared concurrently; refusing replacement")
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps(dict(complete=True, output=str(out), n_candidates=5, n_rows=N_ROWS,
                          n_artifacts=270, scoring_performed=False, plan_sha256=plan_hash)), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "v5_work/plan.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--fold-cache", type=Path, default=ROOT / "v5_work/fold_cache")
    parser.add_argument("--neural", type=Path, default=ROOT / "v5_work/neural_results")
    parser.add_argument("--classical", type=Path, default=ROOT / "v5_work/classical_results")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/cv_predictions")
    args = parser.parse_args()
    assemble(plan_path=args.plan, cache=args.cache, fold_cache=args.fold_cache,
             neural=args.neural, classical=args.classical, out=args.out)
