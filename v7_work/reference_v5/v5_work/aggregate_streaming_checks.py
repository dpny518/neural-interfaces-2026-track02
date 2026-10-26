"""Certify the complete v5 panel after five independent streaming replay shards.

This changes verification parallelism only. It reads no selection result and
computes no accuracy score. Every report, fitting artifact, runtime input and
saved prediction is checked again before full_panel_certified can be true.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from v5_work.pack_runtime import CANDIDATES, build_runtime_source
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, SEED_PAIRS, check_training_artifacts, sha256_file,
    write_json_exclusive,
)
from v5_work.verify_cv_streaming import (
    FileLedger, PLAN_SHA256, PROBABILITY_ATOL, PROBABILITY_RTOL,
    load_expected, load_metadata, require,
)


def resolve_recorded(value, anchor=ROOT):
    path = Path(value)
    choices = [Path(anchor) / path.name]
    if not path.is_absolute():
        choices += [Path(anchor) / path, ROOT / path]
    for prefix in ("v5_work", "v5_deliverables"):
        if prefix in path.parts:
            choices.append(ROOT.joinpath(*path.parts[path.parts.index(prefix):]))
    choices.append(path)
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise FileNotFoundError(value)


def stable_record(record):
    return {key: value for key, value in record.items()
            if key not in {"artifact_path", "manifest_path"}}


def check_prediction_archive(path, candidate, metadata, expected, ledger, digest):
    ledger.add(path, digest)
    with np.load(path, allow_pickle=False) as archive:
        require(np.array_equal(archive["row_indices"], np.arange(2655)),
                candidate + ": replay row coverage differs")
        require(np.array_equal(archive["replicate_indices"], [0, 1, 2])
                and np.array_equal(archive["seed_pairs"], np.asarray(SEED_PAIRS)),
                candidate + ": replicate/seed coverage differs")
        for key, metadata_key in (("y_true", "y"), ("subject", "subject"),
                                  ("session", "session"), ("interface", "interface"),
                                  ("stream_id", "stream_id"), ("record_id", "record_id"),
                                  ("onset", "onset")):
            require(np.array_equal(archive[key], metadata[metadata_key]),
                    candidate + ": replay metadata order differs: " + key)
        actual = np.array(archive["probabilities"], copy=True)
        labels = np.array(archive["y_pred"], copy=True)
        cached = np.array(archive["expected_probabilities"], copy=True)
        cached_labels = np.array(archive["expected_y_pred"], copy=True)
    require(actual.shape == cached.shape == (3, 2655, 3)
            and labels.shape == cached_labels.shape == (3, 2655),
            candidate + ": prediction array shape differs")
    require(np.isfinite(actual).all() and np.all(actual >= 0.)
            and np.allclose(actual.sum(-1), 1., rtol=0., atol=1e-12),
            candidate + ": invalid streamed probabilities")
    require(np.array_equal(cached, expected), candidate + ": cached reference changed")
    require(np.issubdtype(labels.dtype, np.integer)
            and np.array_equal(labels, actual.argmax(-1))
            and np.array_equal(labels, expected.argmax(-1))
            and np.array_equal(cached_labels, expected.argmax(-1)),
            candidate + ": streamed labels differ from actual/merged probabilities")
    require(np.allclose(actual, expected, rtol=PROBABILITY_RTOL, atol=PROBABILITY_ATOL),
            candidate + ": streamed probabilities exceed the fixed tolerance")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=ROOT / "v5_work/cv_streaming_shards")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "v5_work/cv_predictions")
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--plan", type=Path, default=ROOT / "v5_work/plan.json")
    parser.add_argument("--kit", type=Path, default=ROOT / "reference_audit/2026-competition")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/cv_streaming_check")
    args = parser.parse_args()
    started = time.perf_counter()
    ledger = FileLedger()
    ledger.add(args.plan, PLAN_SHA256)
    ledger.add(Path(__file__))
    ledger.add(ROOT / "v5_work/verify_cv_streaming.py")
    ledger.add(ROOT / "v5_work/pack_runtime.py")
    ledger.add(ROOT / "v5_work/protocol_v5.py")
    metadata, _ = load_metadata(args.cache, ledger)
    expected, assembly = load_expected(args.predictions_dir, list(CANDIDATES),
                                       metadata, PLAN_SHA256, ledger)
    require(assembly["metadata_sha256"] == ledger.add(args.cache / "metadata.npz")
            and assembly["raw_covariances_sha256"] == ledger.add(args.cache / "temporal_band_covs.npy"),
            "OOF assembly and replay data differ")
    ledger.add(args.cache / "windows.npy")
    runtime_source, runtime_files = build_runtime_source()
    runtime_hash = hashlib.sha256(runtime_source.encode()).hexdigest()
    for path, digest in runtime_files.items():
        ledger.add(ROOT / path, digest)
    evaluator_path = args.kit / "benchmark_utils/streaming.py"
    ledger.add(evaluator_path)
    all_jobs, shard_records, artifact_records, prediction_artifacts = [], [], {}, {}
    for candidate in CANDIDATES:
        directory = args.shards_dir / candidate
        report_path = directory / "streaming_replay_check.json"
        report_hash = ledger.add(report_path)
        report = json.loads(report_path.read_text())
        require(report.get("schema") == "track2-v5-official-streaming-replay-v1",
                candidate + ": unknown replay report schema")
        require(report["checks_passed"] is True and report["smoke_only"] is False
                and report["full_panel_certified"] is False and report["full_coverage"] is False,
                candidate + ": expected a passing, complete-candidate shard")
        require(report["candidates"] == [candidate]
                and report["requested_subjects"] == list(PUBLIC_SUBJECTS)
                and report["replicate_indices"] == [0, 1, 2]
                and report["unique_windows"] == 2655, candidate + ": incomplete shard coverage")
        require(report["plan_sha256"] == PLAN_SHA256
                and report["runtime_source_sha256"] == runtime_hash
                and report["runtime_source_files"] == runtime_files,
                candidate + ": frozen plan/runtime differs")
        require(report["evaluator"] == "benchmark_utils.streaming.predict_streams"
                and report["expected_source"] == "complete hash-bound merged OOF"
                and report["probability_observer"] ==
                    "read-only wrapper of predict_proba return; deployed predict is unchanged",
                candidate + ": replay/evidence semantics differ")
        require(report["probability_atol"] == PROBABILITY_ATOL
                and report["probability_rtol"] == PROBABILITY_RTOL,
                candidate + ": probability tolerance changed")
        for field in ("exact_predicted_labels", "streaming_cell_metric_identical_by_exact_labels",
                      "adaptive_state_isolated", "current_window_only_model_input",
                      "heldout_excluded_from_every_fitting_stage", "warmup_test_role_retired"):
            require(report[field] is True, candidate + ": failed evidence field " + field)
        for field in ("scores_computed", "selection_performed", "training_performed"):
            require(report[field] is False, candidate + ": verifier performed an unauthorized stage")
        require(report["total_predict_calls"] == 7965
                and report["total_session_deepcopies"] == report["total_session_resets"] == 42,
                candidate + ": incorrect prediction/session counts")
        require(len(report["jobs"]) == 30, candidate + ": expected thirty fold/replicate jobs")
        # Deduplication avoids rehashing the same large public cache five times.
        for path, digest in report["files_unchanged"].items():
            ledger.add(resolve_recorded(path), digest)
        # Verify that the recorded runtime/evaluator/cache files were actual
        # replay inputs, not merely equal-valued strings in the top-level report.
        input_hashes = report["files_unchanged"]
        for required_path in (args.plan, args.cache / "metadata.npz", args.cache / "windows.npy",
                              args.cache / "temporal_band_covs.npy", evaluator_path,
                              ROOT / "v5_work/verify_cv_streaming.py", ROOT / "v5_work/pack_runtime.py"):
            require(any(resolve_recorded(path) == required_path.resolve()
                        and digest == ledger.add(required_path)
                        for path, digest in input_hashes.items()),
                    candidate + ": a required replay input was not hash-bound")
        for record in report["provenance_artifacts"]:
            path = resolve_recorded(record["manifest_path"])
            ledger.add(path, record["manifest_sha256"])
            artifact = resolve_recorded(record["artifact_path"])
            ledger.add(artifact, record["artifact_sha256"])
            if path in artifact_records:
                require(stable_record(record) == stable_record(artifact_records[path]),
                        "Shared fitting artifact has inconsistent shard provenance")
            artifact_records[path] = record
        require(set(report["prediction_artifacts"]) == {candidate}, "Unexpected shard predictions")
        prediction = report["prediction_artifacts"][candidate]
        path = resolve_recorded(prediction["path"], directory)
        require(path == (directory / (candidate + "_streaming.npz")).resolve(),
                candidate + ": unexpected streamed prediction artifact path")
        actual = check_prediction_archive(path, candidate, metadata, expected[candidate],
                                          ledger, prediction["sha256"])
        prediction_artifacts[candidate] = dict(path=str(path), sha256=ledger.add(path))
        observed_pairs = set()
        for job in report["jobs"]:
            subject, replicate = job["subject"], job["replicate"]
            require(subject in PUBLIC_SUBJECTS and replicate in (0, 1, 2), "Unexpected replay job")
            require((subject, replicate) not in observed_pairs, "Duplicate replay job")
            observed_pairs.add((subject, replicate))
            rows = np.flatnonzero(metadata["subject"] == subject)
            sessions = list(dict.fromkeys(metadata["stream_id"][rows]))
            lengths = [int(np.sum(metadata["stream_id"][rows] == stream)) for stream in sessions]
            require(job["candidate"] == candidate and job["seed_pair"] == list(SEED_PAIRS[replicate])
                    and job["n_windows"] == job["batch1_predict_calls"] == len(rows)
                    and job["n_sessions"] == job["deepcopy_calls"] == job["reset_calls"] == len(sessions)
                    and job["session_lengths"] == lengths, "Replay job coverage/counts differ")
            for field in ("template_unchanged", "no_shared_session_buffers",
                          "one_adaptive_update_per_window", "probability_tolerance_passed"):
                require(job[field] is True, "Replay job failed " + field)
            require(job["prediction_mismatches"] == 0 and job["mismatch_row_indices"] == [],
                    "Replay job has label mismatches")
            maximum = float(np.abs(actual[replicate, rows] - expected[candidate][replicate, rows]).max())
            require(job["max_probability_absolute_difference"] == maximum,
                    "Replay probability-difference report differs from saved arrays")
            recipe = "baseline" if candidate in {"baseline_v3", "riemannian_tangent"} else candidate
            suffixes = [(recipe, "hold_" + subject, f"seed{seed}_epoch{epoch}.pt")
                        for seed, epoch in zip(SEED_PAIRS[replicate], (75, 20))]
            suffixes.append(("hold_" + subject,
                             "tangent.pt" if candidate == "riemannian_tangent" else "ridge.pt"))
            require(len(job["member_checkpoints"]) == 3, "A replay member is missing")
            for member, suffix in zip(job["member_checkpoints"], suffixes):
                member_path = resolve_recorded(member)
                require(member_path.parts[-len(suffix):] == suffix,
                        "Replay member differs from the frozen seed/recipe/fold")
                manifest = member_path.with_suffix(".provenance.json")
                require(manifest in artifact_records, "Replay member is absent from fitting provenance")
                require(artifact_records[manifest]["heldout_subject"] == subject,
                        "Replay member used another fitting fold")
            all_jobs.append(job)
        require(observed_pairs == {(subject, rep) for subject in PUBLIC_SUBJECTS for rep in range(3)},
                candidate + ": fold/replicate panel incomplete")
        shard_records.append(dict(candidate=candidate, path=str(report_path), sha256=report_hash,
                                  prediction_artifact=prediction_artifacts[candidate],
                                  n_jobs=30, n_predict_calls=7965, n_session_copies=42))
        print(json.dumps(dict(candidate=candidate, artifact_hashes_valid=True,
                              predicted_labels_exact=True, probabilities_within_tolerance=True,
                              n_windows=2655, n_replicates=3)), flush=True)
    require(len(artifact_records) == 270, "Expected 240 neural fits, 20 classical fits and 10 references")
    provenance = check_training_artifacts(sorted(artifact_records))
    require(provenance["all_passed"] is True and provenance["n_artifacts"] == 270,
            "Independent stage-specific provenance recheck failed")
    for record in provenance["artifacts"]:
        path = Path(record["manifest_path"]).resolve()
        require(stable_record(record) == stable_record(artifact_records[path]),
                "Replay-recorded fitting provenance changed")
    total_calls = sum(job["batch1_predict_calls"] for job in all_jobs)
    total_copies = sum(job["deepcopy_calls"] for job in all_jobs)
    total_resets = sum(job["reset_calls"] for job in all_jobs)
    require(len(all_jobs) == 150 and total_calls == 39825 and total_copies == total_resets == 210,
            "Full-panel prediction/reset counts differ")
    files = ledger.verify_unchanged()
    result = dict(
        schema="track2-v5-official-streaming-panel-v1", checks_passed=True,
        full_panel_certified=True, full_coverage=True, smoke_only=False,
        plan_sha256=PLAN_SHA256, runtime_source_sha256=runtime_hash,
        runtime_source_files=runtime_files, evaluator="benchmark_utils.streaming.predict_streams",
        verification_parallelism="five independently replayed candidate shards; one thread per shard",
        probability_observer="read-only wrapper of predict_proba return; deployed predict is unchanged",
        expected_source="complete hash-bound merged OOF",
        probability_atol=PROBABILITY_ATOL, probability_rtol=PROBABILITY_RTOL,
        exact_predicted_labels=True, streaming_cell_metric_identical_by_exact_labels=True,
        scores_computed=False, selection_performed=False, training_performed=False,
        public_subjects=list(PUBLIC_SUBJECTS), candidates=list(CANDIDATES),
        replicate_indices=[0, 1, 2], seed_pairs=[list(pair) for pair in SEED_PAIRS],
        unique_windows=2655, n_cells=28, n_jobs=150,
        total_predict_calls=total_calls, total_session_deepcopies=total_copies,
        total_session_resets=total_resets, adaptive_state_isolated=True,
        current_window_only_model_input=True, heldout_excluded_from_every_fitting_stage=True,
        warmup_test_role_retired=True, provenance_artifacts=provenance["artifacts"],
        files_unchanged=files, prediction_artifacts=prediction_artifacts,
        shards=shard_records, jobs=all_jobs,
        aggregate_elapsed_seconds=time.perf_counter() - started,
        summed_inference_seconds=sum(job["elapsed_seconds"] for job in all_jobs),
    )
    write_json_exclusive(args.out / "streaming_replay_check.json", result)
    print(json.dumps({key: result[key] for key in (
        "full_panel_certified", "exact_predicted_labels", "total_predict_calls",
        "total_session_deepcopies", "aggregate_elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
