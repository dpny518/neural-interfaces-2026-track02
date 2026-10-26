"""Prepare independently fitted, causal neural covariance caches for v5 LOSO.

Only participant IDs and stream ordering metadata are loaded. Labels are never
read. Every source covariance is fitted after an exact nine-subject assertion;
held-out windows are transformed using that fitted prior in separate, reset
chronological streams. No held-out statistic is retained across sessions.

Run only after the raw window-local feature cache has finished writing. The
parent orchestration is responsible for this barrier. Completed per-fold
checksums allow training runners to verify every byte before using a cache.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bci_work")]
from neural_cov import fit_source_reference
from train_neural_cov import stream_covariances
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, assert_stage_subjects, make_provenance,
    register_training_artifact, check_training_artifacts, sha256_file,
    write_json_exclusive,
)

FROZEN_PLAN_SHA256 = "ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59"


def read_metadata(path):
    fields = ("subject", "session", "interface", "stream_id", "record_id", "onset")
    with np.load(path, allow_pickle=False) as archive:
        metadata = {key: np.array(archive[key], copy=True) for key in fields}
    assert "y" not in metadata
    assert len(metadata["subject"]) == 2655
    assert_stage_subjects(metadata["subject"], stage="selection")
    assert len(np.unique(metadata["stream_id"])) == 14
    assert len(set(zip(metadata["subject"], metadata["session"], metadata["interface"]))) == 28
    for stream in np.unique(metadata["stream_id"]):
        rows = np.flatnonzero(metadata["stream_id"] == stream)
        assert len(set(zip(metadata["subject"][rows], metadata["session"][rows]))) == 1
        ordering = list(zip(metadata["record_id"][rows].tolist(), metadata["onset"][rows].tolist()))
        assert ordering == sorted(ordering), "Session windows are not chronological"
        assert len(set(ordering)) == len(ordering), "Duplicate window onset in a recording"
    return metadata


def _save_npy(path, array):
    """Publish a complete NPY file without exposing a partial final filename."""
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def _save_reference(path, source, fit_rows, held_rows):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, source=source, fit_rows=fit_rows, held_rows=held_rows)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def verify_complete_fold(folder, plan_hash, metadata_hash, raw_hash):
    marker = folder / "cache_checksums.json"
    record = json.loads(marker.read_text())
    assert record["plan_sha256"] == plan_hash
    assert record["metadata_sha256"] == metadata_hash
    assert record["raw_covariances_sha256"] == raw_hash
    for name, digest in record["files"].items():
        assert sha256_file(folder / name) == digest, f"Cache changed: {folder / name}"
    check_training_artifacts([folder / "reference.provenance.json"])
    return record


def prepare(cache, out, plan_path, expected_plan_sha256=FROZEN_PLAN_SHA256, threads=1):
    threadpool_limits(limits=threads)
    import torch
    torch.set_num_threads(threads)
    cache, out, plan_path = map(Path, (cache, out, plan_path))
    plan_hash = sha256_file(plan_path)
    assert plan_hash == expected_plan_sha256, "The pre-fitting fixed plan changed"
    plan = json.loads(plan_path.read_text())
    assert plan["warmup_test_role_retired"] is True
    assert tuple(plan["public_subjects"]) == PUBLIC_SUBJECTS
    alignment = dict(plan["neural_alignment"])
    metadata_path = cache / "metadata.npz"
    raw_path = cache / "temporal_band_covs.npy"
    metadata = read_metadata(metadata_path)
    metadata_hash, raw_hash = sha256_file(metadata_path), sha256_file(raw_path)
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (2655, 14, 41, 41)
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    started = time.perf_counter()
    for heldout in PUBLIC_SUBJECTS:
        folder = out / heldout
        marker = folder / "cache_checksums.json"
        if marker.is_file():
            summaries[heldout] = verify_complete_fold(folder, plan_hash, metadata_hash, raw_hash)
            print(json.dumps({"heldout_subject": heldout, "status": "verified_existing_complete_fold"}), flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Incomplete existing fold cache is not overwritten: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        fold_started = time.perf_counter()
        fit_rows = np.flatnonzero(metadata["subject"] != heldout).astype(np.int64)
        held_rows = np.flatnonzero(metadata["subject"] == heldout).astype(np.int64)
        fit_subjects = assert_stage_subjects(metadata["subject"][fit_rows], stage="cv_fit", heldout_subject=heldout)
        assert_stage_subjects(metadata["subject"][held_rows], stage="cv_evaluate", heldout_subject=heldout)
        assert not np.intersect1d(fit_rows, held_rows).size
        assert np.array_equal(np.sort(np.concatenate((fit_rows, held_rows))), np.arange(len(raw)))
        # The first fitted operation receives only asserted fitting rows.
        fit_raw = np.asarray(raw[fit_rows])
        source = fit_source_reference(fit_raw, metadata["stream_id"][fit_rows],
                                      normalization=alignment["normalization"])
        assert source.shape == (14, 41, 41) and np.isfinite(source).all()
        reference_path = folder / "reference.npz"
        _save_reference(reference_path, source, fit_rows, held_rows)
        provenance = make_provenance(
            stage="cv_fit", fitted_subjects=metadata["subject"][fit_rows], heldout_subject=heldout,
            pretraining_subjects=(), normalization_subjects=metadata["subject"][fit_rows],
            reference_subjects=metadata["subject"][fit_rows], feature_scale_subjects=(),
            batchnorm_subjects=(), checkpoint_selection_subjects=(), seed=0, epoch=None,
            row_indices=fit_rows,
            config={"artifact_kind": "neural_fold_source_and_row_partition",
                    "plan_sha256": plan_hash, "alignment": alignment,
                    "source_fit": "equal-session arithmetic mean of training-only normalized covariance",
                    "metadata_sha256": metadata_hash, "raw_covariances_sha256": raw_hash,
                    "labels_loaded": False, "feature_scaler_fitted": False,
                    "normalization": "fixed window-local broadband trace plus training-only covariance prior"},
        )
        manifest = register_training_artifact(reference_path, provenance)
        aligned_fit = stream_covariances(fit_raw, metadata["stream_id"][fit_rows], source, alignment)
        del fit_raw
        assert aligned_fit.shape == (len(fit_rows), 14, 41, 41)
        assert np.isfinite(aligned_fit).all()
        _save_npy(folder / "fit_covariances.npy", aligned_fit)
        del aligned_fit
        # This separate call creates fresh aligners for each held-out session.
        aligned_held = stream_covariances(np.asarray(raw[held_rows]), metadata["stream_id"][held_rows], source, alignment)
        assert aligned_held.shape == (len(held_rows), 14, 41, 41)
        assert np.isfinite(aligned_held).all()
        _save_npy(folder / "held_covariances.npy", aligned_held)
        del aligned_held
        record = {
            "schema": "track2-v5-fold-covariance-cache-v1", "heldout_subject": heldout,
            "plan_sha256": plan_hash, "metadata_sha256": metadata_hash,
            "raw_covariances_sha256": raw_hash, "fit_subjects": fit_subjects,
            "n_fit": len(fit_rows), "n_held": len(held_rows), "alignment": alignment,
            "labels_loaded": False, "causal_session_transform": True,
            "files": {name: sha256_file(folder / name) for name in (
                "fit_covariances.npy", "held_covariances.npy", "reference.npz", "reference.provenance.json")},
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        check_training_artifacts([manifest])
        write_json_exclusive(marker, record)
        summaries[heldout] = record
        print(json.dumps({"heldout_subject": heldout, "n_fit": len(fit_rows), "n_held": len(held_rows),
                          "elapsed_seconds": record["elapsed_seconds"], "status": "complete"}), flush=True)
    result = {
        "schema": "track2-v5-all-fold-covariance-caches-v1", "all_passed": True,
        "plan_sha256": plan_hash, "metadata_sha256": metadata_hash, "raw_covariances_sha256": raw_hash,
        "n_folds": len(summaries), "labels_loaded": False,
        "folds": summaries,
        "files": {str(path.relative_to(out)): sha256_file(path)
                  for heldout in PUBLIC_SUBJECTS for path in sorted((out / heldout).iterdir()) if path.is_file()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    global_marker = out / "cache_checksums.json"
    if global_marker.is_file():
        old = json.loads(global_marker.read_text())
        assert old["plan_sha256"] == plan_hash and old["files"] == result["files"]
    else:
        write_json_exclusive(global_marker, result)
    print(json.dumps({"all_passed": True, "n_folds": len(summaries),
                      "elapsed_seconds": result["elapsed_seconds"], "plan_sha256": plan_hash}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/fold_cache")
    parser.add_argument("--plan", type=Path, default=ROOT / "v5_work/plan.json")
    parser.add_argument("--expected-plan-sha256", default=FROZEN_PLAN_SHA256)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    prepare(args.cache, args.out, args.plan, args.expected_plan_sha256, args.threads)
