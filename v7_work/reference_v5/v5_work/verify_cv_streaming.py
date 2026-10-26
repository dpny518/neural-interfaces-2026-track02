"""Verify v5 OOF predictions by replaying raw EEG through the official harness.

The full audit covers 5 candidates x 3 paired replicates x all 2,655 windows.
The official evaluator makes a deepcopy, resets it at each participant/session,
and calls the unchanged deployed predict method with one window. A read-only
observer captures predict_proba's return value from that same call; adaptation
is never run twice to obtain probabilities. No score or selection is computed.

The optional smoke mode uses a bounded prefix of each requested session. When
merged OOF files do not yet exist, its reference is cached neural OOF predictions
plus the saved classical member applied to cached window-local covariances.
That narrower check is explicitly not a complete merged-OOF certification.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from v5_work.pack_runtime import CANDIDATES, VIEWS, load_runtime_module, pack_weights
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, SEED_PAIRS, SUBJECT_FIELDS, assert_stage_subjects,
    check_training_artifacts, json_sha256, sha256_file, validate_provenance,
    write_json_exclusive,
)

PLAN_SHA256 = "ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59"
PROBABILITY_ATOL = 2e-6
PROBABILITY_RTOL = 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


class FileLedger:
    """Bind every evidence input to bytes, then check that none changed."""

    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        if path not in self.files:
            self.files[path] = sha256_file(path)
        digest = self.files[path]
        require(expected is None or digest == expected, f"SHA256 mismatch: {path}")
        return digest

    def verify_unchanged(self):
        for path, digest in self.files.items():
            require(sha256_file(path) == digest, f"Audit input changed during replay: {path}")
        return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest
                for path, digest in sorted(self.files.items())}


def _recorded_path(value, anchor):
    path = Path(value)
    # An unpacked reproduction tree must use its own adjacent/current-root
    # artifacts even when the original workspace still exists elsewhere.
    candidates = [Path(anchor) / path.name]
    if not path.is_absolute():
        candidates += [Path(anchor) / path, ROOT / path]
    for prefix in ("v5_work", "v5_deliverables"):
        if prefix in path.parts:
            candidates.append(ROOT.joinpath(*path.parts[path.parts.index(prefix):]))
    candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_metadata(cache, ledger):
    ledger.add(cache / "metadata.npz")
    with np.load(cache / "metadata.npz", allow_pickle=False) as archive:
        meta = {key: np.array(archive[key], copy=True) for key in archive.files}
    required = {"subject", "session", "interface", "stream_id", "record_id", "onset", "y", "split"}
    require(required.issubset(meta), "Missing streaming metadata")
    assert_stage_subjects(meta["subject"], stage="selection")
    require(len(meta["y"]) == 2655 and set(meta["split"]) == {"development"},
            "Expected the frozen all-ten development cache")
    require(set(meta["interface"]) == {"Graz", "BrainHero"}, "Unexpected interface")
    require(len(set(meta["stream_id"])) == 14, "Expected fourteen participant sessions")
    cells = set(zip(meta["subject"], meta["session"], meta["interface"]))
    require(len(cells) == 28, "Expected twenty-eight metric cells")
    for stream in np.unique(meta["stream_id"]):
        rows = np.flatnonzero(meta["stream_id"] == stream)
        require(np.all(np.diff(rows) == 1), "Session rows must be contiguous")
        require(len(set(zip(meta["subject"][rows], meta["session"][rows]))) == 1,
                "A stream must identify one participant/session")
    ledger.add(cache / "dataset_info.json")
    info = json.loads((cache / "dataset_info.json").read_text())
    require(len(info["ch_names"]) == 41, "Expected the original 41-channel montage")
    return meta, info


def load_expected(folder, candidates, metadata, plan_hash, ledger):
    """Require the completed, hash-bound OOF assembly, never a partial panel."""
    manifest_path = folder / "assembly_manifest.json"
    ledger.add(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "track2-v5-complete-probability-assembly-v1"
            and manifest.get("complete") is True, "Complete OOF assembly is required")
    require(manifest["plan_sha256"] == plan_hash, "OOF assembly used another plan")
    require(manifest["n_rows"] == len(metadata["y"]) and manifest["n_cells"] == 28,
            "OOF assembly coverage differs")
    require(set(manifest["prediction_files"]) == set(CANDIDATES),
            "The complete five-candidate panel is required")
    audit_record = manifest["stage_specific_audit"]
    audit_path = _recorded_path(audit_record["path"], folder)
    ledger.add(audit_path, audit_record["sha256"])
    audit = json.loads(audit_path.read_text())
    require(audit.get("all_passed") is True, "Assembly provenance audit did not pass")
    expected = {}
    for name in candidates:
        path = folder / (name + ".npz")
        record = manifest["prediction_files"][name]
        require(_recorded_path(record["path"], folder) == path.resolve(),
                "Prediction filename differs from its assembly manifest")
        ledger.add(path, record["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            for key, metadata_key in (("y_true", "y"), ("subject", "subject"),
                                      ("session", "session"), ("interface", "interface")):
                require(np.array_equal(archive[key], metadata[metadata_key]),
                        f"{name}: {key} order differs")
            require(np.array_equal(archive["row_indices"], np.arange(len(metadata["y"]))),
                    f"{name}: OOF row order differs")
            if "seed_pairs" in archive:
                require(np.array_equal(archive["seed_pairs"], np.asarray(SEED_PAIRS)),
                        f"{name}: replicate seeds differ")
            probabilities = np.array(archive["probabilities"], dtype=np.float64, copy=True)
        require(probabilities.shape == (3, len(metadata["y"]), 3)
                and np.isfinite(probabilities).all(), f"{name}: incomplete OOF probabilities")
        require(np.all(probabilities >= 0.) and np.allclose(probabilities.sum(-1), 1., atol=2e-6),
                f"{name}: invalid OOF probabilities")
        expected[name] = probabilities
    return expected, manifest


class FoldArtifacts:
    """Read-only fitting artifact loader with independent fold-isolation checks."""

    def __init__(self, args, plan, meta, info, ledger):
        self.args, self.plan, self.meta, self.info, self.ledger = args, plan, meta, info, ledger
        self.cache, self.references, self.records = {}, {}, {}

    def reference(self, heldout):
        if heldout in self.references:
            return self.references[heldout]
        folder = self.args.fold_cache / heldout
        marker_path = folder / "cache_checksums.json"
        self.ledger.add(marker_path)
        marker = json.loads(marker_path.read_text())
        require(marker["heldout_subject"] == heldout and marker["plan_sha256"] == PLAN_SHA256,
                "Fold cache identity differs")
        require(marker["metadata_sha256"] == self.ledger.add(self.args.cache / "metadata.npz"),
                "Fold metadata changed")
        require(marker["raw_covariances_sha256"] == self.ledger.add(self.args.cache / "temporal_band_covs.npy"),
                "Fold raw covariance cache changed")
        path = folder / "reference.npz"
        self.ledger.add(path, marker["files"][path.name])
        manifest_path = folder / "reference.provenance.json"
        self.ledger.add(manifest_path, marker["files"][manifest_path.name])
        record = check_training_artifacts([manifest_path])["artifacts"][0]
        with np.load(path, allow_pickle=False) as archive:
            source = np.array(archive["source"], copy=True)
            fit, held = archive["fit_rows"], archive["held_rows"]
        require(np.array_equal(fit, np.flatnonzero(self.meta["subject"] != heldout))
                and np.array_equal(held, np.flatnonzero(self.meta["subject"] == heldout)),
                "Fold cache row partition differs")
        self._check_provenance(record, heldout, fit)
        self.records[str(manifest_path)] = record
        self.references[heldout] = (source, fit, held, self.ledger.add(path))
        return self.references[heldout]

    @staticmethod
    def _check_provenance(record, heldout, fit):
        require(record["stage"] == "cv_fit" and record["heldout_subject"] == heldout,
                "Expected independently fitted LOSO artifact")
        expected = sorted(set(PUBLIC_SUBJECTS) - {heldout})
        require(record["fitted_subjects"] == expected, "Wrong fitting population")
        for field in SUBJECT_FIELDS:
            require(heldout not in record[field] and set(record[field]) <= set(expected),
                    f"Held-out participant entered {field}")
        require(record["checkpoint_selection_subjects"] == [],
                "Fixed epochs must not have checkpoint-selection participants")
        require(record["config"]["plan_sha256"] == PLAN_SHA256, "Checkpoint used another plan")
        require(record["row_indices_sha256"] == json_sha256(fit.tolist())
                and record["n_fitting_rows"] == len(fit), "Checkpoint fitting rows differ")
        return expected

    def load(self, path, heldout, fit):
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        manifest_path = path.with_suffix(".provenance.json")
        self.ledger.add(manifest_path)
        audit = check_training_artifacts([manifest_path])["artifacts"][0]
        require(Path(audit["artifact_path"]).resolve() == path, "Manifest refers to another artifact")
        self.ledger.add(path, audit["artifact_sha256"])
        self._check_provenance(audit, heldout, fit)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        embedded = validate_provenance(checkpoint["provenance"])
        require(all(embedded[key] == audit[key] for key in embedded),
                "Embedded fitting provenance differs from the registered manifest")
        self.records[str(manifest_path)] = audit
        self.cache[path] = checkpoint
        return checkpoint

    def pack(self, candidate, heldout, replicate):
        source, fit, held, reference_hash = self.reference(heldout)
        fitting_subjects = assert_stage_subjects(self.meta["subject"][fit],
                                               stage="cv_fit", heldout_subject=heldout)
        recipe = "baseline" if candidate in {"baseline_v3", "riemannian_tangent"} else candidate
        neural, paths = [], []
        for seed, epoch in zip(SEED_PAIRS[replicate], self.plan["fixed_epochs"]):
            path = self.args.neural_root / recipe / ("hold_" + heldout) / f"seed{seed}_epoch{epoch}.pt"
            checkpoint = self.load(path, heldout, fit)
            require(checkpoint["training_subjects"] == fitting_subjects
                    and checkpoint["heldout_subject"] == heldout, "Neural fitting population differs")
            require(checkpoint["seed"] == seed and checkpoint["epoch"] == epoch
                    and checkpoint["scheduler_T_max"] == 120, "Neural seed/epoch schedule differs")
            require(checkpoint["recipe"] == recipe
                    and checkpoint["model_config"] == self.plan["neural_definitions"]["model"]
                    and checkpoint["alignment_config"] == self.plan["neural_alignment"],
                    "Neural deploy configuration differs")
            require(np.array_equal(_array(checkpoint["source"]), source),
                    "Neural source differs from the isolated fold reference")
            provenance = checkpoint["provenance"]
            require(provenance["config"]["reference_sha256"] == reference_hash
                    and provenance["config"]["neural_definitions"] == self.plan["neural_definitions"],
                    "Neural recipe/reference is not the frozen definition")
            ssl_subjects = fitting_subjects if recipe == "masked_ssl" else []
            require(provenance["pretraining_subjects"] == ssl_subjects
                    and provenance["feature_scale_subjects"] == ssl_subjects,
                    "SSL fitting/scaling participants differ")
            for field in ("normalization_subjects", "reference_subjects", "batchnorm_subjects"):
                require(provenance[field] == fitting_subjects, "Neural fitting field differs: " + field)
            neural.append(checkpoint)
            paths.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        branch = "tangent" if candidate == "riemannian_tangent" else "ridge"
        path = self.args.classical_root / ("hold_" + heldout) / (branch + ".pt")
        classical = self.load(path, heldout, fit)
        require(classical["provenance"]["config"] == dict(self.plan["classical_definition"],
                branch=branch, plan_sha256=PLAN_SHA256), "Classical recipe differs from frozen plan")
        require(classical["center_kwargs"] == self.plan["classical_definition"]["center_kwargs"]
                and classical["temperature"] == .125 and classical["views"] == VIEWS,
                "Classical deployment configuration differs")
        for field in ("normalization_subjects", "reference_subjects", "feature_scale_subjects"):
            require(classical["provenance"][field] == fitting_subjects,
                    "Classical fitting field differs: " + field)
        require(classical["provenance"]["pretraining_subjects"] == []
                and classical["provenance"]["batchnorm_subjects"] == [],
                "Unexpected classical SSL/BN provenance")
        paths.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        packed = pack_weights(neural, classical, self.info["ch_names"], candidate=candidate,
                              training_subjects=fitting_subjects,
                              extra_metadata=dict(stage="cv_fit", heldout_subject=heldout,
                                                  replicate=replicate, plan_sha256=PLAN_SHA256))
        return packed, held, paths


def state_fingerprint(model):
    """Hash numerical model state and ordinary attributes without serialization IDs."""
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.nn.Module):
            update((type(value).__name__, value.training, value.state_dict()))
        elif isinstance(value, torch.Tensor):
            update(value.detach().cpu().contiguous().numpy())
        elif isinstance(value, np.ndarray):
            digest.update(b"array" + value.dtype.str.encode() + str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for element in value:
                update(element)
        elif isinstance(value, np.generic):
            update(value.item())
        elif value is None or isinstance(value, (str, bool, int, float)):
            digest.update(repr(value).encode())
        elif isinstance(value, (Path, torch.device)):
            update(str(value))
        elif hasattr(value, "__dict__"):
            update((type(value).__name__, vars(value)))
        else:
            raise TypeError("Unsupported state fingerprint value: " + str(type(value)))

    update(model)
    return digest.hexdigest()


def member_counts(model):
    return [int(getattr(member, "aligner", getattr(member, "transformer", None)).n_seen)
            for member in model.members]


def check_copy_isolation(template, clone):
    require(clone is not template and clone.n_seen == 0, "Invalid fresh session model")
    require(member_counts(clone) == [0, 0, 0], "Session member state was not reset")
    for original, copied in zip(template.members, clone.members):
        require(original is not copied, "A member is shared across session copies")
        stream_name = "aligner" if hasattr(original, "aligner") else "transformer"
        before, after = getattr(original, stream_name), getattr(copied, stream_name)
        require(before is not after, "An adaptive transformer is shared")
        require(not np.shares_memory(before.running_cov, after.running_cov),
                "Adaptive covariance state is shared across copies")
        if hasattr(original, "net"):
            for key, value in original.net.state_dict().items():
                require(value.data_ptr() != copied.net.state_dict()[key].data_ptr(),
                        "A neural parameter/buffer is shared across copies")
            require(not copied.net.training, "Inference network is in training mode")


@contextmanager
def observe_predictions(runtime, streaming, template):
    """Observe one real predict call; wrappers cannot change its return value."""
    original_proba = runtime.CausalBCIDecoder.predict_proba
    original_reset = runtime.CausalBCIDecoder.reset_state
    original_fresh = streaming.fresh_model
    trace = dict(probabilities=[], fresh_copies=0, reset_calls=0,
                 batch1_calls=0, active=None, session_lengths=[])

    def observed_reset(self):
        result = original_reset(self)
        require(self is not template, "The evaluator reset its model template")
        trace["reset_calls"] += 1
        return result

    def observed_fresh(model):
        require(model is template, "Evaluator changed the initial model template")
        clone = original_fresh(model)
        check_copy_isolation(template, clone)
        if trace["active"] is not None:
            trace["session_lengths"].append(int(trace["active"].n_seen))
        trace["active"] = clone
        trace["fresh_copies"] += 1
        return clone

    def observed_proba(self, X):
        require(self is trace["active"] and self is not template, "Unexpected prediction model")
        require(tuple(X.shape) == (1, 41, 480), "Every deployed predict call must receive B=1")
        before = int(self.n_seen)
        require(member_counts(self) == [before] * 3, "Adaptive members disagree before prediction")
        result = original_proba(self, X)
        require(result.shape == (1, 3) and np.isfinite(result).all()
                and np.all(result >= 0.) and np.allclose(result.sum(-1), 1., atol=1e-12),
                "Invalid runtime probability output")
        require(self.n_seen == before + 1 and member_counts(self) == [before + 1] * 3,
                "A window was applied zero or multiple times to adaptive state")
        trace["probabilities"].append(result[0].copy())
        trace["batch1_calls"] += 1
        return result

    runtime.CausalBCIDecoder.predict_proba = observed_proba
    runtime.CausalBCIDecoder.reset_state = observed_reset
    streaming.fresh_model = observed_fresh
    try:
        yield trace
    finally:
        runtime.CausalBCIDecoder.predict_proba = original_proba
        runtime.CausalBCIDecoder.reset_state = original_reset
        streaming.fresh_model = original_fresh


def replay_one(runtime, streaming, model, windows, metadata, rows):
    """Use official predict_streams unchanged, with labels outside model input."""
    before = state_fingerprint(model)

    def loader():
        for row in rows:
            yield (
                torch.from_numpy(np.array(windows[row:row + 1], copy=True)),
                torch.tensor(metadata["y"][row:row + 1]),
                dict(stream_id=metadata["stream_id"][row:row + 1],
                     record_id=metadata["record_id"][row:row + 1],
                     onset=metadata["onset"][row:row + 1],
                     context_id=(metadata["interface"][row:row + 1] == "BrainHero").astype(np.int64)),
            )

    started = time.perf_counter()
    with observe_predictions(runtime, streaming, model) as trace:
        truth, labels, info = streaming.predict_streams(model, loader())
    elapsed = time.perf_counter() - started
    require(np.array_equal(truth, metadata["y"][rows]), "Harness truth order differs")
    require(labels.shape == (len(rows),) and np.issubdtype(labels.dtype, np.integer)
            and np.isin(labels, [0, 1, 2]).all(), "Incorrect deployed predict contract")
    require(np.array_equal(info["stream_id"], metadata["stream_id"][rows])
            and np.array_equal(info["record_id"], metadata["record_id"][rows])
            and np.array_equal(info["onset"], metadata["onset"][rows]), "Harness order differs")
    probabilities = np.stack(trace["probabilities"])
    require(np.array_equal(probabilities.argmax(1), labels),
            "Observed probabilities differ from the actual predict result")
    trace["session_lengths"].append(int(trace["active"].n_seen))
    expected_lengths = [int(np.sum(metadata["stream_id"][rows] == key))
                        for key in dict.fromkeys(metadata["stream_id"][rows])]
    require(trace["fresh_copies"] == trace["reset_calls"] == len(expected_lengths)
            and trace["session_lengths"] == expected_lengths, "Session reset/copy coverage differs")
    require(trace["batch1_calls"] == len(rows), "Not exactly one deployed predict call per window")
    require(model.n_seen == 0 and member_counts(model) == [0, 0, 0]
            and state_fingerprint(model) == before, "Official evaluator modified the model template")
    summary = dict(n_windows=len(rows), n_sessions=len(expected_lengths),
                   batch1_predict_calls=trace["batch1_calls"], deepcopy_calls=trace["fresh_copies"],
                   reset_calls=trace["reset_calls"], session_lengths=expected_lengths,
                   template_unchanged=True, no_shared_session_buffers=True,
                   one_adaptive_update_per_window=True, elapsed_seconds=elapsed)
    return probabilities, labels.astype(np.int64), summary


def smoke_reference(runtime, artifacts, candidate, subject, replicate, packed, rows):
    """Finite smoke only; full verification must use the completed merged OOF."""
    recipe = "baseline" if candidate in {"baseline_v3", "riemannian_tangent"} else candidate
    folder = artifacts.args.neural_root / recipe / ("hold_" + subject)
    path = folder / "predictions.npz"
    digest = artifacts.ledger.add(path)
    audit_path = folder / "isolation_check.json"
    artifacts.ledger.add(audit_path)
    require(json.loads(audit_path.read_text())["prediction_sha256"] == digest,
            "Smoke neural OOF probabilities changed")
    with np.load(path, allow_pickle=False) as archive:
        held = archive["row_indices"]
        require(np.array_equal(held, np.flatnonzero(artifacts.meta["subject"] == subject)),
                "Smoke fold prediction row order differs")
        local = np.searchsorted(held, rows)
        require(np.array_equal(held[local], rows), "Smoke rows are absent from the fold")
        branch = archive["branch_probabilities"][replicate]
        answer = .45 * branch[0, local].astype(np.float64) + .45 * branch[1, local].astype(np.float64)
    raw = np.load(artifacts.args.cache / "temporal_band_covs.npy", mmap_mode="r", allow_pickle=False)
    record = packed["members"][-1]
    original = (runtime.V5TangentMember(record) if candidate == "riemannian_tangent"
                else runtime._LinearMember(record))
    active, current = None, None
    for index, row in enumerate(rows):
        stream = artifacts.meta["stream_id"][row]
        if stream != current:
            active = copy.deepcopy(original)
            active.reset_state()
            current = stream
        answer[index] += .1 * active.predict_one(raw[row], {})
    return answer


def selected_rows(meta, subjects, prefix):
    rows = np.flatnonzero(np.isin(meta["subject"], subjects))
    if prefix:
        rows = np.concatenate([rows[meta["stream_id"][rows] == key][:prefix]
                               for key in dict.fromkeys(meta["stream_id"][rows])])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "v5_work/plan.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--fold-cache", type=Path, default=ROOT / "v5_work/fold_cache")
    parser.add_argument("--neural-root", type=Path, default=ROOT / "v5_work/neural_results")
    parser.add_argument("--classical-root", type=Path, default=ROOT / "v5_work/classical_results")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--kit", type=Path, default=ROOT / "reference_audit/2026-competition")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/cv_streaming_check")
    parser.add_argument("--subjects", nargs="+", choices=PUBLIC_SUBJECTS, default=list(PUBLIC_SUBJECTS))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--replicates", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-windows-per-stream", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for values in (args.subjects, args.candidates, args.replicates):
        require(len(values) == len(set(values)), "Repeated requested subjects/candidates/replicates")
    require(not args.smoke or 1 <= args.max_windows_per_stream <= 32,
            "Smoke replay must use at most 32 leading windows per session")
    require(args.smoke or args.predictions_dir is not None,
            "Full replay requires --predictions-dir with complete merged OOF files")
    require(args.threads >= 1, "Invalid CPU thread count")
    torch.set_num_threads(args.threads)
    threadpool_limits(args.threads)
    started = time.perf_counter()
    ledger = FileLedger()
    ledger.add(args.plan, PLAN_SHA256)
    ledger.add(Path(__file__))
    ledger.add(ROOT / "v5_work/pack_runtime.py")
    ledger.add(ROOT / "v5_work/protocol_v5.py")
    plan = json.loads(args.plan.read_text())
    require(plan["candidates"] == list(CANDIDATES) and plan["ensemble_weights"] == [.45, .45, .1]
            and plan["seed_pairs"] == [list(pair) for pair in SEED_PAIRS]
            and plan["fixed_epochs"] == [75, 20], "Frozen panel definition differs")
    meta, info = load_metadata(args.cache, ledger)
    ledger.add(args.cache / "windows.npy")
    ledger.add(args.cache / "temporal_band_covs.npy")
    windows = np.load(args.cache / "windows.npy", mmap_mode="r", allow_pickle=False)
    require(windows.shape == (len(meta["y"]), 41, 480) and windows.dtype == np.float32,
            "Wrong raw-window cache contract")
    expected, assembly = (load_expected(args.predictions_dir, args.candidates, meta, PLAN_SHA256, ledger)
                          if args.predictions_dir else ({}, None))
    if assembly is not None:
        require(assembly["metadata_sha256"] == ledger.add(args.cache / "metadata.npz")
                and assembly["raw_covariances_sha256"] == ledger.add(args.cache / "temporal_band_covs.npy"),
                "OOF assembly and replay data differ")
    sys.path.insert(0, str(args.kit.resolve()))
    streaming = importlib.import_module("benchmark_utils.streaming")
    require(Path(streaming.__file__).resolve().is_relative_to(args.kit.resolve()),
            "A different benchmark_utils.streaming module was imported")
    ledger.add(Path(streaming.__file__))
    runtime, source, source_hashes = load_runtime_module()
    for path, digest in source_hashes.items():
        ledger.add(ROOT / path, digest)
    artifacts = FoldArtifacts(args, plan, meta, info, ledger)
    rows = selected_rows(meta, args.subjects, args.max_windows_per_stream if args.smoke else 0)
    probabilities = {name: np.full((len(args.replicates), len(rows), 3), np.nan)
                     for name in args.candidates}
    references = {name: np.full_like(probabilities[name], np.nan) for name in args.candidates}
    labels = {name: np.full((len(args.replicates), len(rows)), -1, dtype=np.int64)
              for name in args.candidates}
    jobs = []
    require(not args.out.exists() or not any(args.out.iterdir()),
            "Refusing to overwrite an existing replay audit; choose another --out")
    args.out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        held = rows[meta["subject"][rows] == subject]
        assert_stage_subjects(meta["subject"][held], stage="cv_evaluate", heldout_subject=subject)
        local = np.searchsorted(rows, held)
        for candidate in args.candidates:
            for output_rep, replicate in enumerate(args.replicates):
                packed, all_held, paths = artifacts.pack(candidate, subject, replicate)
                require(np.isin(held, all_held).all(), "Replay includes fitting rows")
                model_meta = dict(device="cpu", n_chans=41, n_times=480, n_classes=3,
                                  sfreq=120., ch_names=info["ch_names"])
                model = runtime.CausalBCIDecoder(packed, model_meta)
                actual, pred, evidence = replay_one(runtime, streaming, model, windows, meta, held)
                reference = (expected[candidate][replicate, held] if expected
                             else smoke_reference(runtime, artifacts, candidate, subject, replicate,
                                                  packed, held))
                delta = np.abs(actual - reference)
                close = np.isclose(actual, reference, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL)
                mismatch = pred != reference.argmax(1)
                evidence.update(candidate=candidate, subject=subject, replicate=replicate,
                                seed_pair=list(SEED_PAIRS[replicate]), member_checkpoints=paths,
                                max_probability_absolute_difference=float(delta.max()),
                                probability_tolerance_passed=bool(close.all()),
                                prediction_mismatches=int(mismatch.sum()),
                                mismatch_row_indices=held[mismatch].tolist())
                jobs.append(evidence)
                probabilities[candidate][output_rep, local] = actual
                references[candidate][output_rep, local] = reference
                labels[candidate][output_rep, local] = pred
                print(json.dumps({key: evidence[key] for key in (
                    "candidate", "subject", "replicate", "n_windows", "prediction_mismatches",
                    "max_probability_absolute_difference", "elapsed_seconds")}), flush=True)
                del model, packed
    require(all(np.isfinite(value).all() for value in probabilities.values()),
            "Some requested replay probabilities are missing")
    files = ledger.verify_unchanged()
    prediction_artifacts = {}
    for candidate in args.candidates:
        path = args.out / (candidate + "_streaming.npz")
        np.savez_compressed(path, probabilities=probabilities[candidate], y_pred=labels[candidate],
                            expected_probabilities=references[candidate],
                            expected_y_pred=references[candidate].argmax(-1),
                            y_true=meta["y"][rows], subject=meta["subject"][rows],
                            session=meta["session"][rows], interface=meta["interface"][rows],
                            stream_id=meta["stream_id"][rows], record_id=meta["record_id"][rows],
                            onset=meta["onset"][rows], row_indices=rows,
                            replicate_indices=np.asarray(args.replicates),
                            seed_pairs=np.asarray(SEED_PAIRS)[args.replicates])
        prediction_artifacts[candidate] = dict(path=str(path), sha256=sha256_file(path))
    checked = all(job["probability_tolerance_passed"] and job["prediction_mismatches"] == 0
                  for job in jobs)
    complete = (not args.smoke and set(args.candidates) == set(CANDIDATES)
                and set(args.subjects) == set(PUBLIC_SUBJECTS) and set(args.replicates) == {0, 1, 2}
                and len(rows) == 2655)
    report = dict(
        schema="track2-v5-official-streaming-replay-v1", checks_passed=checked,
        full_panel_certified=bool(complete and checked), full_coverage=complete,
        smoke_only=bool(args.smoke), plan_sha256=PLAN_SHA256,
        runtime_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_source_files=source_hashes, evaluator="benchmark_utils.streaming.predict_streams",
        probability_observer="read-only wrapper of predict_proba return; deployed predict is unchanged",
        expected_source=("complete hash-bound merged OOF" if expected else
                         "smoke only: cached neural OOF plus saved classical member on cached covariances"),
        probability_atol=PROBABILITY_ATOL, probability_rtol=PROBABILITY_RTOL,
        exact_predicted_labels=all(job["prediction_mismatches"] == 0 for job in jobs),
        streaming_cell_metric_identical_by_exact_labels=bool(checked and expected),
        scores_computed=False, selection_performed=False, training_performed=False,
        public_subjects=list(PUBLIC_SUBJECTS), requested_subjects=args.subjects,
        candidates=args.candidates, replicate_indices=args.replicates,
        unique_windows=len(rows), total_predict_calls=sum(job["batch1_predict_calls"] for job in jobs),
        total_session_deepcopies=sum(job["deepcopy_calls"] for job in jobs),
        total_session_resets=sum(job["reset_calls"] for job in jobs),
        adaptive_state_isolated=True, current_window_only_model_input=True,
        heldout_excluded_from_every_fitting_stage=True, warmup_test_role_retired=True,
        provenance_artifacts=list(artifacts.records.values()),
        files_unchanged=files, prediction_artifacts=prediction_artifacts,
        jobs=jobs, elapsed_seconds=time.perf_counter() - started,
    )
    write_json_exclusive(args.out / "streaming_replay_check.json", report)
    print(json.dumps({key: report[key] for key in (
        "checks_passed", "full_panel_certified", "smoke_only", "total_predict_calls",
        "total_session_deepcopies", "exact_predicted_labels", "elapsed_seconds")}), flush=True)
    if not checked:
        raise RuntimeError("Deployed B1 streaming predictions differ from cached OOF probabilities/labels")


if __name__ == "__main__":
    main()
