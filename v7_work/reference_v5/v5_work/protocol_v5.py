"""Stage-specific subject provenance for the explicitly retired warm-up split.

The v5 development pool is all ten public participants. A CV fit must exclude
its particular held-out participant from every learned quantity, including SSL.
Final and legacy comparison fits require an immutable passing selection record.
This is executable provenance bookkeeping, not an operating-system sandbox.
The v4 guard is deliberately not imported or modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

PUBLIC_SUBJECTS = tuple(f"P{i:02d}" for i in range(1, 11))
ROOT = Path(__file__).resolve().parent.parent
LEGACY_TRAIN_SUBJECTS = ("P01", "P02", "P03", "P06", "P07", "P10")
LEGACY_REPORT_SUBJECTS = ("P04", "P09")
SEED_PAIRS = ((33, 42), (133, 142), (233, 242))
SUBJECT_FIELDS = (
    "fitted_subjects", "pretraining_subjects", "normalization_subjects",
    "reference_subjects", "feature_scale_subjects", "batchnorm_subjects",
    "checkpoint_selection_subjects",
)


class LeakageError(ValueError):
    """A stage boundary, immutable-selection, or provenance check failed."""


def canonical_subject(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip().upper()
    if value.startswith("SUB-"):
        value = value[4:]
    return value


def subject_set(values: Iterable[Any]) -> set[str]:
    if isinstance(values, (str, bytes)):
        values = [values]
    result = {canonical_subject(v) for v in values}
    unknown = result - set(PUBLIC_SUBJECTS)
    if unknown:
        raise LeakageError(f"Unknown/nonpublic participants: {sorted(unknown)}")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _require_passing_selection(selection: Mapping[str, Any]) -> str:
    decision = selection.get("decision", {})
    if decision.get("adopt") is not True:
        raise LeakageError("Null result: no final fit or new submission is authorized")
    name = decision.get("selected_candidate")
    try:
        comparison = selection["candidates"][name]["paired_comparison"]
        gain = float(comparison["mean_paired_gain"])
        se = float(comparison["paired_gain_subject_cluster_se"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LeakageError("Passing selection lacks its measured paired comparison") from exc
    if not (math.isfinite(gain) and math.isfinite(se) and se >= 0 and gain > se
            and comparison.get("passes_adoption_rule") is True):
        raise LeakageError("Selected candidate does not pass mean paired gain > paired SE")
    return str(name)


def _resolve_recorded(recorded: str | Path, anchor: Path) -> Path:
    """Resolve portable evidence, preferring the current extracted workspace."""
    path = Path(recorded)
    candidates = []
    if not path.is_absolute():
        candidates.append(anchor.parent / path)
    for prefix in ("v5_work", "v5_deliverables"):
        if prefix in path.parts:
            candidates.append(ROOT.joinpath(*path.parts[path.parts.index(prefix):]))
    candidates.extend((anchor.parent / path.name, path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(recorded)


def verify_selection_freeze(freeze_path: str | Path) -> dict[str, Any]:
    freeze_path = _resolve_recorded(freeze_path, ROOT / "v5_work/protocol_v5.py")
    frozen = _read(freeze_path)
    if frozen.get("schema") != "track2-v5-passing-selection-v1":
        raise LeakageError("A v5 passing-selection freeze is required")
    selection_path = _resolve_recorded(frozen["selection_path"], freeze_path)
    plan_path = _resolve_recorded(frozen["plan_path"], freeze_path)
    if sha256_file(selection_path) != frozen["selection_sha256"]:
        raise LeakageError("Selection changed after its freeze")
    if sha256_file(plan_path) != frozen["plan_sha256"]:
        raise LeakageError("Candidate plan changed after its freeze")
    selection = _read(selection_path)
    name = _require_passing_selection(selection)
    if name != frozen["selected_candidate"]:
        raise LeakageError("Selected candidate changed")
    return frozen


def assert_stage_subjects(
    values: Iterable[Any], *, stage: str, heldout_subject: str | None = None,
    selection_freeze: str | Path | None = None,
) -> list[str]:
    """Call on actual selected row IDs before every fitting/evaluation stage."""
    actual = subject_set(values)
    heldout = canonical_subject(heldout_subject) if heldout_subject is not None else None
    if stage in {"cv_fit", "cv_evaluate"}:
        if heldout not in PUBLIC_SUBJECTS:
            raise LeakageError("Each CV stage requires one public held-out participant")
        expected = set(PUBLIC_SUBJECTS) - {heldout} if stage == "cv_fit" else {heldout}
    elif stage == "selection":
        if heldout is not None:
            raise LeakageError("Aggregate selection cannot claim one held-out participant")
        expected = set(PUBLIC_SUBJECTS)
    elif stage in {"final_fit", "legacy_fit", "legacy_evaluate"}:
        if heldout is not None or selection_freeze is None:
            raise LeakageError("Post-selection stages require a passing freeze and no CV heldout")
        verify_selection_freeze(selection_freeze)
        expected = set(PUBLIC_SUBJECTS if stage == "final_fit" else
                       LEGACY_TRAIN_SUBJECTS if stage == "legacy_fit" else LEGACY_REPORT_SUBJECTS)
    else:
        raise LeakageError(f"Unknown v5 stage: {stage}")
    if actual != expected:
        raise LeakageError(f"{stage}: expected {sorted(expected)}, got {sorted(actual)}")
    return sorted(actual)


def make_provenance(
    *, stage: str, fitted_subjects: Iterable[Any], heldout_subject: str | None = None,
    pretraining_subjects: Iterable[Any] = (), normalization_subjects: Iterable[Any] | None = None,
    reference_subjects: Iterable[Any] | None = None, feature_scale_subjects: Iterable[Any] | None = None,
    batchnorm_subjects: Iterable[Any] = (), checkpoint_selection_subjects: Iterable[Any] = (),
    seed: int | None = None, epoch: int | None = None, config: Mapping[str, Any] | None = None,
    row_indices: Iterable[int] | None = None, selection_freeze: str | Path | None = None,
) -> dict[str, Any]:
    fitted = assert_stage_subjects(fitted_subjects, stage=stage, heldout_subject=heldout_subject,
                                   selection_freeze=selection_freeze)
    record: dict[str, Any] = {
        "schema": "track2-v5-training-provenance-v1", "stage": stage,
        "heldout_subject": canonical_subject(heldout_subject) if heldout_subject is not None else None,
        "fitted_subjects": fitted, "pretraining_subjects": sorted(subject_set(pretraining_subjects)),
        "normalization_subjects": sorted(subject_set(fitted if normalization_subjects is None else normalization_subjects)),
        "reference_subjects": sorted(subject_set(fitted if reference_subjects is None else reference_subjects)),
        "feature_scale_subjects": sorted(subject_set(fitted if feature_scale_subjects is None else feature_scale_subjects)),
        "batchnorm_subjects": sorted(subject_set(batchnorm_subjects)),
        "checkpoint_selection_subjects": sorted(subject_set(checkpoint_selection_subjects)),
        "seed": None if seed is None else int(seed), "epoch": None if epoch is None else int(epoch),
        "config": dict(config or {}), "config_sha256": json_sha256(dict(config or {})),
        "public_dataset": "nm000290", "public_tag": "1.0.0",
        "warmup_test_role_retired": True,
        "selection_freeze": str(Path(selection_freeze).resolve()) if selection_freeze is not None else None,
    }
    if row_indices is not None:
        rows = [int(v) for v in row_indices]
        record.update(n_fitting_rows=len(rows), row_indices_sha256=json_sha256(rows))
    return validate_provenance(record)


def validate_provenance(record: Mapping[str, Any], *, check_freeze: bool = True) -> dict[str, Any]:
    record = dict(record)
    if record.get("schema") != "track2-v5-training-provenance-v1":
        raise LeakageError("Unknown training provenance schema")
    stage = record["stage"]
    if stage not in {"cv_fit", "final_fit", "legacy_fit"}:
        raise LeakageError("Training provenance must describe a fitting stage")
    heldout = record.get("heldout_subject")
    if stage == "cv_fit" or check_freeze:
        fitted = set(assert_stage_subjects(record["fitted_subjects"], stage=stage,
                                          heldout_subject=heldout,
                                          selection_freeze=record.get("selection_freeze")))
    else:
        fitted = set(PUBLIC_SUBJECTS if stage == "final_fit" else LEGACY_TRAIN_SUBJECTS)
        if subject_set(record["fitted_subjects"]) != fitted:
            raise LeakageError("Post-selection fitting population differs")
    for field in SUBJECT_FIELDS:
        if field not in record:
            raise LeakageError(f"Missing provenance field: {field}")
        actual = subject_set(record[field])
        if actual - fitted:
            raise LeakageError(f"{field}: participant outside this fit: {sorted(actual - fitted)}")
        if heldout is not None and canonical_subject(heldout) in actual:
            raise LeakageError(f"{field}: held-out participant entered fitting")
        record[field] = sorted(actual)
    if record.get("warmup_test_role_retired") is not True:
        raise LeakageError("V5 must explicitly declare retirement of the old test role")
    if (record.get("public_dataset"), record.get("public_tag")) != ("nm000290", "1.0.0"):
        raise LeakageError("Unexpected public-data source")
    if json_sha256(record["config"]) != record["config_sha256"]:
        raise LeakageError("Configuration differs from its hash")
    return record


def register_training_artifact(artifact_path: str | Path, provenance: Mapping[str, Any],
                               manifest_path: str | Path | None = None) -> Path:
    artifact = Path(artifact_path).resolve()
    provenance = validate_provenance(provenance)
    manifest = Path(manifest_path) if manifest_path is not None else artifact.with_suffix(".provenance.json")
    payload = {"schema": "track2-v5-artifact-manifest-v1", "created_at_utc": _now(),
               "artifact_path": str(artifact), "artifact_name": artifact.name,
               "artifact_sha256": sha256_file(artifact), "provenance": provenance}
    write_json_exclusive(manifest, payload)
    return manifest.resolve()


def resolve_manifest_artifact(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path).resolve()
    manifest = _read(manifest_path)
    # The adjacent basename permits archive relocation without editing evidence.
    candidates = [manifest_path.parent / manifest["artifact_name"], Path(manifest["artifact_path"])]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == manifest["artifact_sha256"]:
            return candidate.resolve()
    raise LeakageError(f"Missing or changed artifact bound to {manifest_path}")


def check_training_artifacts(manifests: Iterable[str | Path]) -> dict[str, Any]:
    records = []
    for path in manifests:
        path = Path(path).resolve()
        manifest = _read(path)
        if manifest.get("schema") != "track2-v5-artifact-manifest-v1":
            raise LeakageError("Unexpected artifact manifest schema")
        provenance = validate_provenance(manifest["provenance"])
        artifact = resolve_manifest_artifact(path)
        records.append({"manifest_path": str(path), "manifest_sha256": sha256_file(path),
                        "artifact_path": str(artifact), "artifact_sha256": manifest["artifact_sha256"],
                        **provenance})
    if not records:
        raise LeakageError("No fitting artifacts supplied for verification")
    return {"all_passed": True, "check": "stage_specific_no_heldout_leakage",
            "n_artifacts": len(records), "artifacts": records,
            "warmup_test_role_retired": True}


def freeze_selection(*, selection_path: str | Path, plan_path: str | Path,
                     freeze_path: str | Path) -> dict[str, Any]:
    selection_path, plan_path = Path(selection_path).resolve(), Path(plan_path).resolve()
    selection = _read(selection_path)
    name = _require_passing_selection(selection)
    if selection.get("warmup_test_role_retired") is not True:
        raise LeakageError("Selection must explicitly retire the legacy warm-up test role")
    if selection.get("plan_sha256") != sha256_file(plan_path):
        raise LeakageError("Selection does not match the fixed candidate plan")
    frozen = {"schema": "track2-v5-passing-selection-v1", "frozen_at_utc": _now(),
              "selection_path": str(selection_path), "selection_sha256": sha256_file(selection_path),
              "plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path),
              "selected_candidate": name,
              "warmup_test_role_retired": True}
    write_json_exclusive(freeze_path, frozen)
    return frozen


def claim_legacy_diagnostic_once(*, selection_freeze: str | Path, model_path: str | Path,
                                receipt_path: str | Path) -> dict[str, Any]:
    """Claim the separately fitted legacy-six model's descriptive evaluation."""
    frozen = verify_selection_freeze(selection_freeze)
    receipt = {"schema": "track2-v5-legacy-diagnostic-receipt-v1", "claimed_at_utc": _now(),
               "status": "claimed_before_diagnostic", "selection_freeze_sha256": sha256_file(selection_freeze),
               "selection_sha256": frozen["selection_sha256"], "model_sha256": sha256_file(model_path),
               "interpretation": "post-selection, selection-contaminated legacy-split diagnostic",
               "used_for_selection": False, "warmup_test_role_retired": True}
    write_json_exclusive(receipt_path, receipt)
    return receipt
