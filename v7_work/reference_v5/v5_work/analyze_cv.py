"""Analyze the complete fixed v5 candidate panel using paired subject clusters.

Input: one NPZ per candidate with y_true, subject, session, interface,
row_indices, and probabilities shaped (3, N, 3). All candidate arrays must be
aligned exactly. No historical warm-up scores or predictions are loaded.

The bootstrap preserves all sessions/interfaces of each sampled participant.
It estimates the SE of the paired gain conditional on the fitted CV runs; with
overlapping fitting sets and ten participants it is approximate, not a proof of
statistical significance. Seed spread is reported separately.
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
from v5_work.protocol_v5 import PUBLIC_SUBJECTS, SEED_PAIRS, assert_stage_subjects, sha256_file

DEFAULT_CANDIDATES = (
    "baseline_v3", "augmentation", "adversarial", "masked_ssl", "riemannian_tangent",
)
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261023
FROZEN_PLAN_SHA256 = "ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _strings(array):
    result = np.asarray(array).astype(str)
    require(result.ndim == 1, "Metadata arrays must be one-dimensional")
    return result


def load_predictions(path):
    with np.load(path, allow_pickle=False) as source:
        required = {"y_true", "subject", "session", "interface", "row_indices", "probabilities", "seed_pairs"}
        require(required.issubset(source.files), f"{path}: missing keys {required - set(source.files)}")
        data = {key: np.array(source[key], copy=True) for key in required}
    data["y_true"] = data["y_true"].astype(np.int64)
    for field in ("subject", "session", "interface"):
        data[field] = _strings(data[field])
    require(data["y_true"].ndim == 1, "Labels must be one-dimensional")
    n = len(data["y_true"])
    require(data["probabilities"].shape == (3, n, 3), "Expected three replicate probability arrays")
    require(np.array_equal(data["seed_pairs"], np.array(SEED_PAIRS)), "Prediction replicate seeds differ")
    require(all(len(data[key]) == n for key in ("subject", "session", "interface", "row_indices")),
            "Metadata and prediction lengths differ")
    require(data["row_indices"].ndim == 1 and len(np.unique(data["row_indices"])) == n,
            "Row indices must uniquely identify every evaluated window")
    require(np.isin(data["y_true"], [0, 1, 2]).all(), "Unexpected command label")
    require(np.isfinite(data["probabilities"]).all(), "Nonfinite probabilities")
    require((data["probabilities"] >= -1e-7).all(), "Negative probabilities")
    require(np.allclose(data["probabilities"].sum(axis=2), 1., atol=1e-5), "Probabilities do not sum to one")
    assert_stage_subjects(data["subject"], stage="selection")
    require(set(data["interface"]) == {"Graz", "BrainHero"}, "Expected both interface labels")
    return data


def score_predictions(data):
    keys = sorted(set(zip(data["subject"], data["session"], data["interface"])))
    predictions = data["probabilities"].argmax(axis=2)
    scores = np.empty((3, len(keys)), np.float64)
    cells = []
    for column, (subject, session, interface) in enumerate(keys):
        selected = ((data["subject"] == subject) & (data["session"] == session)
                    & (data["interface"] == interface))
        labels = data["y_true"][selected]
        counts = [int((labels == k).sum()) for k in range(3)]
        require(all(counts), "Every official scoring cell must contain all three commands")
        recalls = []
        for replicate in range(3):
            p = predictions[replicate, selected]
            one = [float((p[labels == k] == k).mean()) for k in range(3)]
            recalls.append(one)
            scores[replicate, column] = np.mean(one)
        cells.append({"subject": str(subject), "session": str(session), "interface": str(interface),
                      "n_windows": int(selected.sum()), "class_counts": counts,
                      "balanced_accuracy_by_replicate": scores[:, column].tolist(),
                      "class_recalls_by_replicate": recalls,
                      "mean_balanced_accuracy": float(scores[:, column].mean())})
    complete_scores = scores.mean(axis=1)
    subjects = []
    for subject in PUBLIC_SUBJECTS:
        columns = [i for i, key in enumerate(keys) if key[0] == subject]
        per_seed = scores[:, columns].mean(axis=1)
        subjects.append({"subject": subject, "n_cells": len(columns),
                         "cell_mean_by_replicate": per_seed.tolist(),
                         "mean_cell_mean": float(per_seed.mean())})
    summary = {"metric": "equal-cell balanced accuracy (subject x session x interface)",
               "n_windows": len(data["y_true"]), "n_cells": len(keys), "n_subjects": len(subjects),
               "n_sessions": len(set(zip(data["subject"], data["session"]))),
               "cv_scores_by_replicate": complete_scores.tolist(),
               "cv_mean": float(complete_scores.mean()),
               "cv_sample_std": float(complete_scores.std(ddof=1)),
               "cells": cells, "subjects": subjects}
    return summary, scores, keys


def paired_subject_cluster(candidate_cells, baseline_cells, keys, *, draws=BOOTSTRAP_DRAWS,
                           seed=BOOTSTRAP_SEED):
    require(candidate_cells.shape == baseline_cells.shape, "Candidate/baseline cells differ")
    difference = candidate_cells - baseline_cells
    # Average matched optimization replicates first; retain participant units.
    mean_cell_difference = difference.mean(axis=0)
    sums, counts, subjects = [], [], []
    for subject in PUBLIC_SUBJECTS:
        columns = np.array([i for i, key in enumerate(keys) if key[0] == subject], np.int64)
        require(len(columns) > 0, "Missing participant in paired comparison")
        sums.append(float(mean_cell_difference[columns].sum()))
        counts.append(len(columns))
        subjects.append({"subject": subject, "n_cells": len(columns),
                         "paired_gain_by_replicate": difference[:, columns].mean(axis=1).tolist(),
                         "mean_paired_gain": float(mean_cell_difference[columns].mean())})
    sums, counts = np.array(sums), np.array(counts)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(PUBLIC_SUBJECTS), size=(int(draws), len(PUBLIC_SUBJECTS)))
    # Unequal session counts remain part of the official equal-cell estimand.
    boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    gain = float(mean_cell_difference.mean())
    se = float(boot.std(ddof=1))
    seed_gains = difference.mean(axis=1)
    require(np.isclose(gain, seed_gains.mean(), atol=1e-14), "Paired aggregation mismatch")
    result = {"mean_paired_gain": gain, "paired_gain_subject_cluster_se": se,
              "paired_gains_by_replicate": seed_gains.tolist(),
              "paired_gain_seed_sample_std": float(seed_gains.std(ddof=1)),
              "bootstrap_draws": int(draws), "bootstrap_seed": int(seed),
              "bootstrap_percentile_interval_95": np.percentile(boot, [2.5, 97.5]).tolist(),
              "gain_minus_se": gain - se, "passes_adoption_rule": bool(gain > se),
              "rule": "mean paired gain > approximate subject-cluster SE of paired gain",
              "uncertainty_note": "Ten participant clusters, not 30 independent fold/seed observations; conditional and approximate because CV fitting sets overlap.",
              "subjects": subjects}
    return result, boot


def _candidate_names(plan):
    listed = plan.get("candidates", list(DEFAULT_CANDIDATES))
    result = [x if isinstance(x, str) else x["name"] for x in listed]
    require(len(set(result)) == len(result), "Duplicate planned candidate")
    require(result == list(DEFAULT_CANDIDATES) or set(result) == set(DEFAULT_CANDIDATES),
            "Candidate panel differs from the five agreed baseline/ablation methods")
    return result


def analyze(*, plan_path, predictions_dir, out):
    plan_path, predictions_dir, out = map(Path, (plan_path, predictions_dir, out))
    plan = json.loads(plan_path.read_text())
    candidates = _candidate_names(plan)
    # Integrity barrier only: the fixed metric, bootstrap and adoption rule below
    # are unchanged. Incomplete or substituted panels cannot reach scoring.
    plan_hash = sha256_file(plan_path)
    require(plan_hash == FROZEN_PLAN_SHA256, "The plan frozen before fitting has changed")
    assembly_path = predictions_dir / "assembly_manifest.json"
    assembly = json.loads(assembly_path.read_text())
    require(assembly.get("schema") == "track2-v5-complete-probability-assembly-v1"
            and assembly.get("complete") is True, "A completed verified assembly is required")
    require(assembly["plan_sha256"] == plan_hash and assembly["candidates"] == candidates,
            "Assembly belongs to another plan/candidate panel")
    require(assembly["n_artifacts"] == 270 and assembly["all_actual_fit_rows_match"] is True,
            "Complete stage-specific fitting provenance is required")
    for name in candidates:
        require(sha256_file(predictions_dir / (name + ".npz")) == assembly["prediction_files"][name]["sha256"],
                "Assembled prediction file changed: " + name)
    audit_path = predictions_dir / "stage_specific_leakage_check.json"
    require(sha256_file(audit_path) == assembly["stage_specific_audit"]["sha256"],
            "Stage-specific isolation audit changed")
    require(plan.get("warmup_test_role_retired") is True, "Plan must explicitly retire the legacy test role")
    require(plan.get("seed_pairs") == [list(x) for x in SEED_PAIRS], "Unexpected optimization replicates")
    baseline_name = plan.get("baseline_candidate", "baseline_v3")
    require(baseline_name == "baseline_v3", "Unexpected baseline")
    loaded = {name: load_predictions(predictions_dir / (name + ".npz")) for name in candidates}
    base = loaded[baseline_name]
    require(len(base["y_true"]) == 2655, "Expected all 2,655 public command windows")
    for name, data in loaded.items():
        for key in ("y_true", "subject", "session", "interface", "row_indices"):
            require(np.array_equal(base[key], data[key]), f"{name}: row/order mismatch in {key}")
    summaries, arrays = {}, {}
    baseline_summary, baseline_cells, keys = score_predictions(base)
    require(baseline_summary["n_cells"] == 28 and baseline_summary["n_sessions"] == 14,
            "Expected 28 public scoring cells and 14 participant-session streams")
    summaries[baseline_name] = baseline_summary
    arrays[baseline_name + "_cell_scores"] = baseline_cells
    for name in candidates:
        if name == baseline_name:
            continue
        summary, cells, other_keys = score_predictions(loaded[name])
        require(other_keys == keys, "Scoring-cell order mismatch")
        comparison, bootstrap = paired_subject_cluster(cells, baseline_cells, keys)
        summaries[name] = {**summary, "paired_comparison": comparison}
        arrays[name + "_cell_scores"] = cells
        arrays[name + "_bootstrap_paired_gains"] = bootstrap
    passing = [name for name in candidates if name != baseline_name
               and summaries[name]["paired_comparison"]["passes_adoption_rule"]]
    ranked = sorted(passing, key=lambda name: (-summaries[name]["paired_comparison"]["gain_minus_se"], name))
    decision = {"adopt": bool(ranked), "selected_candidate": ranked[0] if ranked else None,
                "passing_candidates": ranked,
                "tie_break": "maximum mean paired gain minus subject-cluster SE, then lexicographic candidate name",
                "reason": "At least one fixed candidate exceeds its paired SE" if ranked else
                          "No fixed candidate exceeds the paired-gain SE; no new submission or all-subject final fit"}
    result = {"schema": "track2-v5-cv-selection-v1", "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan_path": str(plan_path.resolve()), "plan_sha256": plan_hash,
              "assembly_manifest_sha256": sha256_file(assembly_path),
              "baseline_candidate": baseline_name, "seed_pairs": [list(x) for x in SEED_PAIRS],
              "warmup_test_role_retired": True,
              "selection_subjects": list(PUBLIC_SUBJECTS),
              "legacy_diagnostic_used_for_selection": False,
              "scores_are_development_estimates": True,
              "protocol": "Fixed panel; ten-subject LOSO with three paired optimization replicates; official cell-mean scoring",
              "uncertainty_note": "Mean +/- sample SD across three complete CV scores measures optimization variability; the paired participant-cluster bootstrap SE is approximate and does not remove selection bias.",
              "prediction_files": {name: {"path": str((predictions_dir / (name + '.npz')).resolve()),
                                            "sha256": sha256_file(predictions_dir / (name + '.npz'))}
                                   for name in candidates},
              "candidates": summaries, "decision": decision}
    out.mkdir(parents=True, exist_ok=True)
    target = out / "selection.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    np.savez_compressed(out / "paired_statistics.npz", **arrays,
                        cell_subject=np.array([k[0] for k in keys]),
                        cell_session=np.array([k[1] for k in keys]),
                        cell_interface=np.array([k[2] for k in keys]))
    print(json.dumps({"decision": decision, "scores": {
        name: {"mean": score["cv_mean"], "std": score["cv_sample_std"],
               **({k: score["paired_comparison"][k] for k in
                   ("mean_paired_gain", "paired_gain_subject_cluster_se", "passes_adoption_rule")}
                  if "paired_comparison" in score else {})}
        for name, score in summaries.items()}}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "v5_work/plan.json")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "v5_work/cv_predictions")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/cv_analysis")
    args = parser.parse_args()
    analyze(plan_path=args.plan, predictions_dir=args.predictions_dir, out=args.out)
