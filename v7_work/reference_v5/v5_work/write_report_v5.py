"""Write the v5 report only from complete, internally consistent evidence.

No fitting, inference, prediction scoring, hyperparameter changes or new
selection are performed. Missing evidence aborts instead of producing a
placeholder report. A null result forbids a new fitted model/submission; a
passing result requires the final and legacy packages, their 17-check reports,
and the single explicitly contaminated legacy diagnostic.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, LEGACY_TRAIN_SUBJECTS, SEED_PAIRS, canonical_subject,
    check_training_artifacts, sha256_file, verify_selection_freeze,
    write_json_exclusive,
)
from v5_work.assemble_cv_predictions import CANDIDATES, FROZEN_PLAN_SHA256, audit_signature
from v5_work.contract_check_v5 import resolve_recorded_path

SHORT = dict(baseline_v3="V3", augmentation="Aug", adversarial="Adv",
             masked_ssl="SSL", riemannian_tangent="Tan")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(headers, rows):
    rows = [[str(item) for item in row] for row in rows]
    widths = [max(len(str(header)), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    render = lambda row: " | ".join(str(value).ljust(width) for value, width in zip(row, widths))
    return "\n".join([render(headers), "-+-".join("-" * width for width in widths), *map(render, rows)])


def percent(value):
    require(math.isfinite(value), "Nonfinite measured score")
    return f"{100 * value:.2f}"


def signed_points(value):
    require(math.isfinite(value), "Nonfinite measured paired gain")
    return f"{100 * value:+.2f}"


def function_ast(path, name):
    tree = ast.parse(Path(path).read_text())
    matching = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matching) == 1, "Cannot identify preprocessing function")
    return ast.dump(matching[0], include_attributes=False)


class Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        require(expected is None or digest == expected, "Evidence hash differs: " + str(path))
        self.files[str(path)] = digest
        return digest

    def read(self, path, expected=None):
        self.bind(path, expected)
        return json.loads(Path(path).read_text())


def validate_package(directory, stage, candidate, freeze, evidence):
    directory = Path(directory)
    archive = directory / ("submission.zip" if stage == "final_fit" else "legacy_submission.zip")
    build = evidence.read(directory / "build_info.json")
    contract = evidence.read(directory / "contract_check.json")
    fitting = list(PUBLIC_SUBJECTS if stage == "final_fit" else LEGACY_TRAIN_SUBJECTS)
    require(build["stage"] == stage and build["candidate"] == candidate
            and build["training_subjects"] == fitting, "Packaged stage/candidate/population differs")
    archive_hash = evidence.bind(archive, build["sha256"])
    require(contract["all_passed"] is True and contract["n_checks"] == 17
            and len(contract["checks"]) == 17 and all(check["passed"] for check in contract["checks"])
            and contract["stage"] == stage and contract["fitting_subjects"] == fitting,
            "All 17 stage-specific package checks must pass")
    require(contract["checks"][-1]["selection_freeze_sha256"] == freeze
            and build["selection_freeze_sha256"] == freeze, "Package/contract selection freeze differs")
    require(contract["checks"][-1]["n_cv_artifacts"] == 270
            and contract["checks"][-1]["all_seven_fields_validated"] is True
            and contract["checks"][-1]["packaged_tensors_equal_registered_sources"] is True,
            "Incomplete stage-specific contract provenance")
    with zipfile.ZipFile(archive) as handle:
        require(handle.namelist() == ["submission.py", "weights.pt"] and handle.testzip() is None,
                "Invalid submission ZIP root/CRC")
        for name, key in (("submission.py", "submission_sha256"), ("weights.pt", "weights_sha256")):
            require(hashlib.sha256(handle.read(name)).hexdigest() == contract[key], "ZIP no longer matches checked files")
    listing = evidence.read(directory / "source_manifests.json")
    manifests = [resolve_recorded_path(path, directory / "source_manifests.json") for path in listing["manifests"]]
    require(len(manifests) == 4, "Expected one reference, two neural branches and one classical branch")
    current = check_training_artifacts(manifests)
    require(audit_signature(current) == audit_signature(build["source_artifacts"]), "Post-selection sources changed")
    for record in current["artifacts"]:
        require(record["stage"] == stage and record["fitted_subjects"] == fitting, "Post-selection source stage differs")
        evidence.bind(record["manifest_path"], record["manifest_sha256"])
        evidence.bind(record["artifact_path"], record["artifact_sha256"])
    return dict(archive=archive, sha256=archive_hash, build=build, contract=contract)


def write_report(args):
    evidence = Evidence()
    plan = evidence.read(args.plan, FROZEN_PLAN_SHA256)
    selection = evidence.read(args.selection)
    assembly = evidence.read(args.assembly, selection["assembly_manifest_sha256"])
    require(selection["plan_sha256"] == FROZEN_PLAN_SHA256 == assembly["plan_sha256"], "Selection/assembly plan differs")
    require(assembly["complete"] is True and assembly["all_actual_fit_rows_match"] is True
            and assembly["n_artifacts"] == 270 and assembly["n_rows"] == 2655
            and assembly["n_cells"] == 28, "Complete CV assembly is required")
    require(plan["candidates"] == list(CANDIDATES) and set(selection["candidates"]) == set(CANDIDATES),
            "Incomplete or changed candidate panel")
    require(selection["warmup_test_role_retired"] is True
            and selection["scores_are_development_estimates"] is True
            and selection["legacy_diagnostic_used_for_selection"] is False, "Contamination declaration differs")
    require(plan["seed_pairs"] == selection["seed_pairs"] == [list(pair) for pair in SEED_PAIRS], "Seed plan differs")
    audit = evidence.read(args.cv_audit, assembly["stage_specific_audit"]["sha256"])
    require(audit["all_passed"] is True and audit["n_artifacts"] == 270
            and audit["n_reference_artifacts"] == 10 and audit["n_neural_artifacts"] == 240
            and audit["n_classical_artifacts"] == 20 and audit["all_actual_fit_rows_match"] is True,
            "Complete stage-specific CV isolation evidence is required")
    manifests = [resolve_recorded_path(record["manifest_path"], args.cv_audit) for record in audit["artifacts"]]
    current_cv = check_training_artifacts(manifests)
    require(audit_signature(current_cv) == audit_signature(audit), "A CV fitting artifact changed")
    for record in current_cv["artifacts"]:
        evidence.files[record["manifest_path"]] = record["manifest_sha256"]
        evidence.files[record["artifact_path"]] = record["artifact_sha256"]
    for name in CANDIDATES:
        path = args.assembly.parent / (name + ".npz")
        evidence.bind(path, assembly["prediction_files"][name]["sha256"])
        require(assembly["prediction_files"][name]["sha256"] == selection["prediction_files"][name]["sha256"],
                "Analysis and assembled predictions differ")

    replay = evidence.read(args.replay)
    require(replay["full_panel_certified"] is True and replay["full_coverage"] is True
            and replay["checks_passed"] is True and replay["exact_predicted_labels"] is True,
            "Full-panel official streaming replay must pass before reporting")
    require(replay["plan_sha256"] == FROZEN_PLAN_SHA256 and replay["total_predict_calls"] == 39825
            and replay["total_session_deepcopies"] == replay["total_session_resets"] == 210,
            "Streaming coverage differs")
    require(replay["scores_computed"] is False and replay["selection_performed"] is False
            and replay["streaming_cell_metric_identical_by_exact_labels"] is True,
            "Streaming replay must be an independent prediction-equivalence check")
    assembly_hash = sha256_file(args.assembly)
    require(any(Path(path).name == "assembly_manifest.json" and digest == assembly_hash
                for path, digest in replay["files_unchanged"].items()), "Replay was bound to another OOF assembly")
    for name, record in replay["prediction_artifacts"].items():
        evidence.bind(resolve_recorded_path(record["path"], args.replay), record["sha256"])
    require(set(replay["prediction_artifacts"]) == set(CANDIDATES), "Replay predictions omit a candidate")
    references = evidence.read(args.reference_sources)
    metric_check = evidence.read(args.official_metric_check)
    require(references["competition_commit"] == plan["competition_commit"]
            and metric_check["metric_source_sha256"] == references["files"]["benchmark_utils/metrics.py"],
            "Official metric source differs from the pinned competition reference")
    require(metric_check["all_passed"] is True and metric_check["n_checks"] == len(metric_check["checks"]) == 15
            and metric_check["batch_size"] == 1 and metric_check["n_windows_per_replicate"] == 2655
            and metric_check["analysis_sha256"] == sha256_file(args.selection)
            and metric_check["replay_sha256"] == sha256_file(args.replay)
            and metric_check["selection_rule_modified"] is False, "Official metric equality check is incomplete")
    seen = set()
    for check in metric_check["checks"]:
        name, replicate = check["candidate"], check["replicate"]
        require(name in CANDIDATES and replicate in range(3) and (name, replicate) not in seen,
                "Unexpected or duplicate official metric check")
        seen.add((name, replicate))
        require(check["passed"] is True and check["n_cells"] == 28
                and check["seed_pair"] == list(SEED_PAIRS[replicate])
                and check["prediction_sha256"] == replay["prediction_artifacts"][name]["sha256"],
                "Official metric was calculated from different streamed predictions")
        require(abs(check["official_streaming_cell_mean"] - selection["candidates"][name]["cv_scores_by_replicate"][replicate]) < 1e-12,
                "Official streamed metric differs from the report's complete CV score")
    require(len(seen) == 15, "Official metric check omits a candidate/replicate")

    dataset = evidence.read(args.cache / "dataset_info.json")
    feature = evidence.read(args.cache / "feature_config_reproducible.json")
    implementation = evidence.read(args.environment_record)
    evidence.bind(args.requirements)
    description = evidence.read(args.dataset_description)
    require(dataset["dataset"] == "NEMAR nm000290" and dataset["tag"] == "1.0.0"
            and dataset["n_windows"] == 2655 and dataset["n_source_runs"] == 112
            and dataset["n_cued_runs"] == 84 and dataset["n_streams"] == 14,
            "Unexpected public data release/counts")
    require(dataset["input_shape"] == [41, 480] and dataset["sfreq"] == 120.
            and dataset["source_sfreq"] == 500. and dataset["scale_factor"] == 1e6
            and dataset["bandpass_hz"] == [.1, 75] and dataset["window_duration_seconds"] == 4.
            and dataset["window_start_seconds_after_cue"] == 1., "Preprocessing contract differs")
    require(feature["labels_used"] is False and feature["cross_window_statistics"] is False
            and feature["crops"] == [[0, 240], [240, 480]] and len(feature["bands"]) == 7,
            "Window-local feature extraction differs")
    require(description["License"] == "CC-BY-4.0" and description["Version"] == "1.0.0"
            and description["DatasetDOI"] == "10.82901/nemar.nm000290" and description["Authors"],
            "Public data credit/version is missing")
    require(implementation["preprocessing_prepare_run_AST_identical_to_v3"] is True, "Recorded preprocessing identity did not pass")
    for path in (ROOT / "v5_work/prepare_public.py", ROOT / "bci_work/prepare_data.py"):
        evidence.bind(path)
    require(function_ast(ROOT / "v5_work/prepare_public.py", "prepare_run") ==
            function_ast(ROOT / "bci_work/prepare_data.py", "prepare_run"), "Preprocessing function changed from retained v3")
    environment = implementation["environment"]
    require(environment["cuda_available"] is False, "Report's CPU execution description differs")

    evidence.bind(args.cache / "metadata.npz", assembly["metadata_sha256"])
    with np.load(args.cache / "metadata.npz", allow_pickle=False) as archive:
        subject = np.array(archive["subject"], copy=True)
    splits = evidence.read(args.splits)
    require(splits["warmup_test_role_retired"] is True and len(splits["folds"]) == 10
            and splits["cv_seed_pairs"] == plan["seed_pairs"], "Explicit fold/seed record differs")
    fold_rows = []
    for fold in sorted(splits["folds"], key=lambda value: value["heldout_subject"]):
        heldout = canonical_subject(fold["heldout_subject"])
        fitting = sorted(canonical_subject(value) for value in fold["training_subjects"])
        require(fitting == sorted(set(PUBLIC_SUBJECTS) - {heldout}), "Explicit LOSO fitting IDs differ")
        require(np.array_equal(fold["training_row_indices"], np.flatnonzero(subject != heldout))
                and np.array_equal(fold["heldout_row_indices"], np.flatnonzero(subject == heldout)), "Explicit LOSO rows differ")
        fold_rows.append([heldout, ",".join(fitting), len(fold["heldout_row_indices"])])
    require([row[0] for row in fold_rows] == list(PUBLIC_SUBJECTS), "Duplicate or missing explicit fold")

    summaries = selection["candidates"]
    base = summaries["baseline_v3"]
    cell_keys = [(item["subject"], item["session"], item["interface"]) for item in base["cells"]]
    require(len(cell_keys) == len(set(cell_keys)) == 28, "Expected 28 unique scoring cells")
    require(sum(item["n_windows"] for item in base["cells"]) == 2655, "Scoring-cell windows do not cover the dataset")
    cv_rows, paired_rows = [], []
    passing = []
    for name in CANDIDATES:
        summary = summaries[name]
        scores = summary["cv_scores_by_replicate"]
        require(len(scores) == 3 and all(math.isfinite(value) for value in scores), "Missing complete CV replicate")
        require(math.isclose(statistics.mean(scores), summary["cv_mean"], abs_tol=1e-12)
                and math.isclose(statistics.stdev(scores), summary["cv_sample_std"], abs_tol=1e-12), "CV mean/sample SD differs")
        require([(item["subject"], item["session"], item["interface"]) for item in summary["cells"]] == cell_keys,
                "Candidate scoring-cell order differs")
        require(math.isclose(statistics.mean(item["mean_balanced_accuracy"] for item in summary["cells"]),
                             summary["cv_mean"], abs_tol=1e-12), "Official cell weighting differs")
        cv_rows.append([SHORT[name], *map(percent, scores),
                        percent(summary["cv_mean"]) + " +/- " + percent(summary["cv_sample_std"])])
        if name != "baseline_v3":
            comparison = summary["paired_comparison"]
            gain, se = comparison["mean_paired_gain"], comparison["paired_gain_subject_cluster_se"]
            require(math.isfinite(gain) and math.isfinite(se) and se >= 0,
                    "Nonfinite or invalid paired comparison")
            require(comparison["passes_adoption_rule"] == (gain > se)
                    and comparison["bootstrap_draws"] == 10000 and comparison["bootstrap_seed"] == 20261023,
                    "Paired decision differs from the frozen rule")
            require(math.isclose(comparison["gain_minus_se"], gain - se, rel_tol=0., abs_tol=1e-15),
                    "Stored winner-ranking margin differs from paired gain minus SE")
            require(math.isclose(gain, summary["cv_mean"] - base["cv_mean"], abs_tol=1e-12), "Paired/unpaired aggregation mismatch")
            interval = comparison["bootstrap_percentile_interval_95"]
            paired_rows.append([SHORT[name], signed_points(gain), percent(se),
                                "[" + signed_points(interval[0]) + ", " + signed_points(interval[1]) + "]",
                                "PASS" if comparison["passes_adoption_rule"] else "NO"])
            if comparison["passes_adoption_rule"]:
                passing.append(name)
    passing.sort(key=lambda name: (-summaries[name]["paired_comparison"]["gain_minus_se"], name))
    decision = selection["decision"]
    require(decision["adopt"] == bool(passing) and decision["passing_candidates"] == passing
            and decision["selected_candidate"] == (passing[0] if passing else None), "Stored decision differs from predeclared rule")
    adopted = decision["adopt"]
    selected = decision["selected_candidate"]

    subject_rows = []
    for row in base["subjects"]:
        sid = row["subject"]
        gains = []
        for name in CANDIDATES[1:]:
            items = [item for item in summaries[name]["paired_comparison"]["subjects"] if item["subject"] == sid]
            require(len(items) == 1 and items[0]["n_cells"] == row["n_cells"], "Paired subject table differs")
            gains.append(signed_points(items[0]["mean_paired_gain"]))
        subject_rows.append([sid, row["n_cells"], percent(row["mean_cell_mean"]), *gains])
    require([row[0] for row in subject_rows] == list(PUBLIC_SUBJECTS), "Subject diagnostic table is incomplete")
    cell_rows = []
    for index, cell in enumerate(base["cells"]):
        values = []
        for name in CANDIDATES:
            other = summaries[name]["cells"][index]
            require(other["n_windows"] == cell["n_windows"] and other["class_counts"] == cell["class_counts"], "Cell populations differ")
            values.append(percent(other["mean_balanced_accuracy"]))
        cell_rows.append([cell["subject"], cell["session"], cell["interface"], cell["n_windows"],
                          "/".join(map(str, cell["class_counts"])), *values])

    classical = evidence.read(args.classical_summary)
    require(classical["complete"] is True and len(classical["folds"]) == 10
            and classical["plan_sha256"] == FROZEN_PLAN_SHA256, "Classical solver evidence is incomplete")
    converged_folds = sum(bool(fold["tangent_solver_all_bands_converged"]) for fold in classical["folds"])
    maximum_residual = max(float(fold["maximum_tangent_solver_residual"]) for fold in classical["folds"])
    require(math.isfinite(maximum_residual), "Nonfinite tangent solver residual")

    conditional = []
    if adopted:
        frozen = verify_selection_freeze(args.selection_freeze)
        freeze_hash = evidence.bind(args.selection_freeze)
        require(frozen["selection_sha256"] == sha256_file(args.selection)
                and frozen["selected_candidate"] == selected, "Post-selection freeze differs from reported decision")
        final = validate_package(args.final_dir, "final_fit", selected, freeze_hash, evidence)
        legacy = validate_package(args.legacy_dir, "legacy_fit", selected, freeze_hash, evidence)
        for stage, expected in (("final", list(PUBLIC_SUBJECTS)), ("legacy", list(LEGACY_TRAIN_SUBJECTS))):
            fit = evidence.read(ROOT / "v5_work/postselection" / stage / "fit_summary.json")
            require(fit["stage"] == stage + "_fit" and fit["candidate"] == selected
                    and fit["fitting_subjects"] == expected and fit["seed_pair"] == [33, 42]
                    and fit["fixed_epochs"] == [75, 20] and fit["evaluated_windows"] == 0,
                    "Final/legacy neural fitting record differs")
        legacy_result = evidence.read(args.legacy_result)
        receipt = evidence.read(args.legacy_result.parent / "once_receipt.json")
        frozen_packages = evidence.read(args.legacy_result.parent / "frozen_packages.json")
        require(legacy_result["used_for_selection"] is False and legacy_result["passes"] == 1
                and legacy_result["score_of_all_ten_final_model"] is False
                and legacy_result["independent_test_estimate"] is False
                and legacy_result["official_metric_exact_match"] is True
                and legacy_result["selected_candidate"] == selected, "Legacy diagnostic interpretation differs")
        require(legacy_result["final_submission_sha256"] == frozen_packages["final_submission_sha256"] == final["sha256"]
                and legacy_result["legacy_submission_sha256"] == frozen_packages["legacy_submission_sha256"] == legacy["sha256"],
                "A package changed after the diagnostic freeze")
        require(receipt["model_sha256"] == legacy["sha256"]
                and receipt["selection_sha256"] == frozen["selection_sha256"]
                and frozen_packages["diagnostic_passes_authorized"] == 1, "Single diagnostic receipt differs")
        metric = legacy_result["metric"]
        require(metric["n_windows"] == 755 and metric["n_cells"] == len(metric["cells"]) == 8
                and metric["n_sessions"] == 4, "Legacy diagnostic coverage differs")
        evidence.bind(args.legacy_result.parent / "predictions.npz", legacy_result["prediction_sha256"])
        conditional = [
            "7. CONDITIONAL FINAL MODEL AND LEGACY DIAGNOSTIC",
            f"The selected {selected} recipe was frozen and refitted on P01-P10, all 2,655 public windows/all 14 sessions. Deployment uses the predeclared seeds (33,42), not the best CV seed or an untested three-replicate average. The final ZIP and the separate original-six diagnostic ZIP each passed all 17 checks. Packaged learned tensors match their registered fitting sources.",
            "Training on all ten is legitimate preparation for later public-participant sessions plus ten additional unseen participants in the sealed phase [1]. It expands public training coverage; it does not establish a sealed-phase performance gain. No score on these already-fitted public subjects is presented as held-out performance of the final model.",
            "A separate model was fitted on P01,P02,P03,P06,P07,P10, with the same frozen recipe and deployment seeds. After both ZIPs were frozen and checked, one batch-one diagnostic evaluated P04/P09: "
            + percent(metric["balanced_accuracy_cell_mean"]) + "% cell-mean balanced accuracy, 755 windows, four sessions, eight cells. This is selection-contaminated because P04/P09 participated in v5 development CV. It is not an independent test estimate and not the score of the all-ten final model; it could not change the frozen selection or package.",
            "Legacy diagnostic per-cell balanced accuracy (%), from that single pass:\n" + table(
                ["Subject", "Session", "Interface", "N", "BA"],
                [[cell["subject"], cell["session"], cell["interface"], cell["n_windows"], percent(cell["balanced_accuracy"])]
                 for cell in metric["cells"]]),
            "Final submission SHA-256: " + final["sha256"],
        ]
    else:
        require(not (args.final_dir / "submission.zip").exists()
                and not (args.final_dir / "submission/weights.pt").exists()
                and not (args.legacy_dir / "legacy_submission.zip").exists()
                and not args.legacy_result.exists() and not args.selection_freeze.exists(),
                "Null result must not have a passing freeze, new submission or legacy diagnostic")
        for directory in (ROOT / "v5_work/postselection", ROOT / "v5_work/classical_results/full_train",
                          ROOT / "v5_work/classical_results/legacy_six"):
            require(not any(path.suffix in {".pt", ".npz"} for path in directory.rglob("*") if path.is_file()),
                    "Null result must not have a new post-selection fitted model")
        conditional = [
            "7. NULL RESULT AND SUBMISSION STATUS",
            "None of the four fixed ablations exceeded its paired-gain SE. V3 remains the retained submission; no v5 submission.zip was produced. No all-ten final model, original-six diagnostic model, passing-selection freeze, or additional historical P04/P09 diagnostic was fitted/evaluated.",
            "The 17 submission contract checks are NOT APPLICABLE: no new submission exists to check. This is distinct from the completed 270-artifact CV isolation audit and full official-harness replay, which passed. The result supports stopping this bounded search; it does not prove that v3 is a global ceiling or that every untried method would fail.",
        ]

    candidate_outcome = (f"PASS — {selected} passed the predeclared rule and was selected by maximum paired gain minus paired SE."
                         if adopted else "NULL — no candidate passed the predeclared paired-gain adoption rule.")
    baseline_text = percent(base["cv_mean"]) + "% +/- " + percent(base["cv_sample_std"]) + " percentage points"
    authors = "; ".join(description["Authors"])
    seconds = sum(float(job["elapsed_seconds"]) for job in replay["jobs"])
    metric_tables = [
        "CV results (%); SD is sample SD over three complete LOSO runs, not an uncertainty interval:",
        table(["Method", "Seeds 33/42", "133/142", "233/242", "Mean +/- SD"], cv_rows),
        "Paired change versus the refitted V3 baseline (percentage points). The 95% percentile interval is an approximate descriptive bootstrap interval; the adoption rule uses one paired SE, not this interval:",
        table(["Method", "Mean gain", "Paired SE", "Approx. 95% interval", "Adoptable"], paired_rows),
    ]
    text = "\n\n".join([
        "NEURAL INTERFACES 2026 TRACK 2 — V5 OPTION A REPORT",
        candidate_outcome + "\nRefitted V3 baseline: " + baseline_text + " across the three complete CV replicates. All values below are local development results, not a Codabench or sealed-phase score.",
        "1. SELECTION PROTOCOL AND CONTAMINATION",
        "All public participants P01,P02,P03,P04,P05,P06,P07,P08,P09,P10 are development data. The old P04/P09 test role was explicitly retired; P05/P08 also cease to be a standalone selection set. Earlier v3/v4 scores are historical context only, not inputs to this decision. No new validation-only or historical-test-only score is used to choose a candidate. Fold isolation does not erase the earlier selection history, so these estimates are not a newly pristine test.",
        "Each of ten LOSO folds refits V3 and all ablations on exactly the other nine participants. Seeds are paired as (33,42), (133,142), (233,242); branch epochs are fixed at 75/20 with a 120-epoch cosine schedule. No checkpoint picking or early stopping uses the held-out fold. The deterministic classical branch is shared across the three optimization replicates. Every fitted reference, SSL example/scale, classifier scale and BatchNorm statistic excludes the current held-out participant.",
        "Exact folds (IDs have the BIDS prefix sub-; full row memberships are in v5_work/splits.json):\n" + table(
            ["Held out", "Fitting participants", "Held-out windows"], fold_rows),
        "The frozen rule averages balanced accuracy equally over 28 subject x session x interface cells. For each ablation, matched candidate-minus-V3 cell differences are averaged over the three replicates, then ten participant clusters are resampled 10,000 times (seed 20261023), retaining all of each participant's two or six cells. The bootstrap ratio preserves the official cell weighting. Adopt only if mean paired gain exceeds its paired-gain SE; among passes, maximize gain minus SE, then break exact ties lexicographically. Decisions use unrounded values. This one-SE development rule is not a statistical-significance claim: ten participants, overlapping CV fitting sets, and selection among four ablations leave substantial uncertainty and selection optimism.",
        "2. ARCHITECTURE AND FIXED ABLATIONS",
        "V3 is a 45% + 45% softmax mixture of two learned spatial log-variance networks plus a 10% Euclidean-aligned log-covariance ridge model. Seven spectral banks in each two-second half produce 14 covariance banks; each neural branch learns 16 spatial filters per bank, includes channel log powers, and uses training-only BatchNorm, dropout 0.5 and a linear three-class head. Ridge alpha=10,000 and temperature=0.125 remain fixed. Session covariance adaptation is unchanged across candidates: neural prior count 16, classical prior count 8, EMA floor 0.02, shrinkage 0.05 and update clipping 10. No TTA, interface-dependent weight search, architecture sweep or foundation model is claimed.",
        "Aug: 10% channel masking, 10% paired non-broadband bank masking, and cross-subject mixup (alpha 0.2, probability 0.5), using fitting subjects only. Adv: a training-only 64-unit subject classifier with gradient reversal strength 0.1. SSL: 30 epochs of fitting-only masked log-power reconstruction and fitting-only target standardization, retaining only spatial filters and restarting BatchNorm/classifier before supervised training. Tan: replace only the 10% classical branch with a tangent-space ridge model around a fitting-only affine-invariant covariance mean; keep regularization, weighting and adaptation fixed. Auxiliary SSL/domain heads are discarded.",
        f"The tangent mean used its predeclared maximum 12 updates. All band means met the solver tolerance in {converged_folds}/10 folds; the largest recorded final residual was {maximum_residual:.6g}. Finite-iteration results are retained as measured, without post-score solver tuning. Full recipes and numerical settings are in plan.json.",
        "3. COMPLETE CV AND PAIRED ABLATIONS",
        *metric_tables,
        "4. ALL 28 SCORING CELLS",
        "Values are mean balanced accuracy (%) over the three paired ensemble replicates. N is windows; class counts are mi/sub/word. V3=baseline_v3, Aug=augmentation, Adv=adversarial, SSL=masked_ssl, Tan=riemannian_tangent. Every cell receives equal weight in the headline metric.",
        table(["Subject", "Session", "Interface", "N", "Class counts", "V3", "Aug", "Adv", "SSL", "Tan"], cell_rows),
        "5. PARTICIPANT DIAGNOSTICS",
        "V3 is the within-participant cell mean (%); all four changes are paired percentage points. This table is diagnostic: averaging its ten rows equally would not reproduce the 28-cell ranking metric because P04/P10 each contribute six cells.",
        table(["Subject", "Cells", "V3", "Gain Aug", "Gain Adv", "Gain SSL", "Gain Tan"], subject_rows),
        "6. DATA, CAUSAL REPLAY AND FITTING AUDIT",
        "Public nm000290 v1.0.0 contributes 112 source runs, 84 cued runs, 14 sessions and 2,655 windows. Preprocessing matches the retained v3 prepare_run function at the AST level: 50/60-Hz notch bases and their sub-Nyquist harmonics, 0.1-75-Hz bandpass at 500 Hz, resampling to 120 Hz, float32 conversion and scaling to microvolts by 1e6. Four-second windows begin one second after the cue, yielding 41 x 480. Labels are mi=0, sub=1, word=2. There is no additional offline scaling; model-internal covariance transformations are reproduced during inference. Metadata records the exact channel order and pinned preprocessing versions.",
        "The actual official benchmark_utils.streaming.predict_streams harness replayed all five candidates, all three replicates and every window: 39,825 batch-one predict calls with 210 session deepcopies and 210 resets. Every predicted label exactly matched the assembled OOF label; probabilities matched within atol=2e-6/rtol=1e-5. The official group_scores(balanced_accuracy) calculation on these actual streamed labels independently matched all 15 reported CV scores within 1e-12. A read-only observer captured probabilities from the same predict call, so adaptation was not applied twice. Each session starts from its fitted prior; only its current EEG window reaches the model. No labels, future-window statistics or another session's state enter prediction.",
        f"The recorded sum of measured inference time across replay jobs was {seconds:.2f} seconds ({1000 * seconds / 39825:.3f} ms/window); this sum is not parallel wall-clock time. All 270 learned CV artifacts—240 neural checkpoints, 20 classical checkpoints and ten references—passed the stage-specific audit of actual fitting rows, seven fitting-provenance fields, hashes and embedded metadata. P04/P09 are excluded in their own held-out folds and legitimately appear among fitting subjects in other folds. These are executable evidence checks and source review, not an operating-system sandbox or formal proof.",
        *conditional,
        "8. REPRODUCTION, LIMITS AND CREDIT",
        f"Recorded environment: Python {environment['python']}, CPU PyTorch {environment['torch']}, NumPy {environment['numpy']}, SciPy {environment['scipy']}, scikit-learn {environment['scikit_learn']}, MNE {environment['mne']}; no GPU. The fixed plan SHA-256 is {FROZEN_PLAN_SHA256}. reproduce.zip retains code, exact subject/row splits, seed pairs, all CV checkpoints, per-artifact provenance, OOF/replay predictions, logs and analysis. Omitted raw EEG and large covariance caches are regenerated by README_reproduce.txt. Floating-point/library differences may change a fresh fit; the archived checkpoints preserve the measured run.",
        "The bounded search tests four specific changes against the same refitted backbone. Neither a pass nor a null result establishes unseen-subject or sealed-phase performance. No further public-score tuning is justified by this report. Deployment, when present, uses the same replay-verified runtime source and the explicitly authorized all-public fitting stage.",
        f"Data credit: {description['Name']}; authors {authors}. NEMAR nm000290 v1.0.0, DOI https://doi.org/{description['DatasetDOI']}, CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). The dataset_description.json is retained. Derived features and models are experimental outputs from this versioned public release.",
        "PRIMARY REFERENCES AND LOCAL EVIDENCE\n"
        + "[1] Official Track 2 FAQ, including sealed-session/unseen-participant scope: https://github.com/neural-interfaces26/2026-competition/blob/" + dataset["competition_commit"] + "/codabench/pages/scientific_faq_bci.md\n"
        + "[2] Official evaluator: https://github.com/neural-interfaces26/2026-competition/blob/" + dataset["competition_commit"] + "/benchmark_utils/streaming.py ; metric helpers: same commit, benchmark_utils/metrics.py\n"
        + "[3] Submission contract: https://github.com/neural-interfaces26/2026-competition/blob/" + dataset["competition_commit"] + "/codabench/pages/participate.md\n"
        + "[4] Public data: https://data.nemar.org/nm000290/v1.0.0/manifest.json ; retained data/nm000290/v1.0.0/dataset_description.json\n"
        + "[5] Selection and every measured table: v5_work/cv_analysis/selection.json; probabilities and provenance: v5_work/cv_predictions/assembly_manifest.json, stage_specific_leakage_check.json, and candidate NPZs\n"
        + "[6] Official runtime replay: v5_work/cv_streaming_check/streaming_replay_check.json and its hashed prediction artifacts\n"
        + "    Exact metric equality: v5_work/cv_analysis/official_metric_check.json ; pinned official source hashes: v5_work/reference_sources.json\n"
        + "[7] Protocol/preprocessing/environment: v5_work/plan.json, splits.json, cache/dataset_info.json, implementation_record.json, requirements-training.lock.txt, README_reproduce.txt\n"
        + ("[8] Final and legacy contract_check.json/build_info.json; v5_work/legacy_diagnostic/result.json, once_receipt.json and frozen_packages.json" if adopted else
           "[8] Null result: final/legacy submission artifacts and diagnostic are absent; submission runtime checks are not applicable."),
    ]) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    manifest = dict(schema="track2-v5-report-evidence-v1", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    outcome="PASS" if adopted else "NULL", selected_candidate=selected,
                    plan_sha256=FROZEN_PLAN_SHA256, report_sha256=sha256_file(args.output),
                    report_path=str(args.output.resolve()), n_cells=28, n_subjects=10,
                    n_cv_replicates=3, cv_leakage_audit_passed=True,
                    submission_checks="17 passed on each final/legacy package" if adopted else "not applicable: no new submission",
                    new_scoring_performed=False, new_fitting_performed=False, inputs_sha256=evidence.files)
    write_json_exclusive(args.output.with_name("report_evidence.json"), manifest)
    print(json.dumps(dict(outcome=manifest["outcome"], selected_candidate=selected,
                          output=str(args.output), words_including_tables=len(text.split()),
                          report_sha256=manifest["report_sha256"])), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "v5_work/plan.json")
    parser.add_argument("--selection", type=Path, default=ROOT / "v5_work/cv_analysis/selection.json")
    parser.add_argument("--selection-freeze", type=Path, default=ROOT / "v5_work/cv_analysis/selection_freeze.json")
    parser.add_argument("--assembly", type=Path, default=ROOT / "v5_work/cv_predictions/assembly_manifest.json")
    parser.add_argument("--cv-audit", type=Path, default=ROOT / "v5_work/cv_predictions/stage_specific_leakage_check.json")
    parser.add_argument("--replay", type=Path, default=ROOT / "v5_work/cv_streaming_check/streaming_replay_check.json")
    parser.add_argument("--official-metric-check", type=Path, default=ROOT / "v5_work/cv_analysis/official_metric_check.json")
    parser.add_argument("--reference-sources", type=Path, default=ROOT / "v5_work/reference_sources.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--environment-record", type=Path, default=ROOT / "v5_work/implementation_record.json")
    parser.add_argument("--requirements", type=Path, default=ROOT / "v5_work/requirements-training.lock.txt")
    parser.add_argument("--dataset-description", type=Path, default=ROOT / "data/nm000290/v1.0.0/dataset_description.json")
    parser.add_argument("--splits", type=Path, default=ROOT / "v5_work/splits.json")
    parser.add_argument("--classical-summary", type=Path, default=ROOT / "v5_work/classical_results/cv_summary.json")
    parser.add_argument("--final-dir", type=Path, default=ROOT / "v5_deliverables")
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "v5_work/legacy_diagnostic_model")
    parser.add_argument("--legacy-result", type=Path, default=ROOT / "v5_work/legacy_diagnostic/result.json")
    parser.add_argument("--output", type=Path, default=ROOT / "v5_deliverables/report.txt")
    write_report(parser.parse_args())
