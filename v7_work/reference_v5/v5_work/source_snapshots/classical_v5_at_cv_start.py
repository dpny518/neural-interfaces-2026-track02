"""Fixed v3 ridge and one fold-isolated Riemannian tangent-space ablation.

No parameter search occurs here. Both deterministic branches share the v3
causal aligner, class/cell weighting, global feature scale, regularization,
temperature and later 10% ensemble weight. The tangent branch changes only
the origin of the full matrix logarithm, fitting an affine-invariant mean on
training-aligned covariances with equal-session weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bci_work"), str(ROOT / "methods_draft")]

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from build_submission import tensorize
from classical_sweep import class_cell_weights, ridge_path
from evaluation import cell_mean_score
from riemann_features import estimate_source, spd_power, sym_to_vec
from v5_work.riemann_v5_runtime import V5AlignedCovarianceStream, V5TangentMember, v5_tangent_features
from v5_work.protocol_v5 import (
    PUBLIC_SUBJECTS, LEGACY_TRAIN_SUBJECTS, LEGACY_REPORT_SUBJECTS,
    assert_stage_subjects, make_provenance, register_training_artifact,
    check_training_artifacts, sha256_file,
)

CENTER = dict(kind="logcov", alignment="euclidean", strength=1., prior_count=8.,
              ema_rate=.02, shrinkage=.05, update_clip=10.)
SPEC = dict(
    baseline="exact v3 temporal Euclidean log-covariance ridge",
    alpha=10000., scale_mode="global", temperature=.125, ensemble_weight=.1,
    center_kwargs=CENTER, shrinkage=.05, n_bands=14, band_dimension=861,
    tangent_origin="affine-invariant geometric mean of training-aligned covariances",
    tangent_mean_weights="equal session; uniform within session; no labels",
    tangent_solver=dict(initialization="equal-session arithmetic mean", max_iterations=12,
                        tolerance=1e-7, initial_step=1., backtracking_factor=.5,
                        max_backtracking=12, acceptance="nonincreasing weighted squared affine distance"),
    checkpoints="fixed deterministic fit; no held-out parameter or checkpoint selection",
    source_reference="equal-session raw covariance; training-only",
    feature_scaling="training-only weighted global scale, exactly as v3",
    no_test_role=True,
)


def matrix_exp_symmetric(matrix):
    matrix = (matrix + matrix.T) * .5
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.exp(values)[None, :]) @ vectors.T


def riemannian_mean(covariances, weights, *, max_iterations=12, tolerance=1e-7,
                    max_backtracking=12):
    """Weighted Karcher iteration with deterministic objective backtracking.

    All matrices in this function belong to the current fitting partition.
    Numerical stopping uses only the unsupervised training objective.
    """
    c = np.asarray(covariances, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if c.ndim != 3 or not len(c) or w.shape != (len(c),):
        raise ValueError("Expected a nonempty matrix sample and one weight per sample")
    if np.any(w <= 0) or not np.isfinite(c).all():
        raise ValueError("Invalid covariance sample or weights")
    w = w / w.sum()
    reference = np.einsum("n,nij->ij", w, c)

    def evaluate(mean):
        inverse = spd_power(mean, -.5)
        logs = spd_power(inverse @ c @ inverse.T, "log")
        direction = np.einsum("n,nij->ij", w, logs)
        objective = float(np.einsum("n,nij,nij->", w, logs, logs))
        return direction, objective

    direction, objective = evaluate(reference)
    history = []
    converged = False
    for iteration in range(max_iterations):
        residual = float(np.linalg.norm(direction, "fro"))
        history.append(dict(iteration=iteration, objective=objective, residual=residual))
        if residual <= tolerance:
            converged = True
            break
        root = spd_power(reference, .5)
        accepted = False
        step = 1.
        for _ in range(max_backtracking + 1):
            candidate = root @ matrix_exp_symmetric(step * direction) @ root.T
            candidate = (candidate + candidate.T) * .5
            candidate_direction, candidate_objective = evaluate(candidate)
            if candidate_objective <= objective + 1e-12 * max(1., abs(objective)):
                reference, direction, objective = candidate, candidate_direction, candidate_objective
                accepted = True
                break
            step *= .5
        history[-1].update(accepted_step=step if accepted else 0.)
        if not accepted:
            break
    residual = float(np.linalg.norm(direction, "fro"))
    converged = converged or residual <= tolerance
    return reference, dict(converged=converged, residual=residual, objective=objective,
                           n_updates=sum(h.get("accepted_step", 0.) > 0 for h in history), history=history)


def equal_session_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def fit_tangent_origin(aligned, groups):
    weights = equal_session_weights(groups)
    references, diagnostics = [], []
    for band in range(aligned.shape[1]):
        reference, diagnostic = riemannian_mean(aligned[:, band], weights)
        references.append(reference)
        diagnostics.append(dict(band=band, **diagnostic))
    references = np.stack(references)
    return references, spd_power(references, -.5), diagnostics


def aligned_sessions(covariances, groups, source):
    c, groups = np.asarray(covariances), np.asarray(groups)
    result = np.empty(c.shape, dtype=np.float64)
    for key in np.unique(groups):
        indices = np.flatnonzero(groups == key)
        stream = V5AlignedCovarianceStream(source, **CENTER)
        for row in indices:
            result[row] = stream.transform_covariance_one(c[row])
        assert stream.n_seen == len(indices)
    return result


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64) / SPEC["temperature"]
    if len(logits) == 0:
        return np.empty((0, 3), dtype=np.float64)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def subset(meta, rows):
    return {key: value[rows] for key, value in meta.items()}


def fit_pair(cov, metadata, fit, held, *, stage, heldout, selection_freeze, plan_sha256, out,
             branches=("ridge", "tangent")):
    """Fit both fixed branches; score labels are used only after fitting ends."""
    fm, hm = subset(metadata, fit), subset(metadata, held)
    ids = assert_stage_subjects(fm["subject"], stage=stage, heldout_subject=heldout,
                               selection_freeze=selection_freeze)
    if len(held):
        evaluation_stage = "cv_evaluate" if stage == "cv_fit" else "legacy_evaluate"
        assert_stage_subjects(hm["subject"], stage=evaluation_stage, heldout_subject=heldout,
                              selection_freeze=selection_freeze)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / (name + ".pt")).exists() or (out / (name + ".provenance.json")).exists()
           for name in branches):
        raise RuntimeError("Refusing to overwrite an existing registered fit")
    started = time.perf_counter()
    # Fitting functions receive only fitting rows, never the held-out arrays.
    source = estimate_source(cov[fit], fm["stream_id"], shrinkage=.05)
    train_aligned = aligned_sessions(cov[fit], fm["stream_id"], source)
    train_features = sym_to_vec(spd_power(train_aligned, "log")).astype(np.float32)
    diagnostics = []
    if "tangent" in branches:
        tangent_origin, tangent_inverse, diagnostics = fit_tangent_origin(train_aligned, fm["stream_id"])
        train_tangent = v5_tangent_features(train_aligned, tangent_inverse)
    del train_aligned
    # Held-out EEG now updates only a fresh session stream, after fitted origins.
    if len(held):
        held_aligned = aligned_sessions(cov[held], hm["stream_id"], source)
        held_features = sym_to_vec(spd_power(held_aligned, "log")).astype(np.float32)
        if "tangent" in branches:
            held_tangent = v5_tangent_features(held_aligned, tangent_inverse)
        del held_aligned
    else:
        held_features = np.empty((0, train_features.shape[1]), dtype=np.float32)
        if "tangent" in branches:
            held_tangent = np.empty_like(held_features)
    weights = class_cell_weights(fm["y"], fm)
    predictions, records, manifests = {}, {}, []
    feature_pairs = {"ridge": (train_features, held_features)}
    if "tangent" in branches:
        feature_pairs["tangent"] = (train_tangent, held_tangent)
    for name in branches:
        ft, fh = feature_pairs[name]
        _, logits, mean, scale, coefficient, bias = next(
            ridge_path(ft, fm["y"], fh, weights, [10000.], "global", 861))
        if isinstance(coefficient, tuple):
            coefficient = coefficient[0] @ (coefficient[1] @ coefficient[2])
        np.testing.assert_allclose((fh - mean) / scale @ coefficient + bias, logits,
                                   rtol=1e-9, atol=1e-9)
        config = dict(SPEC, branch=name, plan_sha256=plan_sha256)
        provenance = make_provenance(
            stage=stage, fitted_subjects=ids, heldout_subject=heldout,
            pretraining_subjects=(), normalization_subjects=ids, reference_subjects=ids,
            feature_scale_subjects=ids, batchnorm_subjects=(), checkpoint_selection_subjects=(),
            seed=0, config=config, row_indices=fit, selection_freeze=selection_freeze)
        data = dict(type="linear" if name == "ridge" else "riemannian_tangent",
                    family="euclidean" if name == "ridge" else "riemannian_tangent",
                    views=[dict(crop=[0, 240]), dict(crop=[240, 480])], shrinkage=.05,
                    center_kwargs=CENTER, scaler_mean=mean, scaler_scale=scale,
                    coef=coefficient, intercept=bias, temperature=.125,
                    source_cov=source["cov"], source_logcov=source["logcov"],
                    source_logpower=source["logpower"], feature_indices=np.arange(coefficient.shape[0]),
                    provenance=provenance)
        if name == "tangent":
            data.update(tangent_origin=tangent_origin, tangent_inverse_sqrt=tangent_inverse,
                        tangent_solver_diagnostics=diagnostics)
        path = out / (name + ".pt")
        torch.save(tensorize(data), path)
        manifest = register_training_artifact(path, provenance)
        manifests.append(str(manifest))
        probabilities = softmax(logits)
        predictions[name] = probabilities
        if name == "tangent" and len(held):
            # A prefix runtime check uses exactly the saved parameters and one
            # first stream; it never affects fitting, selection or score choice.
            runtime = V5TangentMember(data)
            first_session = hm["stream_id"][0]
            rows = np.flatnonzero(hm["stream_id"] == first_session)[:5]
            actual = np.stack([runtime.predict_one(cov[held[row]]) for row in rows])
            np.testing.assert_allclose(actual, probabilities[rows], rtol=1e-7, atol=1e-9)
        metric = cell_mean_score(hm["y"], probabilities.argmax(1), hm) if len(held) else None
        records[name] = dict(checkpoint=str(path.relative_to(ROOT)), manifest=str(manifest),
                             metric=metric, deterministic=True, stage=stage, heldout_subject=heldout)
    records["elapsed_seconds"] = time.perf_counter() - started
    records["tangent_solver_all_bands_converged"] = all(d["converged"] for d in diagnostics)
    records["maximum_tangent_solver_residual"] = max((d["residual"] for d in diagnostics), default=None)
    (out / "fit_summary.json").write_text(json.dumps(records, indent=2) + "\n")
    return predictions, records, manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "v5_work/cache")
    parser.add_argument("--out", type=Path, default=ROOT / "v5_work/classical_results")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("cv", "final", "legacy"), default="cv")
    parser.add_argument("--subjects", nargs="*", choices=PUBLIC_SUBJECTS)
    parser.add_argument("--branches", nargs="+", choices=("ridge", "tangent"), default=["ridge", "tangent"])
    parser.add_argument("--selection-freeze", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    threadpool_limits(args.threads)
    torch.set_num_threads(args.threads)
    plan = json.loads(args.plan.read_text())
    if not plan.get("warmup_test_role_retired"):
        raise RuntimeError("A frozen option-A plan must explicitly retire the old test role")
    if plan.get("classical_definition") != SPEC:
        raise RuntimeError("Classical executable recipe differs from the frozen plan")
    plan_hash = sha256_file(args.plan)
    with np.load(args.cache / "metadata.npz", allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    assert_stage_subjects(metadata["subject"], stage="selection")
    covariances = np.load(args.cache / "temporal_band_covs.npy", mmap_mode="r")
    assert len(covariances) == len(metadata["subject"])
    args.out.mkdir(parents=True, exist_ok=True)
    spec_path = args.out / "predeclared_recipe.json"
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == SPEC
    else:
        spec_path.write_text(json.dumps(SPEC, indent=2) + "\n")
    rows = np.arange(len(covariances), dtype=np.int64)
    folds = args.subjects or list(PUBLIC_SUBJECTS) if args.stage == "cv" else [args.stage]
    oof = {name: np.full((len(rows), 3), np.nan) for name in args.branches}
    all_records, all_manifests = [], []
    for fold in folds:
        if args.stage == "cv":
            fit = rows[metadata["subject"] != fold]
            held = rows[metadata["subject"] == fold]
            stage, heldout = "cv_fit", fold
            destination = args.out / ("hold_" + fold)
        elif args.stage == "final":
            fit, held = rows, rows[:0]
            stage, heldout = "final_fit", None
            destination = args.out / "full_train"
        else:
            fit = rows[np.isin(metadata["subject"], LEGACY_TRAIN_SUBJECTS)]
            # Only fit the legacy-six model here. Its old warm-up evaluation
            # happens once later, through the frozen complete submission.
            held = rows[:0]
            stage, heldout = "legacy_fit", None
            destination = args.out / "legacy_six"
        predictions, records, manifests = fit_pair(
            covariances, metadata, fit, held, stage=stage, heldout=heldout,
            selection_freeze=args.selection_freeze, plan_sha256=plan_hash, out=destination,
            branches=args.branches)
        all_manifests.extend(manifests)
        all_records.append(dict(fold=fold, **records))
        for name in oof:
            oof[name][held] = predictions[name]
        print(json.dumps({"fold": fold, "seconds": records["elapsed_seconds"],
                          "ridge": records["ridge"]["metric"]["balanced_accuracy_cell_mean"] if len(held) and "ridge" in records else None,
                          "tangent": records["tangent"]["metric"]["balanced_accuracy_cell_mean"] if len(held) and "tangent" in records else None,
                          "tangent_converged": records["tangent_solver_all_bands_converged"]}), flush=True)
    complete = args.stage == "cv" and len(folds) == len(PUBLIC_SUBJECTS)
    if complete:
        assert all(np.isfinite(value).all() for value in oof.values())
    if args.stage == "cv":
        np.savez_compressed(args.out / "cv_predictions.npz", row_indices=rows,
                            **{name + "_probabilities": value for name, value in oof.items()},
                            subjects=metadata["subject"])
    result = dict(stage=args.stage, complete=complete, folds=all_records, plan_sha256=plan_hash,
                  warmup_test_role_retired=True, no_heldout_leakage=check_training_artifacts(all_manifests))
    (args.out / (args.stage + "_summary.json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
