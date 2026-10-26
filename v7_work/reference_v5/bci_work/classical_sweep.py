"""Validation-only compact search for causal spectral/covariance classifiers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "methods_draft"))
from riemann_features import center_feature_sessions, mean_by_group
from evaluation import cell_mean_score


def select_bands(raw, bands, dim_per_band):
    indices = np.concatenate([np.arange(b * dim_per_band, (b + 1) * dim_per_band)
                              for b in bands])
    return np.asarray(raw[:, indices], dtype=np.float64), indices


def class_cell_weights(y, meta):
    keys = np.array([str(a) + "/" + str(b) + "/" + str(c)
                     for a, b, c in zip(meta["subject"], meta["session"], meta["interface"])])
    weights = np.zeros(len(y), dtype=np.float64)
    cells = np.unique(keys)
    for cell in cells:
        for cls in range(3):
            selected = (keys == cell) & (y == cls)
            if selected.any():
                weights[selected] = 1. / selected.sum()
    return weights * len(weights) / weights.sum()


def ridge_path(x_train, y_train, x_val, weights, alphas, scale_mode, band_dim):
    """One eigendecomposition fits all alphas; returns deployment coefficients."""
    wn = weights / weights.sum()
    mean = np.einsum("n,nd->d", wn, x_train)
    centered = x_train - mean
    var = np.einsum("n,nd,nd->d", wn, centered, centered)
    if scale_mode == "standard":
        scale = np.sqrt(np.maximum(var, 1e-8))
    elif scale_mode == "band":
        scale = np.repeat(np.sqrt(np.maximum(var.reshape(-1, band_dim).mean(axis=1), 1e-8)), band_dim)
    elif scale_mode == "global":
        scale = np.full_like(var, np.sqrt(max(float(np.mean(var)), 1e-8)))
    else:
        raise ValueError(scale_mode)
    train = centered / scale
    val = (x_val - mean) / scale
    labels = np.eye(3)[y_train]
    bias = np.einsum("n,nc->c", wn, labels)
    root_w = np.sqrt(weights)
    xw, yw = train * root_w[:, None], (labels - bias) * root_w[:, None]
    if train.shape[1] <= train.shape[0]:
        eigen, vectors = eigh(xw.T @ xw, check_finite=False)
        rhs = vectors.T @ (xw.T @ yw)
        for alpha in alphas:
            coef = vectors @ (rhs / (np.maximum(eigen, 0.) + alpha)[:, None])
            logits = val @ coef + bias
            yield alpha, logits, mean, scale, coef, bias
    else:
        eigen, vectors = eigh(xw @ xw.T, check_finite=False)
        rhs = vectors.T @ yw
        # Save high-dimensional coefficients only for the retained candidates.
        xv = val @ xw.T @ vectors
        for alpha in alphas:
            adjusted = rhs / (np.maximum(eigen, 0.) + alpha)[:, None]
            logits = xv @ adjusted + bias
            yield alpha, logits, mean, scale, (xw.T, vectors, adjusted), bias


def main():
    start_time = time.perf_counter()
    cache = ROOT / "cache"
    out = ROOT / "classical"
    out.mkdir(exist_ok=True)
    with np.load(cache / "metadata.npz", allow_pickle=False) as z:
        metadata = {k: z[k] for k in z.files}
    train_mask, val_mask = metadata["split"] == "train", metadata["split"] == "val"
    train_meta = {k: v[train_mask] for k, v in metadata.items()}
    val_meta = {k: v[val_mask] for k, v in metadata.items()}
    y_train, y_val = train_meta["y"], val_meta["y"]
    weights = class_cell_weights(y_train, train_meta)
    # Test labels are never selected, scored, or used for any fitted quantity.
    raw_caches = {"logcov": np.load(cache / "raw_log_features.npy", mmap_mode="r"),
                  "logpower": np.load(cache / "raw_power_features.npy", mmap_mode="r")}
    # Broader bands first gives a quick deployable baseline while the rest runs.
    feature_views = [("logcov", [6]), ("logcov", [0, 1, 2, 3, 4, 5, 6]),
                     ("logcov", [1, 2, 3, 4, 5]), ("logcov", [2, 3, 4]),
                     ("logpower", [0, 1, 2, 3, 4, 5, 6])]
    adaptation = [dict(strength=1., prior_count=16., ema_rate=.02),
                  dict(strength=0., prior_count=16., ema_rate=.02),
                  dict(strength=1., prior_count=4., ema_rate=.02),
                  dict(strength=1., prior_count=16., ema_rate=.05),
                  dict(strength=1., prior_count=64., ema_rate=.02),
                  dict(strength=.7, prior_count=16., ema_rate=.02),
                  dict(strength=1., prior_count=4., ema_rate=.10)]
    alphas = [10., 100., 1000., 10000.]
    results = []
    retained = []
    counter = 0
    for kind, bands in feature_views:
        dim = 861 if kind == "logcov" else 41
        raw, feature_indices = select_bands(raw_caches[kind], bands, dim)
        source_mean = mean_by_group(raw[train_mask], train_meta["stream_id"])
        for center_kwargs in adaptation:
            train = center_feature_sessions(raw[train_mask], train_meta["stream_id"], source_mean, **center_kwargs)
            val = center_feature_sessions(raw[val_mask], val_meta["stream_id"], source_mean, **center_kwargs)
            for scale_mode in ["band", "standard"]:
                group_scores = []
                for alpha, logits, mean, scale, coef, bias in ridge_path(
                        train, y_train, val, weights, alphas, scale_mode, dim):
                    pred = logits.argmax(axis=1).astype(np.int64)
                    scores = cell_mean_score(y_val, pred, val_meta)
                    score = scores["balanced_accuracy_cell_mean"]
                    config = {"name": f"ridge_{counter:03d}", "family": "ridge",
                              "kind": kind, "shrinkage": .05, "bands": bands,
                              "crop": [0, 480], "center_kwargs": center_kwargs,
                              "scale_mode": scale_mode, "alpha": alpha,
                              "training_weights": "inverse class count per subject-session-interface cell",
                              "score_validation": score, "feature_dim": len(source_mean)}
                    results.append(config)
                    group_scores.append(round(score, 4))
                    if len(retained) < 20 or score > min(x[0] for x in retained):
                        if isinstance(coef, tuple):
                            coef = coef[0] @ (coef[1] @ coef[2])
                        weight_path = out / (config["name"] + ".npz")
                        np.savez_compressed(weight_path, source_mean=source_mean,
                                            scaler_mean=mean, scaler_scale=scale,
                                            coef=np.asarray(coef), intercept=bias,
                                            feature_indices=feature_indices,
                                            val_logits=logits, val_pred=pred,
                                            config_json=np.array(json.dumps(config)))
                        (out / (config["name"] + "_validation.json")).write_text(json.dumps(scores, indent=2))
                        retained.append((score, config["name"]))
                        retained.sort(reverse=True)
                        if len(retained) > 20:
                            _, removed = retained.pop()
                            (out / (removed + ".npz")).unlink(missing_ok=True)
                            (out / (removed + "_validation.json")).unlink(missing_ok=True)
                    counter += 1
                print(json.dumps({"kind": kind, "bands": bands, "center": center_kwargs,
                                  "scale": scale_mode, "alpha": alphas, "scores": group_scores,
                                  "best": retained[0], "seconds": round(time.perf_counter()-start_time, 1)}), flush=True)
            (out / "results.json").write_text(json.dumps(sorted(results, key=lambda x: -x["score_validation"]), indent=2))
            (out / "retained.json").write_text(json.dumps(retained, indent=2))
    print("FINISHED", len(results), "configurations", "seconds", time.perf_counter()-start_time, flush=True)


if __name__ == "__main__":
    main()
