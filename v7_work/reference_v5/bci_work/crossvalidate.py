"""Compact training-subject LOSO checks for shortlisted causal ridge models.

Each fold estimates source means, scalers, and classifiers on five training
subjects only. The held-out subject is replayed in order with independent
session state. The public test split is never read/scored by this script.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from evaluation import cell_mean_score
from temporal_sweep import cell_class_weights, fit_head, logits_head
from classical_sweep import ridge_path
from riemann_features import mean_by_group, center_feature_sessions


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cv_results"


def take_meta(meta, select):
    return {k: v[select] for k, v in meta.items()}


def add_subject_diagnostics(score):
    subjects = sorted({c["subject"] for c in score["cells"]})
    score["subject_cell_means"] = {
        subject: float(np.mean([c["balanced_accuracy"] for c in score["cells"] if c["subject"] == subject]))
        for subject in subjects
    }
    score["subject_mean_balanced_accuracy"] = float(np.mean(list(score["subject_cell_means"].values())))
    return score


def raw_head(head):
    coef = head["coef"] / head["scaler_scale"][None, :]
    intercept = head["intercept"] - coef @ head["scaler_mean"]
    return coef, intercept


def mix_heads(heads, weights=None):
    weights = np.ones(len(heads)) / len(heads) if weights is None else np.asarray(weights)
    raw = [raw_head(h) for h in heads]
    coef = sum(w * h[0] for w, h in zip(weights, raw))
    intercept = sum(w * h[1] for w, h in zip(weights, raw))
    return {"scaler_mean": np.zeros(coef.shape[1], np.float32),
            "scaler_scale": np.ones(coef.shape[1], np.float32),
            "coef": coef.astype(np.float32), "intercept": intercept.astype(np.float32)}


def fit_for_config(F, meta, cfg):
    weights = cell_class_weights(meta)
    if cfg.get("family") == "classical":
        alpha, _, mean, scale, coef, bias = next(ridge_path(
            F, meta["y"], F[:1], weights, [cfg["alpha"]], cfg["scale_mode"], 861))
        if isinstance(coef, tuple):
            coef = coef[0] @ (coef[1] @ coef[2])
        # RidgeClassifier uses {-1,+1}, the classical path uses {0,1}.
        # Convert now so OOF logits are comparable for later ensembling.
        return {"scaler_mean": mean, "scaler_scale": scale,
                "coef": (2. * coef).T, "intercept": 2. * bias - 1.}
    shared = fit_head(F, meta["y"], weights, cfg["alpha"], cfg["scaling"])
    if cfg.get("interface_experts", False):
        experts = []
        for interface in ("BrainHero", "Graz"):
            mask = meta["interface"] == interface
            # Match the shared fit's average-loss regularization strength.
            alpha = cfg["alpha"] * int(mask.sum()) / len(mask)
            experts.append(fit_head(F[mask], meta["y"][mask], cell_class_weights(take_meta(meta, mask)), alpha, cfg["scaling"]))
        expert_mean = mix_heads(experts)
        strength = cfg.get("expert_weight", 1.0)
        return mix_heads([shared, expert_mean], [1.0 - strength, strength])
    return shared


def main():
    threadpool_limits(limits=2)
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT / "cache/metadata.npz") as z:
        metadata = {k: z[k] for k in z.files}
    train_rows = np.flatnonzero(metadata["split"] == "train")
    val_rows = np.flatnonzero(metadata["split"] == "val")
    train_meta, val_meta = take_meta(metadata, train_rows), take_meta(metadata, val_rows)
    temporal = json.loads((ROOT / "temporal_results/scores.json").read_text())
    key_map = {c["key"]: c for c in temporal}
    keys = ["early_late_s1_p8_global_a1000", "early_late_s1_p8_global_a10000",
            "early_late_s1_p32_global_a1000", "early_late_s1_p32_global_a10000"]
    keys.append(next(c["key"] for c in temporal if c["view"] == "early_late" and c["center"]["strength"] == 0.))
    keys.append(next(c["key"] for c in temporal if c["view"] == "early"))
    keys.append(next(c["key"] for c in temporal if c["view"] == "late"))
    configs = [{**key_map[key], "family": "temporal"} for key in keys]
    with np.load(ROOT / "classical/ridge_095.npz") as z:
        cfg = json.loads(z["config_json"].item())
    configs.append({**cfg, "key": cfg["name"], "family": "classical", "view": "full",
                    "center": cfg["center_kwargs"], "val": cfg["score_validation"]})
    # Train separate interface heads, but at inference always average their
    # predictions. No interface labels or run counter are model inputs.
    for alpha in (1000., 10000.):
        base = key_map[f"early_late_s1_p8_global_a{alpha:g}"]
        for weight in (1.0, .5):
            configs.append({**base, "family": "temporal", "key": f"interface_avg_w{weight:g}_a{alpha:g}",
                            "interface_experts": True, "expert_weight": weight})
    (OUT / "shortlist.json").write_text(json.dumps(configs, indent=2) + "\n")

    # Only select train/validation rows from feature caches. Cached features
    # are strictly current-window functions, containing no fitted statistics.
    raw = {}
    for view in ("early", "late", "full"):
        fn = "raw_log_features.npy" if view == "full" else f"raw_{view}_log_features.npy"
        a = np.load(ROOT / "cache" / fn, mmap_mode="r")
        raw[view] = (np.array(a[train_rows], copy=True), np.array(a[val_rows], copy=True))
    raw["early_late"] = tuple(np.concatenate([raw["early"][i], raw["late"][i]], axis=1) for i in (0, 1))
    results = []
    all_started = time.perf_counter()
    for cfg in configs:
        start = time.perf_counter()
        F, Fval = raw[cfg["view"]]
        oof = np.empty((len(F), 3), dtype=np.float64)
        folds = []
        for subject in sorted(set(train_meta["subject"])):
            fit = train_meta["subject"] != subject
            held = ~fit
            source = mean_by_group(F[fit], train_meta["stream_id"][fit])
            # Both sets start every session from source-only state, then
            # update after transforming each current window.
            centered_train = center_feature_sessions(F[fit], train_meta["stream_id"][fit], source, **cfg["center"])
            centered_held = center_feature_sessions(F[held], train_meta["stream_id"][held], source, **cfg["center"])
            head = fit_for_config(centered_train, take_meta(train_meta, fit), cfg)
            oof[held] = logits_head(centered_held, head)
            fold = cell_mean_score(train_meta["y"][held], oof[held].argmax(1), take_meta(train_meta, held))
            folds.append({"heldout_subject": subject, "n_fit": int(fit.sum()), **fold})
            print(cfg["key"], subject, f"{fold['balanced_accuracy_cell_mean']:.5f}", flush=True)
        score = add_subject_diagnostics(cell_mean_score(train_meta["y"], oof.argmax(1), train_meta))

        # Final train-six head, using identical train-only fitted quantities.
        source = mean_by_group(F, train_meta["stream_id"])
        train_centered = center_feature_sessions(F, train_meta["stream_id"], source, **cfg["center"])
        val_centered = center_feature_sessions(Fval, val_meta["stream_id"], source, **cfg["center"])
        head = fit_for_config(train_centered, train_meta, cfg)
        val_logits = logits_head(val_centered, head)
        val_score = add_subject_diagnostics(cell_mean_score(val_meta["y"], val_logits.argmax(1), val_meta))
        record = {"key": cfg["key"], "config": cfg, "oof": score, "val": val_score,
                  "folds": folds, "elapsed_seconds": time.perf_counter() - start,
                  "logit_encoding": "ridge {-1,+1} class targets", "test_evaluated": False}
        np.savez_compressed(OUT / (cfg["key"] + ".npz"),
                            oof_logits=oof, train_row_indices=train_rows,
                            val_logits=val_logits, val_row_indices=val_rows,
                            source_mean=source, **head, config_json=np.array(json.dumps(cfg)))
        (OUT / (cfg["key"] + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("DONE", cfg["key"], "oof_cell", score["balanced_accuracy_cell_mean"],
              "oof_subject", score["subject_mean_balanced_accuracy"], "val", val_score["balanced_accuracy_cell_mean"],
              "seconds", round(time.perf_counter()-start, 1), flush=True)
    summary = [{"key": r["key"], "oof_cell": r["oof"]["balanced_accuracy_cell_mean"],
                "oof_subject": r["oof"]["subject_mean_balanced_accuracy"],
                "val_cell": r["val"]["balanced_accuracy_cell_mean"],
                "oof_subjects": r["oof"]["subject_cell_means"]} for r in results]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FINISHED", json.dumps(summary), "seconds", time.perf_counter()-all_started, flush=True)


if __name__ == "__main__":
    main()
