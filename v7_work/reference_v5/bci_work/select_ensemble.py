"""Select a small probability ensemble on validation + train-subject LOSO.

The coarse grid is explicitly written before any ensemble score is computed.
No test probabilities or test labels are used by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation import cell_mean_score
from crossvalidate import add_subject_diagnostics


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.exp(x - x.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def main():
    root = Path(__file__).resolve().parent
    out = root / "cv_results"
    definitions = {
        "neural_members": ["temporal_causal_seed33_epoch75", "temporal_causal_seed42_epoch20"],
        "neural_member_weights": [.5, .5],
        "baseline_candidates": ["neural_seed33", "neural_seed42", "equal_neural_pair"],
        "ridge_candidates": ["euclidean_temporal_002", "euclidean_temporal_001",
                             "early_late_s1_p8_global_a1000", "early_late_s1_p32_global_a10000"],
        "ridge_weights": [0., .1, .25, .5], "ridge_temperatures": [.25, .5],
        "ridge_logit_encoding": "plus_minus_one",
        "selection_objective": "0.5 * validation_cell_mean + 0.5 * training_subject_LOSO_cell_mean",
        "tie_break": "smaller ridge weight, then listed candidate order", "test_evaluated": False,
    }
    (out / "ensemble_grid_definition.json").write_text(json.dumps(definitions, indent=2) + "\n")
    with np.load(root / "cache/metadata.npz") as z:
        train_indices = np.flatnonzero(z["split"] == "train")
        val_indices = np.flatnonzero(z["split"] == "val")
        train_meta = {k: z[k][train_indices] for k in z.files}
        val_meta = {k: z[k][val_indices] for k in z.files}
    neural = {}
    for seed in (33, 42):
        with np.load(out / f"neural_temporal_seed{seed}_oof.npz") as z:
            assert np.array_equal(z["train_row_indices"], train_indices)
            oof = z["oof_probabilities"]
        val = np.load(root / f"neural_results/temporal/spatial_causal_seed{seed}_validation_probabilities.npy")
        neural[seed] = (oof, val)
    pair_oof = (neural[33][0] + neural[42][0]) * .5
    pair_val = (neural[33][1] + neural[42][1]) * .5
    records = []

    def evaluate(name, oof, val, **config):
        o = add_subject_diagnostics(cell_mean_score(train_meta["y"], oof.argmax(1), train_meta))
        v = add_subject_diagnostics(cell_mean_score(val_meta["y"], val.argmax(1), val_meta))
        record = {"name": name, "config": config, "oof": o, "val": v,
                  "selection_objective": .5 * (o["balanced_accuracy_cell_mean"] + v["balanced_accuracy_cell_mean"])}
        return record

    baselines = [evaluate(f"neural_seed{seed}", *neural[seed], neural_seeds=[seed],
                          neural_weights=[1.], ridge_weight=0.) for seed in (33, 42)]
    base = evaluate("neural_seed33_seed42", pair_oof, pair_val,
                    neural_seeds=[33, 42], neural_weights=[.5, .5], ridge_weight=0.)
    records.extend(baselines + [base])
    candidate_arrays = {base["name"]: (pair_oof, pair_val),
                        **{f"neural_seed{seed}": neural[seed] for seed in (33, 42)}}
    for name in definitions["ridge_candidates"]:
        if name.startswith("euclidean_temporal"):
            index = name.split("_")[-1]
            with np.load(out / f"{name}_oof.npz") as z:
                assert np.array_equal(z["train_row_indices"], train_indices)
                lo = z["oof_logits"]
            with np.load(root / f"euclidean_temporal/euclidean_{index}.npz") as z:
                lv = 2. * z["val_logits"] - 1.
            ridge_path = f"euclidean_temporal/euclidean_{index}.npz"
        else:
            with np.load(out / f"{name}.npz") as z:
                assert np.array_equal(z["train_row_indices"], train_indices)
                assert np.array_equal(z["val_row_indices"], val_indices)
                lo, lv = z["oof_logits"], z["val_logits"]
            ridge_path = f"temporal_results/{name}.npz"
        for temperature in definitions["ridge_temperatures"]:
            po, pv = softmax(lo / temperature), softmax(lv / temperature)
            for weight in definitions["ridge_weights"][1:]:
                oof = (1. - weight) * pair_oof + weight * po
                val = (1. - weight) * pair_val + weight * pv
                key = f"neuralpair_plus_{name}_w{weight:g}_t{temperature:g}"
                record = evaluate(key, oof, val, neural_seeds=[33, 42],
                                  neural_weights=[(1. - weight)/2] * 2,
                                  ridge_name=name, ridge_path=ridge_path,
                                  ridge_weight=weight, ridge_temperature=temperature,
                                  ridge_logit_encoding="plus_minus_one")
                records.append(record)
                candidate_arrays[key] = (oof, val)
    records.sort(key=lambda r: (-r["selection_objective"], r["config"].get("ridge_weight", 0.)))
    winner = records[0]
    config = {"name": winner["name"], **winner["config"],
              "selection_objective": winner["selection_objective"],
              "validation_cell_mean": winner["val"]["balanced_accuracy_cell_mean"],
              "LOSO_cell_mean": winner["oof"]["balanced_accuracy_cell_mean"],
              "LOSO_subject_mean": winner["oof"]["subject_mean_balanced_accuracy"],
              "n_grid_candidates": len(records), "test_evaluated": False}
    (out / "ensemble_grid_results.json").write_text(json.dumps({"single_neural_baselines": baselines, "grid": records}, indent=2) + "\n")
    (out / "selected_ensemble.json").write_text(json.dumps(config, indent=2) + "\n")
    oof, val = candidate_arrays[winner["name"]]
    np.savez_compressed(out / "selected_ensemble_probabilities.npz", oof_probabilities=oof, val_probabilities=val,
                        train_row_indices=train_indices, val_row_indices=val_indices,
                        config_json=np.array(json.dumps(config)))
    print("BASELINES", json.dumps([{ "name": b["name"], "oof_cell": b["oof"]["balanced_accuracy_cell_mean"],
                                    "val_cell": b["val"]["balanced_accuracy_cell_mean"]} for b in baselines] +
                                   [{"name": base["name"], "oof_cell": base["oof"]["balanced_accuracy_cell_mean"],
                                     "val_cell": base["val"]["balanced_accuracy_cell_mean"]}]), flush=True)
    print("SELECTED", json.dumps(config), flush=True)
    print("TOP", json.dumps([{ "name": r["name"], "oof": r["oof"]["balanced_accuracy_cell_mean"],
                              "val": r["val"]["balanced_accuracy_cell_mean"], "objective": r["selection_objective"]}
                             for r in records[:8]]), flush=True)


if __name__ == "__main__":
    main()
