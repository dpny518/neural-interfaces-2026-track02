"""Train-subject LOSO of the frozen temporal spatial-logvariance recipe.

Fold labels are read only after fixed-epoch training finishes. Checkpoint
selection is never performed on the held-out fold. The source prior is
refitted on the five fitting subjects; session alignment remains causal.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from evaluation import cell_mean_score
from crossvalidate import add_subject_diagnostics, take_meta
from temporal_sweep import cell_class_weights
from neural_cov import SpatialLogVarianceNet, fit_source_reference
from train_neural_cov import stream_covariances


def fit_fixed(c_train, y_train, loss_weights, config, seed=33, epochs=75, schedule_epochs=120):
    """Original training loop, with no heldout data or checkpoint selection."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SpatialLogVarianceNet(**config)
    optim = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=schedule_epochs, eta_min=.0001)
    criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=.05)
    X, Y, W = torch.from_numpy(c_train), torch.from_numpy(y_train), torch.from_numpy(loss_weights.astype(np.float32))
    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(X))
        losses = []
        for start in range(0, len(X), 96):
            idx = order[start:start + 96]
            if len(idx) < 2:
                continue
            batch = X[idx].clone()
            gains = torch.exp(torch.randn(len(idx), 1, batch.shape[-1]) * .08)
            batch = batch * gains.unsqueeze(-1) * gains.unsqueeze(-2)
            logits = net(batch)
            loss = (criterion(logits, Y[idx]) * W[idx]).mean()
            w = torch.nn.functional.normalize(net.spatial_weight, dim=-1)
            gram = w @ w.transpose(-1, -2)
            eye = torch.eye(gram.shape[-1])
            loss = loss + .001 * ((gram - eye) ** 2).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optim.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % 15 == 0:
            history.append({"epoch": epoch, "training_loss": float(np.mean(losses))})
            print("training", json.dumps(history[-1]), flush=True)
    return net.eval(), history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--schedule-epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    root = Path(__file__).resolve().parent
    out = root / "cv_results"
    models = out / "neural_loso_models"
    models.mkdir(parents=True, exist_ok=True)
    selected = torch.load(root / f"neural_results/temporal/spatial_causal_seed{args.seed}.pt", map_location="cpu", weights_only=True)
    model_config, alignment = selected["model_config"], selected["alignment_config"]
    assert selected["epoch"] == args.epochs
    assert selected["csp_initialized"] is False
    with np.load(root / "cache/metadata.npz") as z:
        train_rows = np.flatnonzero(z["split"] == "train")
        meta = {k: z[k][train_rows] for k in z.files}
    c = np.asarray(np.load(root / "cache/temporal_band_covs.npy", mmap_mode="r")[train_rows])
    logits_oof = np.empty((len(c), 3), np.float32)
    probabilities = np.empty((len(c), 3), np.float32)
    folds = []
    started = time.perf_counter()
    for subject in sorted(set(meta["subject"])):
        print("START FOLD", subject, flush=True)
        fold_started = time.perf_counter()
        fit = meta["subject"] != subject
        held = ~fit
        source = fit_source_reference(c[fit], meta["stream_id"][fit], normalization=alignment["normalization"])
        c_train = stream_covariances(c[fit], meta["stream_id"][fit], source, alignment)
        # Do not let heldout EEG or labels enter the training function.
        net, history = fit_fixed(c_train, meta["y"][fit], cell_class_weights(take_meta(meta, fit)),
                                 model_config, seed=args.seed, epochs=args.epochs, schedule_epochs=args.schedule_epochs)
        del c_train
        c_held = stream_covariances(c[held], meta["stream_id"][held], source, alignment)
        with torch.inference_mode():
            logits = torch.cat([net(x) for x in torch.from_numpy(c_held).split(128)])
            probabilities[held] = logits.softmax(-1).numpy()
            logits_oof[held] = logits.numpy()
        del c_held
        score = cell_mean_score(meta["y"][held], probabilities[held].argmax(1), take_meta(meta, held))
        folds.append({"heldout_subject": subject, "n_fit": int(fit.sum()), "score": score,
                      "epoch": args.epochs, "scheduler_T_max": args.schedule_epochs, "training_history": history,
                      "elapsed_seconds": time.perf_counter()-fold_started})
        torch.save({"model_type": "spatial_logvariance", "model_config": model_config,
                    "state_dict": net.state_dict(), "alignment_config": alignment,
                    "source": torch.from_numpy(source), "seed": args.seed, "epoch": args.epochs,
                    "scheduler_T_max": args.schedule_epochs,
                    "training_subjects": sorted(set(meta["subject"][fit])), "heldout_subject": subject,
                    "checkpoint_selection": "fixed epoch; heldout labels not used for selection"},
                   models / f"temporal_seed{args.seed}_hold_{subject}.pt")
        (out / f"neural_temporal_seed{args.seed}_folds.json").write_text(json.dumps(folds, indent=2) + "\n")
        print("DONE FOLD", subject, score["balanced_accuracy_cell_mean"], "seconds", round(time.perf_counter()-fold_started, 1), flush=True)
    score = add_subject_diagnostics(cell_mean_score(meta["y"], probabilities.argmax(1), meta))
    result = {"name": f"neural_temporal_seed{args.seed}", "oof": score, "folds": folds,
              "fixed_epoch": args.epochs, "scheduler_T_max": args.schedule_epochs,
              "model_config": model_config, "alignment_config": alignment,
              "test_evaluated": False, "elapsed_seconds": time.perf_counter()-started}
    np.savez_compressed(out / f"neural_temporal_seed{args.seed}_oof.npz", oof_probabilities=probabilities,
                        oof_logits=logits_oof, train_row_indices=train_rows,
                        config_json=np.array(json.dumps(result)))
    (out / f"neural_temporal_seed{args.seed}_oof.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FINISHED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
