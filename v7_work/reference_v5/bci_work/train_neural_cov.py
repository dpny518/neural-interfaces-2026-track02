"""Train and select covariance neural decoders using training/validation only.

The script never reads test targets or computes test predictions. The final
evaluation is the parent's separately frozen test pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from neural_cov import CausalCovarianceAligner, SpatialLogVarianceNet, fit_source_reference, shrink, invsqrt


def cell_score(y, pred, subjects, sessions, interfaces):
    keys = np.array([f"{s}/{r}/{i}" for s, r, i in zip(subjects, sessions, interfaces)])
    scores = {k: float(balanced_accuracy_score(y[keys == k], pred[keys == k])) for k in np.unique(keys)}
    return float(np.mean(list(scores.values()))), scores


def stream_covariances(covs, groups, source, config):
    result = np.empty(covs.shape, dtype=np.float32)
    for group in np.unique(groups):
        inds = np.flatnonzero(groups == group)
        aligner = CausalCovarianceAligner(source, **config)
        result[inds] = aligner.transform(covs[inds])
    return result


def initialize_csp(net, covs, labels, sample_weights):
    """Training-label-only one-versus-rest CSP initialization."""
    total = np.average(covs, axis=0, weights=sample_weights)
    whitener = invsqrt(shrink(total, .05))
    filters = []
    for label in np.unique(labels):
        idx = labels == label
        mean = np.average(covs[idx], axis=0, weights=sample_weights[idx])
        white = whitener @ mean @ whitener.swapaxes(-1, -2)
        _, eigvec = np.linalg.eigh(white)
        spatial = eigvec.swapaxes(-1, -2) @ whitener
        filters.append(spatial[:, [0, 1, -2, -1], :])
    learned = np.concatenate(filters, axis=1)
    n = min(net.n_filters, learned.shape[1])
    with torch.no_grad():
        net.spatial_weight[:, :n, :] = torch.from_numpy(learned[:, :n, :]).float()


def train_one(c_train, y_train, loss_weights, c_val, y_val, cell_val, config, seed, epochs, csp_init=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SpatialLogVarianceNet(**config)
    if csp_init:
        initialize_csp(net, c_train, y_train, loss_weights)
    optim = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=.0001)
    criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=.05)
    X = torch.from_numpy(c_train)
    Y = torch.from_numpy(y_train)
    W = torch.from_numpy(loss_weights.astype(np.float32))
    Xv = torch.from_numpy(c_val)
    best = None
    history = []
    best_score = -1.
    stale = 0
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(X))
        losses = []
        for start in range(0, len(X), 96):
            idx = order[start:start + 96]
            if len(idx) < 2:
                continue
            batch = X[idx].clone()
            # Multiplicative sensor gain perturbations preserve PSD. Mild
            # per-channel noise is a within-training augmentation only.
            gains = torch.exp(torch.randn(len(idx), 1, batch.shape[-1]) * .08)
            batch = batch * gains.unsqueeze(-1) * gains.unsqueeze(-2)
            logits = net(batch)
            loss = (criterion(logits, Y[idx]) * W[idx]).mean()
            # Filters within a band are softly encouraged to offer distinct
            # spatial views, instead of all converging to one direction.
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
        if epoch % 5 == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                logits = torch.cat([net(x) for x in Xv.split(128)])
                prob = torch.softmax(logits, dim=-1).numpy()
            score, by_cell = cell_score(y_val, prob.argmax(1), *cell_val)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": score, "cells": by_cell})
            if score > best_score + 1e-12:
                best_score = score
                best = {"state_dict": copy.deepcopy(net.state_dict()), "epoch": epoch,
                        "validation_probabilities": prob.copy(), "validation_cell_scores": by_cell}
                stale = 0
            else:
                stale += 5
            if epoch % 20 == 0 or epoch == 1:
                print(json.dumps({"seed": seed, "epoch": epoch, "val": score, "best": best_score}), flush=True)
            if epoch >= 60 and stale >= 45:
                break
    best["validation_score"] = best_score
    best["history"] = history
    return best


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cache", type=Path, default=here / "cache")
    ap.add_argument("--covariances", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=here / "neural_results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[33, 42])
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--modes", nargs="+", default=["static", "causal"])
    ap.add_argument("--n-filters", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=.5)
    ap.add_argument("--no-diagonal", action="store_true")
    ap.add_argument("--csp-init", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = np.load(args.cache / "metadata.npz")
    # Access labels by train/val subset only. No test labels or predictions
    # enter model selection, source estimation, or checkpoint files.
    split = metadata["split"]
    itrain, ival = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    if args.covariances is None:
        args.covariances = args.cache / "band_covariances.npy"
    covs = np.load(args.covariances, mmap_mode="r")
    groups = metadata["stream_id"]
    c_train = np.asarray(covs[itrain])
    c_val = np.asarray(covs[ival])
    y_train, y_val = metadata["y"][itrain], metadata["y"][ival]
    val_cells = tuple(metadata[k][ival] for k in ("subject", "session", "interface"))
    train_cells = np.array([f"{s}/{r}/{i}/{y}" for s, r, i, y in zip(metadata["subject"][itrain], metadata["session"][itrain], metadata["interface"][itrain], y_train)])
    _, inverse, counts = np.unique(train_cells, return_inverse=True, return_counts=True)
    loss_weights = 1. / counts[inverse]
    loss_weights /= loss_weights.mean()
    source = fit_source_reference(c_train, groups[itrain], normalization="broadband")
    net_config = dict(n_bands=c_train.shape[1], n_chans=c_train.shape[-1], n_filters=args.n_filters,
                      n_classes=3, dropout=args.dropout, include_diagonal=not args.no_diagonal, hidden=args.hidden)
    all_results = []
    all_probs = []
    started = time.monotonic()
    for mode in args.modes:
        config = dict(normalization="broadband", strength=0. if mode == "static" else 1.,
                      prior_count=16., ema_rate=.02, shrinkage=.05, update_clip=10.)
        print("Preparing covariance alignment " + mode, flush=True)
        train_aligned = stream_covariances(c_train, groups[itrain], source, config)
        val_aligned = stream_covariances(c_val, groups[ival], source, config)
        for seed in args.seeds:
            result = train_one(train_aligned, y_train, loss_weights, val_aligned, y_val, val_cells, net_config, seed, args.epochs, csp_init=args.csp_init)
            filename = f"spatial_{mode}_seed{seed}.pt"
            checkpoint = {"model_type": "spatial_logvariance", "model_config": net_config,
                          "state_dict": result["state_dict"], "alignment_config": config,
                          "source": torch.from_numpy(source), "seed": seed, "epoch": result["epoch"],
                          "validation_score": result["validation_score"], "training_subjects": sorted(np.unique(metadata["subject"][itrain]).tolist()),
                          "validation_subjects": sorted(np.unique(metadata["subject"][ival]).tolist()), "csp_initialized": args.csp_init}
            torch.save(checkpoint, args.output / filename)
            np.save(args.output / filename.replace(".pt", "_validation_probabilities.npy"), result["validation_probabilities"])
            all_probs.append(result["validation_probabilities"])
            record = {"filename": filename, "mode": mode, "seed": seed, "best_epoch": result["epoch"],
                      "validation_score": result["validation_score"], "cells": result["validation_cell_scores"],
                      "history": result["history"]}
            all_results.append(record)
            (args.output / "training_results.json").write_text(json.dumps(all_results, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)
    ensembles = []
    for count in [len(all_probs)]:
        p = np.mean(all_probs[:count], axis=0)
        score, by_cell = cell_score(y_val, p.argmax(1), *val_cells)
        ensembles.append({"members": [r["filename"] for r in all_results[:count]], "validation_score": score, "cells": by_cell})
        np.save(args.output / "ensemble_validation_probabilities.npy", p)
    summary = {"results": [{k: v for k, v in r.items() if k != "history"} for r in all_results],
               "ensembles": ensembles, "elapsed_seconds": time.monotonic() - started,
               "test_evaluated": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
