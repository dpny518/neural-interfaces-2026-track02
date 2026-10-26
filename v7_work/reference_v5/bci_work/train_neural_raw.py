"""Small raw-EEG baseline/augmentation sweep, validation selection only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from neural_raw import make_raw_model, window_zscore
from train_neural_cov import cell_score


def augment(x, rng):
    # Input normalization is identical at training and inference. Subsequent
    # perturbations are training-only, local to each supplied four-second cue.
    x = window_zscore(x)
    b, c, t = x.shape
    gains = torch.exp(torch.randn(b, c, 1) * .08)
    keep = (torch.rand(b, c, 1) > .025).float()
    x = x * gains * keep
    # One batch-wide shift keeps the implementation efficient on CPU. All
    # values are from this window; vacated samples are zero-filled.
    shift = int(rng.integers(-12, 13))
    if shift > 0:
        x = torch.cat([torch.zeros_like(x[..., :shift]), x[..., :-shift]], -1)
    elif shift < 0:
        x = torch.cat([x[..., -shift:], torch.zeros_like(x[..., : -shift])], -1)
    if rng.random() < .3:
        start = int(rng.integers(0, t - 20))
        x[..., start : start + 20] = 0.
    return x


def train_one(kind, seed, X, Y, W, Xv, yv, cell_val, epochs, patience):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_raw_model(kind)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=.0001)
    criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=.03)
    best_score = -1.
    best = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X))
        losses = []
        for start in range(0, len(X), 48):
            idx = order[start:start + 48]
            x = augment(X[idx], rng)
            logits = model(x, normalize=False)
            loss = (criterion(logits, Y[idx]) * W[idx]).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()
            model.constrain()
            losses.append(float(loss.detach()))
        schedule.step()
        model.eval()
        with torch.inference_mode():
            prob = torch.cat([torch.softmax(model(x), -1) for x in Xv.split(64)]).numpy()
        score, cells = cell_score(yv, prob.argmax(1), *cell_val)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_score": score, "cells": cells})
        if score > best_score + 1e-12:
            best_score = score
            best = {"model_type": kind, "model_config": model.config,
                    "state_dict": copy.deepcopy(model.state_dict()), "seed": seed, "epoch": epoch,
                    "validation_score": score, "validation_probabilities": prob.copy(), "cells": cells}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"kind": kind, "seed": seed, "epoch": epoch, "val": score, "best": best_score}), flush=True)
        if epoch >= 20 and stale >= patience:
            break
    best["history"] = history
    return best


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cache", type=Path, default=here / "cache")
    ap.add_argument("--output", type=Path, default=here / "raw_results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[33, 42])
    ap.add_argument("--kinds", nargs="+", default=["eegnet", "shallow"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = np.load(args.cache / "metadata.npz")
    tr = np.flatnonzero(meta["split"] == "train")
    va = np.flatnonzero(meta["split"] == "val")
    windows = np.load(args.cache / "windows.npy", mmap_mode="r")
    X = torch.from_numpy(np.asarray(windows[tr]))
    Xv = torch.from_numpy(np.asarray(windows[va]))
    y = meta["y"][tr]
    Y = torch.from_numpy(y)
    yv = meta["y"][va]
    cell_val = tuple(meta[k][va] for k in ("subject", "session", "interface"))
    keys = np.array([f"{s}/{t}/{i}/{label}" for s, t, i, label in zip(meta["subject"][tr], meta["session"][tr], meta["interface"][tr], y)])
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    weights /= weights.mean()
    W = torch.from_numpy(weights.astype(np.float32))
    summary = []
    probabilities = []
    started = time.monotonic()
    for kind in args.kinds:
        for seed in args.seeds:
            result = train_one(kind, seed, X, Y, W, Xv, yv, cell_val, args.epochs, args.patience)
            stem = f"{kind}_seed{seed}"
            np.save(args.output / (stem + "_validation_probabilities.npy"), result["validation_probabilities"])
            probabilities.append(result["validation_probabilities"])
            ckpt = {k: v for k, v in result.items() if k not in ("history", "validation_probabilities", "cells")}
            ckpt["training_subjects"] = sorted(np.unique(meta["subject"][tr]).tolist())
            ckpt["validation_subjects"] = sorted(np.unique(meta["subject"][va]).tolist())
            ckpt["preprocessing"] = "per-window per-channel population z-score"
            torch.save(ckpt, args.output / (stem + ".pt"))
            record = {k: v for k, v in result.items() if k not in ("state_dict", "validation_probabilities")}
            summary.append(record)
            (args.output / "training_results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)
    p = np.mean(probabilities, axis=0)
    score, cells = cell_score(yv, p.argmax(1), *cell_val)
    np.save(args.output / "ensemble_validation_probabilities.npy", p)
    output = {"results": [{k: v for k, v in r.items() if k != "history"} for r in summary],
              "ensemble": {"validation_score": score, "cells": cells},
              "elapsed_seconds": time.monotonic() - started, "test_evaluated": False}
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
