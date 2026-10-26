"""Track 2 cell-mean balanced accuracy and causal stream evaluation.

Replicates benchmark_utils/{metrics,streaming}.py at competition kit commit
a66bc23d1a0e686b895c093a642b45c74970758b. All grouping fields stay outside
the model; predict sees only its current EEG tensor.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


def load_cache(cache_dir=None, split=None, mmap_mode="r"):
    """Return (X, metadata, dataset_info); optionally select train/val/test."""
    cache_dir = Path(cache_dir or Path(__file__).resolve().parent / "cache")
    X = np.load(cache_dir / "windows.npy", mmap_mode=mmap_mode)
    with np.load(cache_dir / "metadata.npz", allow_pickle=False) as z:
        meta = {k: z[k] for k in z.files}
    info = json.loads((cache_dir / "dataset_info.json").read_text())
    if split is not None:
        select = meta["split"] == split
        X, meta = X[select], {k: value[select] for k, value in meta.items()}
    return X, meta, info


def balanced_accuracy(y_true, y_pred):
    """Official union-of-true-and-predicted-classes macro recall."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError((y_true.shape, y_pred.shape))
    classes = np.union1d(y_true, y_pred)
    if not len(classes):
        return float("nan")
    return float(np.mean([np.mean(y_pred[y_true == c] == c) if np.any(y_true == c) else 0.0 for c in classes]))


def cell_mean_score(y_true, y_pred, meta):
    """Return official score and diagnostic scores, including all cells."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Mismatched prediction shape {y_pred.shape} vs {y_true.shape}")
    if not np.issubdtype(y_pred.dtype, np.integer):
        raise TypeError(f"Predictions must be integer class indices, got {y_pred.dtype}")
    if not np.isin(y_pred, [0, 1, 2]).all():
        raise ValueError("Predicted class outside [0,2]")
    keys = list(zip(meta["subject"].astype(str), meta["session"].astype(str), meta["interface"].astype(str)))
    cells = []
    for subject, session, interface in sorted(set(keys)):
        select = ((meta["subject"] == subject) & (meta["session"] == session) & (meta["interface"] == interface))
        t, p = y_true[select], y_pred[select]
        cells.append({"subject": subject, "session": session, "interface": interface,
                      "n_windows": int(select.sum()), "balanced_accuracy": balanced_accuracy(t, p),
                      "class_counts": [int(np.sum(t == c)) for c in range(3)],
                      "class_recalls": [float(np.mean(p[t == c] == c)) if np.any(t == c) else None for c in range(3)]})
    sessions = []
    for subject, session in sorted(set((s, e) for s, e, _ in keys)):
        select = (meta["subject"] == subject) & (meta["session"] == session)
        sessions.append({"subject": subject, "session": session, "n_windows": int(select.sum()),
                         "balanced_accuracy": balanced_accuracy(y_true[select], y_pred[select])})
    return {"balanced_accuracy_cell_mean": float(np.mean([c["balanced_accuracy"] for c in cells])),
            "balanced_accuracy_session_mean": float(np.mean([s["balanced_accuracy"] for s in sessions])),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "accuracy": float(np.mean(y_true == y_pred)), "n_windows": len(y_true),
            "n_cells": len(cells), "n_sessions": len(sessions), "cells": cells, "sessions": sessions}


def predict_streams(model, X, meta, device="cpu"):
    """One window at a time, deepcopy/reset at each subject-session boundary.

    Returns np.int64 predictions in the original metadata row order. Inputs
    must already be in official acquisition order; the check rejects unsafe
    interleaving or reverse chronology instead of silently sorting them.
    """
    import torch
    n = len(X)
    if any(len(v) != n for v in meta.values()):
        raise ValueError("Metadata length mismatch")
    preds = np.empty(n, dtype=np.int64)
    current_key = None
    finished = set()
    last_run, last_onset = -1, -np.inf
    current = None
    for i in range(n):
        key = (str(meta["subject"][i]), str(meta["session"][i]))
        run = int(meta["run"][i])
        onset = float(meta["onset"][i])
        if key != current_key:
            if key in finished:
                raise ValueError("Stream resumes after a different stream")
            if current_key is not None:
                finished.add(current_key)
            current_key = key
            current = None
            current = copy.deepcopy(model)
            if callable(getattr(current, "reset_state", None)):
                current.reset_state()
            last_run, last_onset = -1, -np.inf
        if run < last_run or (run == last_run and onset < last_onset):
            raise ValueError("Windows are not in acquisition order")
        last_run, last_onset = run, onset
        inp = torch.as_tensor(np.array(X[i : i + 1], dtype=np.float32, copy=True), device=device)
        pred = current.predict(inp)
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred)
        if pred.shape != (1,) or not np.issubdtype(pred.dtype, np.integer):
            raise ValueError(f"predict returned {pred.shape} {pred.dtype}, expected integer (1,)")
        preds[i] = pred[0]
    return preds


def evaluate_model(model, X, meta, device="cpu"):
    """Return (score dictionary, predictions) after a causal stream replay."""
    pred = predict_streams(model, X, meta, device=device)
    return cell_mean_score(meta["y"], pred, meta), pred
