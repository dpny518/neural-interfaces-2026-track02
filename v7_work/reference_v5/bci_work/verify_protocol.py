"""Check local preprocessing and stream/metric implementations against kit.

Uses one training recording, all metadata, and a deterministic dummy model.
It never uses validation/test labels to fit or select a model.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import torch
from mne.annotations import _annotations_starts_stops
from mne_bids import BIDSPath, read_raw_bids

from evaluation import balanced_accuracy, cell_mean_score, load_cache, predict_streams


class CountingModel:
    def __init__(self):
        self.count = 0

    def reset_state(self):
        self.count = 0

    def predict(self, X):
        # Current-window feature and mutable count exercise both ordering
        # and the new-copy/reset boundary without any label access.
        out = []
        for x in X:
            out.append((int(abs(float(x[0, 0]))) + self.count) % 3)
            self.count += 1
        return np.array(out, dtype=np.int64)


def main():
    root = Path(__file__).resolve().parent.parent
    kit = root / "reference_audit/2026-competition"
    neuro = root / "reference_audit/neuroai"
    sys.path.insert(0, str(kit))
    from benchmark_utils.metrics import balanced_accuracy as official_ba, group_scores
    from benchmark_utils.streaming import predict_streams as official_streams

    X, metadata, info = load_cache()
    report = {"competition_commit": info["competition_commit"], "neuroai_commit": info["neuroai_commit"], "checks": {}}
    rng = np.random.default_rng(417)
    pred = rng.integers(0, 3, len(X))
    local = cell_mean_score(metadata["y"], pred, metadata)
    context = (metadata["interface"] == "Graz").astype(int)
    cells = np.stack([metadata["stream_id"], context], axis=1)
    official = float(group_scores(official_ba, metadata["y"], pred, cells).mean())
    assert local["balanced_accuracy_cell_mean"] == official
    for true, predicted in [([0, 0], [1, 1]), ([0, 1], [0, 2]), ([0, 0, 1, 2], [0, 1, 1, 0])]:
        assert balanced_accuracy(true, predicted) == official_ba(true, predicted)
    report["checks"]["cell_metric_identical_to_official"] = True

    model = CountingModel()
    local_pred = predict_streams(model, X, metadata)

    def loader():
        for start in range(0, len(X), 17):
            sl = slice(start, start + 17)
            yield torch.from_numpy(np.array(X[sl], copy=True)), torch.from_numpy(metadata["y"][sl]), {
                "stream_id": metadata["stream_id"][sl], "record_id": metadata["record_id"][sl],
                "onset": metadata["onset"][sl], "context_id": context[sl]}
    true, official_pred, official_info = official_streams(model, loader())
    assert np.array_equal(local_pred, official_pred)
    assert model.count == 0
    report["checks"]["stateful_stream_replay_identical_to_official"] = True
    report["checks"]["original_model_remains_unmodified"] = True
    report["stream_replay_n_windows"] = len(X)
    report["stream_replay_n_sessions"] = int(len(np.unique(metadata["stream_id"])))

    # Execute the exact official filter-method ASTs in a small namespace;
    # avoiding the framework's large optional dependencies does not alter
    # these method bodies or the MNE defaults they call.
    source = neuro / "neuralset-repo/neuralset/extractors/neuro.py"
    parsed = ast.parse(source.read_text())
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_fir_n_samples":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "MneRaw":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in ("_notch_filter", "_band_filter"):
                    method.decorator_list = []
                    selected.append(method)
    namespace = {"mne": mne, "np": np, "functools": functools, "logger": logging.getLogger(__name__),
                 "_annotations_starts_stops": _annotations_starts_stops}
    code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)
    source_file = str(metadata["source_file"][0])
    ent = dict(item.split("-", 1) for item in Path(source_file).stem.split("_") if "-" in item)
    bids = BIDSPath(root=root / "data/nm000290/v1.0.0", subject=ent["sub"], session=ent["ses"],
                    task=ent["task"], run=ent["run"], datatype="eeg", suffix="eeg", extension=".edf")
    raw = read_raw_bids(bids, verbose="ERROR")
    raw.set_montage("standard_1020")
    raw.pick(("eeg",), verbose=False)
    raw.load_data()
    namespace["_notch_filter"](raw, [50.0, 60.0], 1)
    namespace["_band_filter"](SimpleNamespace(mne_cpus=1, on_filter_too_long="raise"), raw, (0.1, 75.0), None)
    raw.resample(120.0, n_jobs=1, verbose=False)
    official_signal = raw.get_data().astype(np.float32)
    assert raw.ch_names == info["ch_names"]
    select = np.flatnonzero(metadata["source_file"] == source_file)
    reconstructed = np.stack([official_signal[:, metadata["start_sample"][i]:metadata["start_sample"][i] + 480] * np.float32(1e6) for i in select])
    diff = float(np.max(np.abs(reconstructed - X[select])))
    assert np.array_equal(reconstructed, X[select]), diff
    report["checks"]["mne_bids_exact_official_filter_methods_match_cached_run"] = True
    report["preprocessing_checked_training_recording"] = source_file
    report["preprocessing_checked_n_windows"] = int(len(select))
    report["preprocessing_max_absolute_difference_uV"] = diff
    report["mne_version"] = mne.__version__
    report["all_passed"] = all(report["checks"].values())
    output = Path(__file__).resolve().parent / "cache/protocol_check.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
