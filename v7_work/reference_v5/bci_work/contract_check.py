"""Executable contract/causality checks for a root-level Track 2 ZIP.

Only public TRAINING windows and synthetic signals are used. This does not
claim leaderboard validation; it checks local inference, state isolation,
causality, packaging, and the integer output contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from evaluation import load_cache


def _digest(model):
    """Content digest of arrays, tensors, scalars and nested model state."""
    h = hashlib.sha256()
    visited = {}

    def add(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            h.update(repr(value).encode())
        elif isinstance(value, torch.Tensor):
            a = value.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        elif isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())
        elif isinstance(value, (dict, list, tuple, set)) or hasattr(value, "__dict__"):
            key = id(value)
            if key in visited:
                h.update(f"REF:{visited[key]}".encode())
                return
            visited[key] = len(visited)
            h.update(type(value).__qualname__.encode())
            if isinstance(value, dict):
                for k in sorted(value, key=str):
                    add(k)
                    add(value[k])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    add(item)
            elif isinstance(value, set):
                for item in sorted(value, key=str):
                    add(item)
            else:
                add(vars(value))
        else:
            h.update(str(type(value)).encode() + str(value).encode())
    add(model)
    return h.hexdigest()


class _HideOfficialBase(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "benchmark_utils" or fullname.startswith("benchmark_utils."):
            raise ModuleNotFoundError("Forced local fallback test", name=fullname)
        return None


def _import_submission(path, module_name, force_fallback=False):
    blocker = _HideOfficialBase()
    removed = {}
    if force_fallback:
        removed = {k: v for k, v in list(sys.modules.items()) if k == "benchmark_utils" or k.startswith("benchmark_utils.")}
        for k in removed:
            del sys.modules[k]
        sys.meta_path.insert(0, blocker)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if force_fallback:
            sys.meta_path.remove(blocker)
            sys.modules.update(removed)


def _prediction(model, tensor, n_classes=3):
    original = tensor.clone()
    result = model.predict(tensor)
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy()
    result = np.asarray(result)
    assert result.shape == (len(tensor),), f"Output shape {result.shape} instead of {(len(tensor),)}"
    assert np.issubdtype(result.dtype, np.integer), f"Output dtype {result.dtype} is not integer"
    assert np.isfinite(result).all(), "Non-finite output"
    assert ((result >= 0) & (result < n_classes)).all(), "Class outside range"
    assert torch.equal(tensor, original), "predict mutated its input tensor"
    return result


def _reset_copy(model):
    out = copy.deepcopy(model)
    reset = getattr(out, "reset_state", None)
    assert callable(reset), "Expected explicit reset_state for the submitted stateful model"
    reset()
    return out


def _stream(model, frames):
    return np.concatenate([_prediction(model, x[None]) for x in frames])


def check_submission(submission_dir, output, zip_file=None, kit_dir=None):
    submission_dir = Path(submission_dir).resolve()
    output = Path(output).resolve()
    temporary = None
    report = {"all_passed": False, "device": "cpu", "data_used": "public train split + synthetic signals; no validation/test labels", "checks": []}
    started = time.perf_counter()

    def record(name, detail=None):
        report["checks"].append({"name": name, "passed": True, **({"detail": detail} if detail is not None else {})})

    try:
        if submission_dir.is_file():
            zip_file = submission_dir
            temporary = tempfile.TemporaryDirectory(prefix="bci-contract-")
            submission_dir = Path(temporary.name)
        if zip_file is not None:
            with zipfile.ZipFile(zip_file) as z:
                names = z.namelist()
                assert "submission.py" in names and "weights.pt" in names, names
                assert all("/" not in name and "\\" not in name for name in names), "ZIP files must be at root"
                assert [name for name in names if name.endswith(".py")] == ["submission.py"], "Only submission.py inference module is supported"
                assert z.testzip() is None, "ZIP CRC check failed"
                if temporary:
                    z.extractall(submission_dir)
                else:
                    for name in ("submission.py", "weights.pt"):
                        assert z.read(name) == (submission_dir / name).read_bytes(), "ZIP differs from checked directory"
            record("zip_root_and_crc", names)
        assert (submission_dir / "submission.py").is_file()
        assert (submission_dir / "weights.pt").is_file()
        report["submission_sha256"] = hashlib.sha256((submission_dir / "submission.py").read_bytes()).hexdigest()
        report["weights_sha256"] = hashlib.sha256((submission_dir / "weights.pt").read_bytes()).hexdigest()

        X, metadata, info = load_cache(split="train")
        frames = torch.from_numpy(np.array(X[:12], dtype=np.float32, copy=True))
        meta = {"device": "cpu", "submission_dir": submission_dir, "n_chans": 41, "n_times": 480,
                "n_classes": 3, "sfreq": 120.0, "ch_names": info["ch_names"],
                "chs_info": [{"ch_name": x} for x in info["ch_names"]]}
        # Force the optional local fallback even when the official kit exists.
        local = _import_submission(submission_dir / "submission.py", "_contract_submission_local", force_fallback=True)
        assert hasattr(local, "Solver") and issubclass(local.Solver, local.CompetSolver)
        local_model = local.Solver().load_model(meta)
        _prediction(local_model, frames[:1])
        record("local_fallback_load_model_predict")

        if kit_dir is None:
            kit_dir = Path(__file__).resolve().parent.parent / "reference_audit/2026-competition"
        kit_dir = Path(kit_dir).resolve()
        sys.path.insert(0, str(kit_dir))
        from benchmark_utils.base_solver import CompetSolver
        module = _import_submission(submission_dir / "submission.py", "_contract_submission_official")
        assert issubclass(module.Solver, CompetSolver)
        base = module.Solver().load_model(meta)
        record("official_CompetSolver_inheritance_load_model")
        baseline_state = _digest(base)

        for b in (1, 4, 0):
            model = _reset_copy(base)
            p = _prediction(model, frames[:b])
            record(f"predict_shape_batch_{b}", {"shape": list(p.shape), "dtype": str(p.dtype)})

        for name, signal in (("zero", torch.zeros_like(frames[:1])),
                             ("large_finite", torch.full_like(frames[:1], 1e6)),
                             ("negative_finite", torch.full_like(frames[:1], -1e6))):
            _prediction(_reset_copy(base), signal)
            record(f"finite_{name}_input")

        a = _reset_copy(base)
        b = _reset_copy(base)
        assert np.array_equal(_stream(a, frames[:6]), _stream(b, frames[:6]))
        record("deepcopy_before_prediction")
        # Both copies now hold an adapted six-window history.
        c = copy.deepcopy(a)
        assert np.array_equal(_stream(a, frames[6:]), _stream(c, frames[6:]))
        record("deepcopy_after_prediction_preserves_state")

        a.reset_state()
        initial = _reset_copy(base)
        assert np.array_equal(_stream(a, frames), _stream(initial, frames))
        record("reset_restores_fresh_predictions")

        fresh = _reset_copy(base)
        bulk = _prediction(fresh, frames)
        stepwise = _stream(_reset_copy(base), frames)
        assert np.array_equal(bulk, stepwise), "Batch inference sees a different history from one-at-a-time inference"
        record("batch_equals_one_window_stream")

        # Different future suffixes must leave the fixed prefix unchanged,
        # including the optional batch API's sequential behavior.
        altered = frames.clone()
        altered[6:] = torch.flip(altered[6:], dims=(-1,)) * 17.0
        left = _prediction(_reset_copy(base), frames)
        right = _prediction(_reset_copy(base), altered)
        prefix = _prediction(_reset_copy(base), frames[:6])
        assert np.array_equal(left[:6], right[:6])
        assert np.array_equal(left[:6], prefix)
        record("causal_prefix_invariance_to_future_suffix")

        a, b = _reset_copy(base), _reset_copy(base)
        state_b = _digest(b)
        _stream(a, frames)
        assert _digest(b) == state_b, "Predicting on one copy mutated another copy"
        assert _digest(base) == baseline_state, "Predicting on a copy mutated the loaded template model"
        assert np.array_equal(_stream(b, frames), stepwise)
        record("clone_isolation_and_unchanged_template")

        # Channels-by-time strides can differ between a torch loader and a
        # NumPy-made tensor. Values are identical, storage layout is not.
        noncontiguous = frames[:4].transpose(1, 2).contiguous().transpose(1, 2)
        assert not noncontiguous.is_contiguous()
        assert np.array_equal(_prediction(_reset_copy(base), noncontiguous), _prediction(_reset_copy(base), frames[:4]))
        record("noncontiguous_input_and_input_immutability")

        report["all_passed"] = True
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if temporary:
            temporary.cleanup()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True, type=Path, help="Directory with submission.py and weights.pt, or ZIP")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zip-file", type=Path)
    parser.add_argument("--kit-dir", type=Path)
    args = parser.parse_args()
    check_submission(args.submission_dir, args.output, args.zip_file, args.kit_dir)


if __name__ == "__main__":
    main()
