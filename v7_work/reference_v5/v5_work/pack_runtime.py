"""One source-composition and weight-packing path for v5 replay and deployment.

There is no fitting or model selection in this module. The four neural
recipes use the unchanged v3 inference classes. The only added dispatch is
the frozen Riemannian tangent member. Source is composed in memory, allowing
CV verification before the decision whether a submission should exist.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_FILES = (
    ROOT / "methods_draft/riemann_features.py",
    ROOT / "bci_work/neural_cov.py",
    ROOT / "v5_work/riemann_v5_runtime.py",
    ROOT / "bci_work/deploy_runtime.py",
)
VIEWS = [{"crop": [0, 240]}, {"crop": [240, 480]}]
CANDIDATES = ("baseline_v3", "augmentation", "adversarial", "masked_ssl",
              "riemannian_tangent")


def _strip_embedded_imports(source: str) -> str:
    """Remove complete import statements, including multiline statements."""
    removed = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "__future__", "riemann_features",
        }:
            removed.update(range(node.lineno, node.end_lineno + 1))
    return "\n".join(line for number, line in enumerate(source.splitlines(), 1)
                     if number not in removed) + "\n"


def build_runtime_source() -> tuple[str, dict[str, str]]:
    """Return standalone source and hashes of its exact input source files.

    Imports in the resulting artifact are restricted to the standard library,
    NumPy, PyTorch and the official competition base class. No training helper
    or workspace package remains in the generated artifact.
    """
    header = '''"""Neural Interfaces 2026 Track 2: causal v5 EEG decoding.

Input is the official microvolt-valued (B, 41, 480) tensor at 120 Hz.
Each weight artifact records the participants used for that fitting stage.
All online adaptation belongs to a session-local, deepcopy-safe model and
reset_state restores its fitted priors. A prediction uses only the current
window and its own preceding session state. No labels, future EEG, network
calls or installation steps enter inference.
"""
import numpy as np
import torch
'''
    pieces, hashes = [header], {}
    for path in RUNTIME_FILES:
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        source = _strip_embedded_imports(raw.decode("utf-8"))
        if path.name == "deploy_runtime.py":
            original = "            elif kind=='neural_cov':\n"
            replacement = (
                "            elif kind=='riemannian_tangent':\n"
                "                member=V5TangentMember(record)\n" + original
            )
            if source.count(original) != 1:
                raise RuntimeError("v3 runtime dispatch changed: inspect source composition")
            source = source.replace(original, replacement)
        pieces.append("\n# ---- " + str(path.relative_to(ROOT)) + " ----\n" + source)
    source = "".join(pieces)
    tree = ast.parse(source, filename="submission.py")
    permitted = {"numpy", "torch", "pathlib", "benchmark_utils"}
    for node in ast.walk(tree):
        modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if any(module.split(".")[0] not in permitted for module in modules):
            raise RuntimeError("Unexpected standalone runtime import: " + str(modules))
    compile(tree, "submission.py", "exec")
    return source, hashes


def load_runtime_module(name: str = "v5_standalone_runtime_audit"):
    """Execute the same composed source that the final builder can write."""
    source, hashes = build_runtime_source()
    module = types.ModuleType(name)
    module.__file__ = "<" + name + ">"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, source, hashes


def pack_weights(neural_checkpoints, classical_checkpoint, ch_names, *, candidate,
                 training_subjects, extra_metadata=None):
    """Pack two already-verified neural branches and one classical branch.

    Provenance is deliberately retained. Callers must verify the stage-specific
    fitting manifests before supplying these objects; this function never
    loads, fits, chooses or changes any learned parameters.
    """
    if candidate not in CANDIDATES or len(neural_checkpoints) != 2:
        raise ValueError("Expected one fixed v5 candidate and two neural branches")
    expected_recipe = "baseline" if candidate in {"baseline_v3", "riemannian_tangent"} else candidate
    expected_classical = "riemannian_tangent" if candidate == "riemannian_tangent" else "linear"
    if classical_checkpoint["type"] != expected_classical:
        raise ValueError("Classical checkpoint differs from the candidate definition")
    subjects = sorted(set(map(str, training_subjects)))
    members = []
    for checkpoint in neural_checkpoints:
        if checkpoint["recipe"] != expected_recipe:
            raise ValueError("Neural recipe differs from the candidate definition")
        if sorted(checkpoint["training_subjects"]) != subjects:
            raise ValueError("Neural checkpoint fitting participants differ")
        members.append(dict(type="neural_cov", weight=.45, temperature=1.,
                            views=[dict(crop=list(view["crop"])) for view in VIEWS],
                            checkpoint=checkpoint))
    members.append(dict(classical_checkpoint, weight=.1))
    packed = dict(schema_version=1, ch_names=list(ch_names), members=members,
                  dataset="nm000290", version="1.0.0", candidate=candidate,
                  training_subjects=subjects, warmup_test_role_retired=True)
    extra = dict(extra_metadata or {})
    if set(extra).intersection(packed):
        raise ValueError("Extra metadata cannot overwrite runtime contract fields")
    packed.update(extra)
    return packed
