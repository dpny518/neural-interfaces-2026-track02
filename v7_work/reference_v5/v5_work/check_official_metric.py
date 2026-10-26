"""Match all 15 completed CV scores to the official cell-mean implementation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v5_work.protocol_v5 import sha256_file, write_json_exclusive


def resolve_prediction_artifact(record, replay_report, *, legacy_path=None):
    """Prefer this extracted workspace and verify the replay-bound bytes."""
    path = Path(record['path'])
    anchor = Path(replay_report).resolve().parent
    choices = [anchor / path.name]
    if not path.is_absolute():
        choices.extend((anchor / path, ROOT / path))
    for prefix in ('v5_work', 'v5_deliverables'):
        if prefix in path.parts:
            choices.append(ROOT.joinpath(*path.parts[path.parts.index(prefix):]))
    if legacy_path is not None:
        choices.append(Path(legacy_path))
    choices.append(path)
    for candidate in choices:
        if candidate.is_file():
            if sha256_file(candidate) != record['sha256']:
                raise RuntimeError('Streamed prediction differs from its replay SHA-256: ' + str(candidate))
            return candidate.resolve()
    raise FileNotFoundError('Missing replay-bound streamed prediction: ' + str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', type=Path, default=ROOT / 'v5_work/cv_analysis/selection.json')
    ap.add_argument('--shards', type=Path, default=None,
                    help='Optional legacy shard-directory fallback; replay artifact hashes remain authoritative')
    ap.add_argument('--replay-report', type=Path,
                    default=ROOT / 'v5_work/cv_streaming_check/streaming_replay_check.json')
    ap.add_argument('--kit', type=Path, default=ROOT / 'reference_audit/2026-competition')
    ap.add_argument('--output', type=Path, default=ROOT / 'v5_work/cv_analysis/official_metric_check.json')
    args = ap.parse_args()
    analysis = json.loads(args.analysis.read_text())
    replay = json.loads(args.replay_report.read_text())
    assert replay['full_panel_certified'] is True
    assert replay['plan_sha256'] == analysis['plan_sha256']
    assert set(replay['prediction_artifacts']) == set(analysis['candidates'])
    sys.path.insert(0, str(args.kit.resolve()))
    from benchmark_utils.metrics import balanced_accuracy, group_scores
    checks = []
    for candidate, expected in analysis['candidates'].items():
        legacy_path = (args.shards / candidate / (candidate + '_streaming.npz')
                       if args.shards is not None else None)
        path = resolve_prediction_artifact(replay['prediction_artifacts'][candidate],
                                           args.replay_report, legacy_path=legacy_path)
        with np.load(path, allow_pickle=False) as z:
            truth, prediction = z['y_true'], z['y_pred']
            groups = np.stack([z['stream_id'], (z['interface'] == 'BrainHero').astype(np.int64)], axis=1)
            assert prediction.shape == (3, 2655)
            for replicate in range(3):
                cells = group_scores(balanced_accuracy, truth, prediction[replicate], groups)
                assert len(cells) == 28
                official = float(cells.mean())
                reference = expected['cv_scores_by_replicate'][replicate]
                assert abs(official - reference) < 1e-12
                checks.append(dict(candidate=candidate, replicate=replicate,
                                   seed_pair=analysis['seed_pairs'][replicate],
                                   official_streaming_cell_mean=official,
                                   reported_cell_mean=reference, n_cells=28,
                                   prediction_sha256=sha256_file(path), passed=True))
    assert len(checks) == 15
    result = dict(all_passed=True, n_checks=15, checks=checks,
                  source='benchmark_utils.metrics.group_scores(balanced_accuracy).mean()',
                  metric_source_sha256=sha256_file(args.kit / 'benchmark_utils/metrics.py'),
                  analysis_sha256=sha256_file(args.analysis),
                  replay_sha256=sha256_file(args.replay_report),
                  batch_size=1, n_windows_per_replicate=2655,
                  score_inputs='actual official-harness streamed labels, not pooled probabilities',
                  selection_rule_modified=False)
    write_json_exclusive(args.output, result)
    print(json.dumps(dict(all_passed=True, n_checks=15, output=str(args.output))))


if __name__ == '__main__':
    main()
