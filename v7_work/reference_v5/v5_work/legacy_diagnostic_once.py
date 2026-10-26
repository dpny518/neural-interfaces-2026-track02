"""One descriptive warm-up diagnostic after the passing v5 package is frozen.

This evaluates the separately fitted original-six model. P04/P09 are already
development subjects for v5 selection, so the number is selection-contaminated
and is not a held-out estimate for the all-ten deployment model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'bci_work')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from evaluation import cell_mean_score
from v5_work.protocol_v5 import (
    LEGACY_REPORT_SUBJECTS, verify_selection_freeze, assert_stage_subjects,
    claim_legacy_diagnostic_once, sha256_file, write_json_exclusive,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection-freeze', type=Path, required=True)
    ap.add_argument('--final-dir', type=Path, default=ROOT / 'v5_deliverables')
    ap.add_argument('--legacy-dir', type=Path, default=ROOT / 'v5_work/legacy_diagnostic_model')
    ap.add_argument('--cache', type=Path, default=ROOT / 'v5_work/cache')
    ap.add_argument('--kit', type=Path, default=ROOT / 'reference_audit/2026-competition')
    ap.add_argument('--out', type=Path, default=ROOT / 'v5_work/legacy_diagnostic')
    args = ap.parse_args()
    torch.set_num_threads(1)
    threadpool_limits(1)
    frozen = verify_selection_freeze(args.selection_freeze)
    final_zip = args.final_dir / 'submission.zip'
    legacy_zip = args.legacy_dir / 'legacy_submission.zip'
    final_hash, legacy_hash = sha256_file(final_zip), sha256_file(legacy_zip)
    for directory, archive, stage in ((args.final_dir, final_zip, 'final_fit'),
                                      (args.legacy_dir, legacy_zip, 'legacy_fit')):
        contract = json.loads((directory / 'contract_check.json').read_text())
        assert contract['all_passed'] is True and contract['n_checks'] == 17
        assert contract['stage'] == stage
        assert contract['checks'][-1]['name'] == 'stage_specific_no_heldout_leakage'
        assert contract['checks'][-1]['selection_freeze_sha256'] == sha256_file(args.selection_freeze)
        with zipfile.ZipFile(archive) as handle:
            assert handle.namelist() == ['submission.py', 'weights.pt']
            for name, key in (('submission.py', 'submission_sha256'), ('weights.pt', 'weights_sha256')):
                assert handle.read(name) == (directory / 'submission' / name).read_bytes()
                assert sha256_file(directory / 'submission' / name) == contract[key]
    # This exclusive receipt is created before any diagnostic inference.
    args.out.mkdir(parents=True, exist_ok=True)
    claim_legacy_diagnostic_once(selection_freeze=args.selection_freeze,
                                 model_path=legacy_zip, receipt_path=args.out / 'once_receipt.json')
    write_json_exclusive(args.out / 'frozen_packages.json', dict(
        final_submission_sha256=final_hash, legacy_submission_sha256=legacy_hash,
        selection_freeze_sha256=sha256_file(args.selection_freeze),
        selected_candidate=frozen['selected_candidate'], diagnostic_passes_authorized=1,
    ))
    with np.load(args.cache / 'metadata.npz', allow_pickle=False) as archive:
        all_meta = {key: archive[key] for key in archive.files}
    rows = np.flatnonzero(np.isin(all_meta['subject'], LEGACY_REPORT_SUBJECTS))
    meta = {key: value[rows] for key, value in all_meta.items()}
    assert_stage_subjects(meta['subject'], stage='legacy_evaluate',
                          selection_freeze=args.selection_freeze)
    assert len(rows) == 755
    windows = np.load(args.cache / 'windows.npy', mmap_mode='r')
    info = json.loads((args.cache / 'dataset_info.json').read_text())
    sys.path.insert(0, str(args.kit.resolve()))
    from benchmark_utils.streaming import predict_streams
    from benchmark_utils.metrics import balanced_accuracy, group_scores
    source = args.legacy_dir / 'submission/submission.py'
    spec = importlib.util.spec_from_file_location('v5_frozen_legacy_diagnostic', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model_meta = dict(device='cpu', submission_dir=source.parent, n_chans=41,
                      n_times=480, n_classes=3, sfreq=120., ch_names=info['ch_names'])
    model = module.Solver().load_model(model_meta)

    def loader():
        for local, row in enumerate(rows):
            yield (torch.from_numpy(np.array(windows[row:row+1], copy=True)),
                   torch.tensor(meta['y'][local:local+1]),
                   dict(stream_id=meta['stream_id'][local:local+1],
                        record_id=meta['record_id'][local:local+1],
                        onset=meta['onset'][local:local+1],
                        context_id=(meta['interface'][local:local+1] == 'BrainHero').astype(np.int64)))

    started = time.perf_counter()
    true, predicted, stream_info = predict_streams(model, loader())
    elapsed = time.perf_counter() - started
    predicted = predicted.astype(np.int64)
    np.testing.assert_array_equal(true, meta['y'])
    metric = cell_mean_score(true, predicted, meta)
    official = float(group_scores(balanced_accuracy, true, predicted,
                     np.stack([stream_info['stream_id'], stream_info['context_id']], axis=1)).mean())
    assert abs(official - metric['balanced_accuracy_cell_mean']) < 1e-12
    assert metric['n_cells'] == 8 and metric['n_sessions'] == 4
    np.savez_compressed(args.out / 'predictions.npz', y_true=true, y_pred=predicted,
                        row_indices=rows, **{key: meta[key] for key in
                            ('subject', 'session', 'interface', 'stream_id', 'record_id', 'onset')})
    assert sha256_file(final_zip) == final_hash and sha256_file(legacy_zip) == legacy_hash
    result = dict(
        schema='track2-v5-selection-contaminated-legacy-diagnostic-v1',
        interpretation='Post-selection original-six fit on P04/P09; selection-contaminated development diagnostic',
        score_of_all_ten_final_model=False, independent_test_estimate=False,
        used_for_selection=False, warmup_test_role_retired=True, passes=1,
        selected_candidate=frozen['selected_candidate'], metric=metric,
        official_metric_exact_match=True, evaluator='benchmark_utils.streaming.predict_streams',
        batch_size=1, deepcopy_and_reset_per_session=True, device='cpu',
        inference_seconds=elapsed, mean_milliseconds_per_window=elapsed * 1000 / len(rows),
        final_submission_sha256=final_hash, legacy_submission_sha256=legacy_hash,
        prediction_sha256=sha256_file(args.out / 'predictions.npz'),
    )
    write_json_exclusive(args.out / 'result.json', result)
    print(json.dumps(dict(legacy_diagnostic_cell_mean=official, n_windows=755, n_cells=8,
                          selection_contaminated=True, used_for_selection=False), indent=2))


if __name__ == '__main__':
    main()
