# Recorded v7 result

This is the immutable lightweight evidence snapshot from the completed run.
No candidate passed; `analysis/selection.json` records `selected: null`.
The baseline is explicitly reconstructed v6, not the lost original package.

See [the full report](report.txt) and
[the developer handoff](../../docs/DEVELOPER_HANDOFF.md).

| Path | Purpose |
| --- | --- |
| `analysis/` | Decision, per-cell arrays and 10,000-draw paired bootstrap arrays |
| `predictions/` | All five models' three-repetition out-of-fold probabilities and replay manifest |
| `runs/` | Per-fit provenance, histories, predictions and nested selections; binary weights omitted |
| `verification/` | Fitting-isolation audit, official replay and isolated reanalysis proof |
| `provenance/` | Frozen protocol/source hashes, environment, dataset metadata and completion records |
| `executed_source/` | Exact top-level Python source used in the archived measured run |
| `logs/` | Completed neural/classical worker logs |
| `snapshot_manifest.json` | Each extracted member's original archive path and SHA-256 |
| `local_review.json` | Successful independent local reanalysis and complete artifact verification |
| `SHA256SUMS.json` | Full deliverable sizes and hashes |

All snapshot members were extracted from the verified reproduction ZIP and
checked against its embedded manifest. Checkpoints and raw EEG are not in Git.
The full reproduction ZIP contains all 460 CV checkpoints/reference artifacts
and instructions/checksums for regenerating the omitted raw data and caches.

The current `v7_work/` fitting files remain bound by the same implementation
freeze. Some non-fitting delivery/documentation helpers were added after the
run. Use this executed-source snapshot or the full archive when checking the
precise recorded execution, rather than inferring it from later delivery code.

`provenance/status.json` is the coordinator's intermediate status captured when
the reproduction bundle was assembled. The final outcome is in
`analysis/selection.json`; completed delivery and integrity checks are recorded
in `local_review.json` and the current `v7_work/status.json`.
