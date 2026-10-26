# Developer handoff — 26 October 2026

## Outcome and scope

V7 is complete. None of four predeclared candidates passed the adoption rule.
The baseline was reconstructed from surviving v5 source and the user's saved
architectural description; it is **not the recovered original v6**. The
historical 57.91% result and deployment hash `ce6c36d9…` remain unverified.

| Model | CV mean ± sample SD | Paired gain | Participant-bootstrap SE | Decision |
| --- | ---: | ---: | ---: | --- |
| Reconstructed v6 | 57.6982% ± 0.7423% | — | — | Baseline |
| Masked self-supervised pretraining | 57.6729% ± 1.2592% | −0.0253 pp | 0.1934 pp | Reject |
| Replacement nested classical ensemble | 57.4077% ± 0.6611% | −0.2905 pp | 0.1315 pp | Reject |
| Expanded band/view grid | 57.9641% ± 0.1150% | +0.2659 pp | 0.4880 pp | Reject |
| Cross-subject covariance mixup | 58.0473% ± 0.3010% | +0.3491 pp | 0.5063 pp | Reject |

The retained recipe has 7 frequency bands × early/late/full views (21 banks),
shared spatial filters, a 128-unit residual head, and an ensemble of two neural
members at 45% each plus covariance ridge at 10%. Exact inferred choices are
recorded in [plan.json](../v7_work/plan.json), including the explicit replacement
for the lost v6 nested-classical recipe.

All P01–P10 are adaptively reused public development participants under Option A.
P04/P09 have no independent test role. CV is not a sealed-phase performance claim.

## What is available

- `v7_work/`: current code, frozen fitting implementation, pinned evaluator,
  preserved v5 sources, synthetic checks, and acquisition/transfer helpers.
- `results/v7/`: immutable measured-run report, predictions, analysis,
  per-fit metadata and histories, nested selection evidence, provenance,
  verification records, and the executed top-level source snapshot.
- `v7_deliverables/`: small delivery receipts are tracked. The full local ZIPs
  are intentionally ignored by Git and are obtainable from Bay.
- `docs/v6_recovery.md`: historical recovery investigation, clearly dated.
- `bci/report (4).txt`: surviving v5 report. The original downloaded ZIPs remain
  unchanged locally and are excluded from Git.

No all-ten reconstructed-baseline deployment model exists yet. No v7 candidate
deployment or legacy refit was performed, and no new P04/P09 diagnostic was run.
An older v3 package exists on the maintainer's Mac at `~/Downloads/submission.zip`:
998,788 bytes, SHA-256
`8287c844ac1bcc159b9f1b654e970ec58fc2b37d1e51fcbfdfce432fcb95cdcd`.
It matches the submission embedded in the surviving v3 reproduction archive;
it is not the reconstructed multiscale model.

## Review without a GPU or raw EEG

Use Python 3.12 or 3.13 in a virtual environment, then run from the repository root:

```sh
python -m pip install -r requirements-review.txt
python v7_work/analyze.py --plan v7_work/plan.json --predictions results/v7/predictions --out reanalysis-review
```

The output directory must be new. Compare `reanalysis-review/selection.json`
with `results/v7/analysis/selection.json`; the cell and bootstrap arrays should
also match exactly with the recorded NumPy version. The local delivery review
already performed these comparisons, including all bootstrap samples.

For inference, training and synthetic checks, use the packages recorded in
`results/v7/provenance/environment_*.json` and `v7_work/README.md`. The actual
training host used Python 3.12.13, NumPy 2.2.6, PyTorch 2.13.0+cu126 and an RTX3090.
Preprocessing used MNE 1.10.2. Full fitting needs the public EEG or regenerated
arrays and the original pinned preprocessing source.

The fitting freeze is enforced by `protocol.freeze_implementation()`. Do not
silently change frozen fitting files or reuse their cached fits after an edit.
`results/v7/executed_source/` is a reference snapshot, not a standalone run
directory. The complete reproduction ZIP supplies the full original layout.

## Review priorities

1. Read `plan.json`, then `protocol.py`, `run_cv.py`, `training.py` and
   `classical.py`. Check exact fitting rows for every stage: outer fit excludes
   one participant; nested fit excludes both outer and inner participants.
   SSL, covariance references, scaling and BatchNorm use fitting data only.
2. Review `runtime.py` and `verify.py` together. Prediction transforms each
   window using prior session state before updating it. The official harness
   deepcopies and resets at each session boundary. The observation sink only
   records outputs and does not feed metadata into adaptation.
3. Review `analyze.py` and `test_statistics.py`. The metric equally weights
   subject/session/interface cells, not participants or windows. Average
   matched seed-repetition differences before the 10,000 participant-cluster
   bootstrap; resamples retain all sessions/interfaces and use sampled cell
   sums divided by sampled cell counts.
4. Review `finalize.py`'s null guard and deployment provenance. All 17 package
   contract checks are conditional on a package being built. They were **not
   applicable** to this null round; do not report them as passed.

Completed evidence: 460 fitting artifacts audited, 39,825 actual batch-one
predictions, 210 deepcopies and resets, zero class-label mismatches, and all
15 complete CV scores matched the pinned official metric. Isolated reanalysis
blocked access to the original workspace and reproduced the decision exactly.
Delivery verification checked every ZIP CRC and all 2,046 embedded manifest
entries. These are executable checks and code review, not a formal leakage proof.

## Recommended next task: deployment recovery

The user requested recommendations; this recovery build has not been executed.

1. Create a **separate baseline-recovery plan and output directory**, explicitly
   anchored to the frozen `reconstructed_v6` recipe and the completed v7 evidence.
   Keep the recorded v7 decision `selected: null` unchanged. Do not force a
   winner into `finalize.postselection()` to bypass its null guard. Implement
   a dedicated recovery driver with an explicit recovery-provenance binding,
   reusing the fitting and contract-check primitives where appropriate.
2. Refit on all 2,655 windows from P01–P10 using seeds **33/42**, fixed **75/20
   epochs**, and the frozen 45%/45%/10% mixture. No new parameter search,
   held-out checkpoint selection, or attempt to reproduce the historical
   57.91% number. The resulting package has its own new hash and identity.
3. Build a separate legacy refit using exactly **P01,P02,P03,P06,P07,P10**.
   Audit the actual fitting populations and freeze both ZIPs. Run all **17
   package checks** on each, including stage-specific fitting provenance,
   required-only metadata, output shape/dtype, reset/deepcopy, causal prefix,
   batch/sequential equivalence, clone isolation and input immutability.
4. After both packages are frozen, perform the legacy model's single P04/P09
   diagnostic: 755 windows, four sessions, eight cells. Record an exclusive
   evaluation receipt. It cannot influence selection, fitting or calibration,
   and must be labeled selection-contaminated public-development evidence.
5. Deliver `submission.zip`, `legacy_submission.zip`, `report.txt`,
   `verification.zip` and `reproduce.zip`, clearly labeled reconstructed-v6
   deployment recovery. Verify hashes after the Bay-to-Mac transfer. The
   57.6982% figure describes the preceding LOSO recipe evaluation, not an
   observed score of the all-ten deployment fit.

Use the measured Dell inputs or regenerate and verify them. A separate Mac
data recovery completed during the run but was **never activated on Dell**;
its ignored local `v7_work/cache/` is not automatically the measured cache.

## Further research after a usable package exists

Review the per-participant/cell errors and candidate disagreement first. Then
declare at most two targeted v8 candidates and a compute budget before fitting.
Retain the ten-subject LOSO protocol, seed pairs (33,42), (133,142), (233,242),
complete fitting isolation, participant-cluster bootstrap and gain > SE rule.
The v7 mixup and grid gains were insufficient; do not relabel them as adopted
improvements or tune repeatedly against the same reported scores without
recording that adaptive reuse. P04/P09 remain development participants.

## Artifact access and operational notes

The private Bay shelf is configured through the sibling `cloudflare-storage`
project. Stored folder:

```text
transfers/neural-interfaces-2026-track02/v7-20261025
```

Use your own authorized Bay login/config. No credentials or public share tokens
are in this repository. With that project's CLI:

```sh
cli/bay ls transfers/neural-interfaces-2026-track02/v7-20261025
cli/bay download transfers/neural-interfaces-2026-track02/v7-20261025/reproduce.zip -o /tmp/ni26-v7-reproduce.zip
```

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `report.txt` | 10,243 | `086bf05c1f7d3cf96349bfded7e13cae69328c218aeed7d6db6634996e5e62f4` |
| `verification.zip` | 28,758 | `43a44207f239dc264ef71e07f40d4dccb3ee361f2e86682758925a0bfb14bd92` |
| `reproduce.zip` | 1,269,782,187 | `7a101ef47340cbb4d2b9a331329cd35b61ca1fb8476c5aef8aece8632cb980ff` |

The measured run is also on Dell under
`/mnt/18t/neural-interfaces-2026-track02-v7/v7_work`.

During transfer, Bay emitted a malformed `Content-Range` header containing
`NaN`. The successful workaround verified every requested 4 MiB segment against
hashes computed on Dell, then verified the complete archive hash. The small
segment manifest and transfer receipt are committed. No Bay deployment was
changed. `bay_transfer.py` documents that workaround but contains machine-specific
paths; adapt its configuration before reuse. Earlier transfer/activation helpers
and `SESSION_HANDOFF.md` are operational history, not the recommended entrypoints.

On the external Mac volume, AppleDouble `._` files matched the EDF glob in the
preserved preprocessing script. `prepare_local.py` stages only manifest-listed
files on a native temporary filesystem without editing the frozen source.
Use the same precaution for fresh preprocessing on such volumes.
