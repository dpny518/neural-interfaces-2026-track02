# V6 recovery record — 2026-10-24

This is the historical pre-training recovery snapshot. V7 subsequently
completed with an authorized reconstructed baseline and a null result; all
delivery checks passed on 2026-10-25. See the current
[developer handoff and recommendations](DEVELOPER_HANDOFF.md). The original v6
packages remain unrecovered.

## Current status

V7 training has not started. The original v6 artifacts have not been recovered.
The historical results below come from the user's saved message, not from
artifacts independently inspected in this workspace. Missing artifacts are
not a null experimental result. No new submission may be represented as the
original v6 model without verifying the original artifact.

## Verified surviving local artifacts

Files are under `bci/`. SHA-256 hashes were computed from the file bytes.

| File | Bytes | Identification | SHA-256 |
| --- | ---: | --- | --- |
| reproduce (2).zip | 2377463 | v3; scores match v4's historical v3 reference | a06af3ad781b3800dd341704c53f6b2a466c10e13fd1a5347b10991883a0e336 |
| reproduce (3).zip | 99078716 | v4; explicit internal README/report | 55d382f9518aabbcde32fd67b23b9c4bfbdb94f92eac952ce747b5909a937b0f |
| reproduce (4).zip | 65174999 | v5 Option A; explicit internal README/report | ebc27e96bbfd8b7ea72ae4fed67842efe37ee584e722cb935ca70ce72dd7d15f |
| report (4).txt | 15528 | v5; byte-identical to report in reproduce (4).zip | a90e4ab1890245b5d244b8a073c8501707ed53b599eab8839bd58162d78ef4c7 |

The v5 report gives 55.30% +/- 0.45 for its refitted v3 backbone, reports no
passing candidate, and marks the 17 submission checks not applicable.

The project and `~/Downloads` were searched recursively without access errors.
Downloads contains older v2/v3/v4 submission artifacts, but no deployment ZIP
with the expected v6 hash and no `legacy_submission.zip`. Relevant ZIP member
lists were also checked. No original artifact was modified.

## Historical v6 claims supplied by the user

These numbers must remain labeled historical/unverified until recovered or
independently recomputed; they are not measurements from a reconstruction.

| Candidate | CV mean +/- SD (%) | Gain over v3 (pp) | Paired SE (pp) |
| --- | ---: | ---: | ---: |
| Refitted v3 | 55.30 +/- 0.45 | — | — |
| Nonlinear log-variance head | 55.13 +/- 0.77 | -0.17 | 0.41 |
| Learned projection correlations | 53.19 +/- 0.42 | -2.11 | 0.74 |
| Multiscale log-variance | 57.91 +/- 0.58 | +2.60 | 0.84 |
| Nested classical ensemble | 55.83 +/- 0.31 | +0.52 | 0.35 |
| Four-network ensemble | 55.10 +/- 0.54 | -0.20 | 0.33 |
| Cumulative session adaptation | 55.34 +/- 0.40 | +0.04 | 0.22 |

Reported winning model:

- Two spatial log-variance networks weighted 0.45 each plus the original
  covariance ridge weighted 0.10.
- Early two-second, late two-second, and full four-second views, each using
  seven frequency bands: 21 banks total.
- Spatial filters shared across views and a 128-unit nonlinear residual head.
- Causal covariance adaptation: transform with preceding state, then update.
- Final fit on all 2,655 windows from P01–P10, seeds 33/42, epochs 75/20.
- Separate original-six fit on P01,P02,P03,P06,P07,P10, evaluated once on
  P04/P09: 755 windows, four sessions, eight cells; reported score 45.32%.
- Reported audit: 17 checks for each package, 55,755 batch-one CV predictions,
  294 deepcopies/resets, 21 CV scores, 471 fitting artifacts. These checks
  have not been independently verified here.

Expected deployment ZIP SHA-256:

```text
ce6c36d951c07d41db28179a541bac56865ddda717861a91d413bd198d80daa4
```

Expected files: submission.zip (~2.26 MB), legacy_submission.zip (~2.26 MB),
report.txt (~17.5 KB), verification.zip (~59.7 KB), reproduce.zip (~2.27 GB).

The saved message says Library delivery was rejected by automatic approval
review. That is a historical statement; no such rejection was observed in
this session. No tool for retrieving files from the prior ChatGPT workspace
or Library is exposed in the current tool set.

## Dell compute and recovery search

SSH key authentication succeeded for `dell@100.71.130.36`; the supplied
password was not needed and is not stored in this record.

Observed: Linux, 16 logical CPUs, RTX 3090 with 24,576 MiB VRAM, NVIDIA driver
570.133.20. At inspection, 7,747 MiB VRAM was allocated to existing processes.
No existing process was stopped or changed, and no training job was launched.

A filename/directory search traversed `/home/dell`, `/mnt`, and `/media`
(and `/data` if present), excluding common environment/cache/git directories.
697,822 directories were scanned. No relevant v6 artifact or project/data
directory was found. `/mnt/18t/docker_data` was inaccessible. A separate
search of home project/download/data directories completed without errors.
This does not prove the files are absent from unsearched locations, renamed
archives, backups, or the original ChatGPT workspace.

## Continuation requirements

Preferred: recover the actual v6 reproduction bundle and verify its internal
manifests, training recipe, architecture, nested classical definitions, and
predictions; verify the deployment ZIP against the full expected hash.

Alternative, pending the user's choice: build an explicitly labeled v6
reconstruction from the preserved v5 code and the saved architectural
description. Freeze every inferred implementation detail before measuring it.
Its measured baseline, checkpoint hashes, and results may differ from v6.
Do not tune the reconstruction repeatedly to match 57.91% or treat that
historical number as a target. The exact nested classical recipe is unknown
and requires recovery or an explicitly declared replacement.

For v7, preserve the requested ten-subject LOSO protocol with seed pairs
(33,42), (133,142), (233,242). Fit the baseline anew in each fold. All fitting
stages exclude the held-out participant, including SSL, references, scales,
BatchNorm, checkpoint decisions, and any inner-fold classical selection.

Use the official equally weighted subject/session/interface cell metric.
Average matched candidate-minus-baseline cell differences across the three
replicates before a 10,000-draw participant-cluster bootstrap; retain all cells
of each sampled participant and their official cell weighting. Adopt only
if mean gain exceeds paired SE. Freeze the bootstrap seed, exact candidate
recipes, winner/tie rule, final seeds, and artifact provenance before fitting.

P04/P09 remain development subjects under Option A. Any original-six diagnostic
is selection-contaminated, performed once after selection, and cannot affect
selection or fitting. A null v7 result means a completed panel with no passing
candidate, not missing inputs or an interrupted run. In that case ship no new
submission ZIP; report submission-only checks as not applicable while retaining
the completed CV isolation/replay evidence.
