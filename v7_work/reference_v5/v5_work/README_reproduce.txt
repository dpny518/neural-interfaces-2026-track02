NEURAL INTERFACES 2026 TRACK 2 — V5 OPTION A
Reproducibility instructions

Read report.txt for the measured result and deployment decision. The frozen
experiment is v5_work/plan.json. Its SHA-256 is:
ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59

1. WHAT IS IN THIS BUNDLE

The bundle preserves training and evaluation source, the exact candidate plan,
all ten fold definitions and row memberships, all three ensemble seed pairs,
the 240 neural CV checkpoints, the 20 classical CV checkpoints, the ten fitted
covariance references, provenance manifests, predictions, logs, and analysis.
The report records whether a candidate passed the predefined adoption rule.
If it did not, there is intentionally no final fitted v5 model or submission.

Raw EDF recordings, processed EEG windows, and large aligned covariance caches
are omitted because they can be regenerated from the public versioned data.
Metadata, channel order, preprocessing settings, public download manifest and
checksums, and the checksums of omitted intermediate arrays are retained.
No sealed-phase data were accessed.

2. SUBJECT ROLES AND SEEDS

Development subjects: sub-P01, sub-P02, sub-P03, sub-P04, sub-P05, sub-P06,
sub-P07, sub-P08, sub-P09, sub-P10.
Each LOSO fold fits exactly the other nine participants. The held-out subject
contributes no EEG, labels, SSL examples, normalization statistics, covariance
references, feature scales, BatchNorm statistics, or checkpoint decisions.

The three paired ensemble replicates use seeds (33,42), (133,142), (233,242).
Their two neural branches train for fixed 75 and 20 epochs respectively, with
the same 120-epoch cosine schedule definition. No validation early stopping.
The deterministic classical branch is shared across optimization replicates.

Historical six-subject training set: P01,P02,P03,P06,P07,P10.
Historical validation: P05,P08. Historical warm-up test: P04,P09.
These historical labels describe v3/v4 provenance; all ten subjects are now
development subjects. There is no independent public warm-up test remaining.
See v5_work/splits.json for every explicit fold list and exact window rows.

Only a passing, immutable CV selection permits a post-selection fit. The final
deployment seeds were fixed at (33,42) before CV. A final deployment model uses
all ten public subjects/all sessions. A separate original-six fit may provide
one post-selection, selection-contaminated legacy diagnostic on P04/P09.
That diagnostic is neither an independent test nor the score of the all-ten
deployment model. It cannot change selection or the frozen submission.

3. ENVIRONMENT

Observed environment: Python 3.12.14, CPU PyTorch 2.14.1+cpu, NumPy 2.3.5,
SciPy 1.17.0, scikit-learn 1.8.0, MNE 1.10.2, mne-bids 0.17.0.
Exact installed versions: v5_work/requirements-training.lock.txt.
There was no GPU. Each neural recipe used two CPU threads; classical fitting
used one. The four neural recipes ran independently in parallel.

The classical CV process loaded the original source retained as
v5_work/source_snapshots/classical_v5_at_cv_start.py. Its SHA-256 matches the
classical entry in implementation_record.json. The exact two-line change to
the current classical_v5.py is recorded in
v5_work/classical_implementation_changes.json: successful final/legacy fits
are marked complete, while finite-OOF assertions remain CV-only. CV control
flow and fitting numerics are unchanged. The initial snapshot hash and the
current source hash are deliberately different and both are preserved.

A training environment needs the dependencies in the lock file. CPU PyTorch
can be installed from https://download.pytorch.org/whl/cpu before installing
the remaining lock-file packages. These installations are for reproduction
only. A submitted inference artifact does not install anything.

The official competition source was pinned to:
https://github.com/neural-interfaces26/2026-competition
commit a66bc23d1a0e686b895c093a642b45c74970758b
Place that checkout at reference_audit/2026-competition for runtime checks.
The preprocessing reference is NeuroAI v0.4.0, commit
7aaccf34b11540e00cca2425c6b9094cae396bf6.

The scripts resolve project paths relative to their own location. Run commands
below from the extracted bundle root. Use your chosen Python interpreter in
place of python. Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS to
the documented per-process thread count when reproducing timings.

4. RECOMPUTE THE REPORTED DECISION WITHOUT RETRAINING

The archived candidate NPZ files contain probabilities shaped (3,2655,3),
labels, subject/session/interface keys, exact row indices, and seed pairs.
The analysis requires the complete, hash-verified assembly manifest. It does
not require omitted EEG or covariance arrays and does not refit a model.

python v5_work/analyze_cv.py --plan v5_work/plan.json --predictions-dir v5_work/cv_predictions --out v5_work/reanalysis

The output directory must be new. The selection record is written exclusively
and is never overwritten. Apart from timestamps and output paths, its scores
and decision should match the archived v5_work/cv_analysis/selection.json.

Metric: balanced accuracy within every subject x session x interface cell,
then the unweighted mean of all 28 cells. Three complete LOSO scores are
reported as mean +/- sample SD. The paired gain uses matched folds/seeds.
For its approximate SE, paired cell differences are averaged over the three
replicates, then ten participant clusters are resampled 10,000 times with
seed 20261023, keeping each participant's 2 or 6 cells together. The official
cell weighting is preserved in each resample. This is not a window bootstrap
or a calculation treating 30 fold/seed values as independent participants.

Adopt only when mean paired gain > paired-gain subject-cluster SE. Among
passing candidates, maximize gain minus SE; break exact ties lexicographically.
This is the agreed development adoption rule, not a 95% significance claim.
The participant sample is small, CV fitting sets overlap, and selecting among
four ablations leaves selection optimism. The public results are development
estimates after explicit retirement of the old test role.

5. REGENERATE PUBLIC DATA AND WINDOW-LOCAL FEATURES

The data are PROTEUS BCI Bordeaux, NEMAR nm000290 v1.0.0, CC-BY-4.0:
https://doi.org/10.82901/nemar.nm000290
https://data.nemar.org/nm000290/v1.0.0/manifest.json

python v5_work/download_public.py --destination data/nm000290/v1.0.0 --workers 8
python v5_work/prepare_public.py --data-dir data/nm000290/v1.0.0 --output-dir v5_work/cache --workers 4
python bci_work/extract_features.py --cache v5_work/cache

Expected totals: 112 EDF runs, 84 cued runs, 14 subject-session streams,
28 scoring cells, 2,655 command windows of shape (41,480). The exact signal
transform is unchanged from v3: notch 50/60 Hz and their sub-Nyquist harmonics,
bandpass 0.1-75 Hz at 500 Hz, resample to 120 Hz, convert to float32 and scale
to microvolts by 1e6; four-second windows start one second after the cue.
No additional offline scaling. Model-internal covariance operations are
identical in fitting and causal inference. Source references use fitting
subjects only. The preprocessing function's AST equality to v3 is recorded.

6. REPEAT ALL FITS IN A FRESH RUN DIRECTORY

Preserve the evidence in this bundle. To refit from scratch, copy source,
plan.json, plan.sha256, and recipe-definition JSON files to a fresh project
directory with the same relative layout, then regenerate data there. Do not
copy the archived result folders into that fresh run. The fit scripts refuse
to overwrite registered artifacts. Do not delete or edit the frozen plan to
make a different experiment look like this one.

python v5_work/prepare_fold_cache.py --plan v5_work/plan.json --threads 2
python v5_work/run_neural_cv.py --recipe baseline --threads 2
python v5_work/run_neural_cv.py --recipe augmentation --threads 2
python v5_work/run_neural_cv.py --recipe adversarial --threads 2
python v5_work/run_neural_cv.py --recipe masked_ssl --threads 2
python v5_work/classical_v5.py --plan v5_work/plan.json --stage cv --threads 1
python v5_work/finalize_classical_integrity.py --plan v5_work/plan.json --expected-plan-sha256 ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59
python v5_work/assemble_cv_predictions.py
python v5_work/verify_cv_streaming.py --predictions-dir v5_work/cv_predictions --threads 1

The final command above is a complete serial replay alternative. The recorded
run may shard replay by candidate and combine coverage using
v5_work/aggregate_streaming_checks.py; sharding changes no models or predictions.
Run replay after every fit has finished. It uses raw windows, batch-one
prediction, and deepcopy/reset at each session. Only after complete assembly
and successful replay may the complete panel be scored:

python v5_work/analyze_cv.py --plan v5_work/plan.json --predictions-dir v5_work/cv_predictions --out v5_work/cv_analysis
python v5_work/check_official_metric.py

The metric check reads each candidate's exact saved prediction artifact and
SHA-256 from the full replay report, so it accepts either serial replay or the
combined five-shard report. It verifies all fifteen complete CV scores using
the pinned official group_scores(balanced_accuracy) implementation. No
partial-fold scores are selection inputs.

The neural recipe commands can run concurrently if resources permit. Their
checkpoint seeds and numerical recipes are independent of launch order.
Floating-point/library/hardware differences can affect exact training outcomes;
the archived checkpoints and predictions preserve the actual measured runs.

7. CONDITIONAL DEPLOYMENT AND CONTRACT

No post-selection fitting is permitted for a null result. When selection
passes, freeze_selection in protocol_v5.py binds the selection and plan hashes.
train_postselection.py then fits the selected neural recipe on all ten public
subjects using the predetermined seed pair, with no evaluation. classical_v5.py
fits only the selected ridge or tangent branch. A separate legacy stage fits
the original six subjects, without using evaluation labels during fitting.
verify_postselection_inputs in train_postselection.py binds the plan and the
current metadata/raw-covariance hashes to the assembly used by the passing
selection before any new fit. The same guard precedes classical post-selection
fitting, without modifying its already-running CV fitting code.

The original 16 packaging/inference/causality checks are reused. Check 17 is
stage-specific provenance: exact held-out exclusion in every fitting field
of all 270 CV artifacts, followed by correct final/legacy fitting populations
and matching embedded checkpoint metadata. It is deliberately not a global
P04/P09 blacklist. All-ten final fitting legitimately includes both subjects.
For a null result there is no new submission to run these 17 checks on: they
are reported as not applicable. The separate completed 270-artifact CV fold
isolation audit can pass without implying that submission checks were run.

Once the complete official-metric evidence exists, and any required passing
selection, final/legacy fits, 17-check reports and one-time diagnostic are
finished, generate the report with:

python v5_work/write_report_v5.py --output v5_deliverables/report.txt

The generator refuses missing evidence and refuses to overwrite its output.
It writes report_evidence.json beside report.txt, binding the report to its
checked input hashes. To recreate the archived report without retraining or
rescoring, retain the archived selection and checks and choose a new output:

python v5_work/write_report_v5.py --output v5_work/recreated_report/report.txt

Fresh training runs need their own actual environment/source and split records
in the same schemas; the archived implementation_record.json describes the
recorded run, not the environment of a later rerun.

Runtime requires only NumPy/PyTorch, standard-library modules and the official
benchmark_utils.base_solver import with local fallback. If a new submission
is authorized, its ZIP contains only submission.py and weights.pt at its root.
Training-only domain and reconstruction heads are discarded before deployment.

8. INTEGRITY AND LIMITS

SHA256SUMS.txt at the bundle root covers its archived files. Per-artifact
provenance ties each checkpoint to the frozen recipe, actual fitting row hash,
subject population, source reference, seed and fixed epoch. These are
executable bookkeeping assertions plus source inspection and runtime replay;
they are not a claim of an operating-system sandbox or formal leakage proof.

The Riemannian ablation uses the predeclared maximum 12 optimization steps for
its training-only geometric mean. Per-band convergence flags and residuals
are retained. A finite-iteration reference is not silently relabeled as an
exactly converged mean, and no extra iteration search is performed after seeing
scores. No foundation model or architecture sweep beyond the frozen four
ablations is claimed. A null result applies to this bounded experiment.
