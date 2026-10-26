# V7 with a reconstructed v6 baseline

The original v6 artifacts were not recovered. The user authorized reconstruction
on 2026-10-24. The historical 57.91% result and ce6c36d9… hash are unverified
historical claims, not targets for tuning this reconstruction. The exact
inferences and replacement nested classical recipe are frozen in `plan.json`.

## Status and locations

The measured run is complete with a null result. Start with the current
[developer handoff](../docs/DEVELOPER_HANDOFF.md) and the committed
[evidence snapshot](../results/v7/README.md). `status.json` records completion
and delivery verification. A prepared plan or incomplete run is not a null result.
The Dell's generated packages are under its `v7_work/deliverables/`; the verified
Mac copies are in the repository's `v7_deliverables/`, with large ZIPs excluded
from Git and retained in the private Bay shelf.

Dell workspace: `/mnt/18t/neural-interfaces-2026-track02-v7/v7_work`.
Environment: `/mnt/18t/neural-interfaces-2026-track02-v7/.venv`.
The original Mac `bci/` artifacts are unchanged. `reference_v5/` contains
byte-preserved extracted source, metadata and provenance, with its own manifest.

## Reproduction

Use a fresh run directory for a new fit. Preserve recorded results and do not
silently overwrite a frozen plan or implementation. Python needs PyTorch,
NumPy, SciPy, scikit-learn, MNE 1.10.2, threadpoolctl, and benchopt for official
base-class contract checks. The measured versions are recorded when jobs run.
Inference packages require only NumPy, PyTorch and the optional official base
class, with a local fallback. No inference installation occurs.

Run from this directory's parent, replacing `python` with the training interpreter:

```sh
python v7_work/prepare_data.py
python v7_work/selfcheck.py
python v7_work/run_cv.py --part features
python v7_work/run_cv.py --part neural --device cuda:0
python v7_work/run_cv.py --part classical
python v7_work/verify.py
python v7_work/finalize.py
```

The neural and classical commands may run concurrently. They use disjoint
fitting artifacts and no partial outer-fold scores for selection. The baseline
neural checkpoints are intentionally reused by the nested-classical candidate,
whose only change is its classical mixture. Each baseline fold is freshly fit.

`prepare_data.py` downloads the versioned public data using the preserved
manifest and validates every supplied checksum. It runs the unchanged v5
preprocessing source and compares metadata with the preserved 2,655-row archive.
All window-local feature computation may be cached globally; references,
normalization statistics and learned parameters are fitted only within the
authorized fold populations. Fold caches are hash bound to their inputs.

An alternative recovery completed on the Mac using `mirror_data.py` and
`prepare_local.py`, but its transfer stalled and was never activated on the
Dell. The measured run instead completed using the Dell's original downloader
and preprocessing pipeline. Local `data_transfer.json` describes the unused
Mac recovery, not the inputs used for the reported GPU fits. The complete
measured-run source and provenance are in `v7_deliverables/reproduce.zip`.

`verify.py` audits all 460 expected CV checkpoints/reference artifacts and all
inner selections, then executes the actual pinned official `predict_streams`
harness for every window, candidate and replicate. Predictions are observed
from the same model `predict` call, without applying adaptation a second time.
Only the model's EEG input participates in inference. The observer receives
probabilities but never supplies information to inference.

`finalize.py` requires complete verified replay before analysis. It refits and
packages only a passing winner, using seeds 33/42 and fixed 75/20 epochs. Both
all-ten and original-six ZIPs are frozen before the one historical P04/P09
diagnostic. All-ten fitting legitimately includes P04/P09. A null result creates
report/reproduction/verification evidence only, with submission-only contract
checks marked not applicable.

For prediction-only reanalysis, use a new output directory:

```sh
python v7_work/analyze.py --plan v7_work/plan.json --predictions results/v7/predictions --out reanalysis-review
```

The 10,000-draw bootstrap samples participants, retaining every session and
interface cell. Matched replicate differences are averaged before resampling;
the ratio of sampled cell sums to cell counts preserves the official equal-cell
metric. The result is not an equal-subject metric or a window bootstrap.

## Evidence and limitations

`implementation_freeze.json` binds the fitting code before EEG fitting.
Every fitted checkpoint embeds its actual fitting rows, excluded populations,
seed/epoch policy, metadata hash, plan hash and source-freeze hash. Matching
sidecars bind artifact bytes. Inner fits exclude both outer and inner subjects.

`verification/` holds complete replay and provenance audits, conditional
17-check package reports, and isolated reanalysis evidence. The latter copies
only the analysis inputs into a temporary directory and denies file opens into
the original workspace; every result and bootstrap array must match exactly.

The reproduction archive keeps all fitted candidates, including rejected ones.
Raw EEG and large regenerable covariance/alignment arrays are omitted; their
manifests, hashes and regeneration scripts are retained. Different numerical
libraries/hardware can change a fresh fit. Saved checkpoints preserve the
measured run. Audit assertions and code review are not a formal leakage proof
or an operating-system training sandbox.

All ten public participants are adaptively reused development data. Neither CV
nor the single legacy diagnostic constitutes a new independent test, and no
sealed-phase performance claim is made.
