# Track 02 submission contract

Source: <https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding>
(captured 2026-10-23). See the Codabench **Track description** for the active
phase data and official ranking specification.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B,)`, one integer class index per window, with
  `meta["n_classes"]` possible classes.
- **Evaluation (model-facing):** causal and streamed. Windows reach `predict`
  **one at a time** (`B = 1`), each session of each participant in recording
  order. Every session starts from a `copy.deepcopy` of the model whose
  optional `reset_state()` is called first: a model can adapt within a
  session but carries nothing over to the next one.
- **Metric:** balanced accuracy within each subject × session × context
  (interface: Graz or BrainHero) cell, averaged over cells. Balanced accuracy
  averaged over sessions alone (NeuralBench's `bal_acc_stream_mean`), pooled
  balanced accuracy, and plain accuracy are reported alongside as diagnostics.
- Benchopt objective: `BCI-decoding`.

## Public data choices (Benchopt `-d` selector)

| Selector | Data |
|---|---|
| `BCI[study=dreyer2026proteus]` | PROTEUS (NEMAR nm000290), **default warm-up study** |
| `BCI[study=tangermann2012]` | BNCI2014_001, smaller real-data check |
| `Simulated` | tiny synthetic contract check, no download |

Real selectors run NeuralBench's streamed Track 02 task,
`eeg _motor_imagery_stream`. Each window is 4 s, notch 50/60 Hz + band-pass
0.1–75 Hz at recorded rate, resampled to 120 Hz, in microvolts with no
scaling/clamping. On PROTEUS a window holds 41 EEG channels and starts 1 s
after its cue.

## Worked examples

### NeuralBench start kit

```bash
pip install neuralbench 'nemar-py>=0.3.1'
neuralbench eeg _motor_imagery_stream --download
neuralbench eeg _motor_imagery_stream --prepare
neuralbench eeg _motor_imagery_stream -m eegnet --debug
```

**Note:** NeuralBench 0.4.0 is required:
`python -m pip install --upgrade "neuralbench==0.4.0"`.
PROTEUS is the default dataset of `_motor_imagery_stream` (no dataset flag).
After training EEGNet, use the shared NeuralBench-to-Codabench bridge.

### Benchopt competition kit

```bash
pip install benchopt
git clone https://github.com/neural-interfaces26/2026-competition
benchopt install 2026-competition/tracks/bci_decoding
benchopt prepare 2026-competition/tracks/bci_decoding --config 2026-competition/tracks/bci_decoding/training.yml
benchopt run 2026-competition/tracks/bci_decoding --config 2026-competition/tracks/bci_decoding/training.yml
```

`training.yml` selects PROTEUS and exports the `Torch-Linear` submission.
Zip the submission folder before uploading to Codabench.

Editable solvers in `solvers/`: `Constant` (uploadable floor), `MeanLogReg`
(sklearn linear, joblib weights), `Torch-Linear` (PyTorch, `fit` +
`save_model`), `EEGNet` (Braindecode EEGNet), `REVE` (frozen pretrained REVE
encoder + ridge classifier).

## Reference baselines (public PROTEUS split, NeuralBench 0.4.0, seeds 33–35)

| Baseline | Balanced accuracy |
|---|---|
| Constant prediction | 33.3% |
| EEGNet | 32.6% |
| CTNet | 35.0% |
| REVE fine-tuned | 36.0% |

This public split is purely cross-subject and is harder than the mixed
cross-session + cross-subject sealed protocol.

## Adaptation notes (from the NeuralBench Track 2 guide)

- Use `_motor_imagery_stream` for the competition: scores each session one
  window at a time, forward in time; `reset_state()` called on a fresh copy
  per session (`stream_by: [subject, session]`).
- Validation stays batched; set `data.val_batch_size: 1` to select
  checkpoints on streams too (several times the training cost).
- Local split: 60/20/20 subject-level, seeds 33 → 6 train / 2 validation /
  2 test participants on the 10 released.
- Model selection: keep the checkpoint with highest `val/bal_acc_stream_mean`.
  Max 40 epochs, early stop after 5 without improvement.
- REVE fine-tuned applies its own preprocessing (200 Hz, scaled) instead of
  the task's; its scaler is fit on each whole recording, which a streamed
  submission cannot do — so its test score is optimistic.

## Packaging & submission

1. Read the contract: Get Started → Submission Guide on the Codabench track.
2. Build the submission: wrap architecture, inference preprocessing, trained
   weights, and optional files in the required `Solver`.
3. Validate and ZIP: test inference locally with Benchopt, then zip the
   submission folder contents.
4. Upload and check: submit through My Submissions, confirm evaluation
   succeeds.

Validate the ZIP locally before uploading. Read the phase quotas, automatic
leaderboard selection, and sealed feedback rules. Warm-up scores are
indicative; only sealed scores determine the final ranking.
