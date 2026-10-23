# Neural Interfaces 2026 — Track 02: BCI Decoding

Repo for the new **Track 02 warm-up (update 1)**: PROTEUS BCI Bordeaux, three
cued mental commands, causal streaming evaluation on Codabench
([competition 17982](https://www.codabench.org/competitions/17982/)).

> The previous Dreyer-2023 warm-up is now a **legacy phase** — its scores are
> not comparable with update 1. Legacy submissions were migrated to the new
> leaderboard. Update 1 allowance: **100 total submissions**, daily allowance
> resets.

## What changed (warm-up update 1)

| | Legacy warm-up | Warm-up update 1 |
|---|---|---|
| Dataset | Dreyer 2023 (public test participants) | PROTEUS BCI Bordeaux (public test participants) |
| Task | 2-class motor imagery | **3 commands**: motor imagery, mental subtraction, word generation |
| Input | Batched windows, 27-ch Dreyer montage | 41 EEG channels, 4 s @ 120 Hz → `(1, 41, 480)` |
| Delivery | Batched | **One window at a time, chronological within each participant × session stream, no shuffling** |
| State | — | Reset per session (`reset_state()` on a fresh copy); causal label-free adaptation within a session allowed |
| Ranking metric | Balanced accuracy pooled over windows | Balanced accuracy per **participant × session × interface** cell, macro-averaged across cells |

Warm-up scores are indicative; only the sealed phase determines the final
ranking. The sealed phase uses **private later sessions + entirely new
participants** from the same PROTEUS programme — no calibration data for
unseen participants.

## The submission contract (must-haves)

- `predict(X)` receives **only the EEG tensor** — no participant, session, run,
  interface, or label metadata.
- Input shape at evaluation: `(1, 41, 480)` — batch size 1, microvolts, already
  notch-filtered (50/60 Hz), band-passed (0.1–75 Hz), resampled to 120 Hz.
- Return **one integer class index per window**, shape `(B,)`:
  `0` = kinesthetic motor imagery, `1` = mental subtraction, `2` = word generation.
- Models hard-coded for Dreyer / 2 classes / batched inputs will fail —
  rebuild for the 41-channel, 3-class, streaming contract.

Full details: [docs/submission_contract.md](docs/submission_contract.md),
[docs/scientific_faq_bci.md](docs/scientific_faq_bci.md).

## Key resources

- [Tracks page](https://neural-interfaces26.github.io/tracks.html) · [Dataset on NEMAR (nm000290)](https://nemar.org/dataset/nm000290) · [Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md)
- [Participant guide — Track 02 baselines](https://neural-interfaces26.github.io/participant-guide.html) · [Competition kit (tracks/bci_decoding)](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding) · [NeuralBench Track 2 tutorial](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)

All links collected in [docs/resources.md](docs/resources.md).

## Registration

User's application for **Neural Interfaces 2026 — Track 2: BCI Decoding**
(Codabench username `dhonam`) was **accepted on 2026-10-15**
(see [docs/email_context.md](docs/email_context.md)).
