# Track 02 warm-up update announcement (verbatim)

> Message received by the user (text as provided). Source: The Neural
> Interfaces 2026 Organizing Team.

Dear Track 02 participants,

We are pleased to announce that the new Track 02 warm-up is now live on the official PROTEUS BCI Bordeaux dataset.
The website was updated: https://neural-interfaces26.github.io/tracks.html. The released PROTEUS data is available here: https://nemar.org/dataset/nm000290.

Please use Warm-up Phase (update 1) for all new submissions: https://www.codabench.org/competitions/17982/
The previous warm-up phase is now retained as a legacy phase. Because the dataset and scoring protocol have changed, its scores are not comparable with those from Update 1.

What has changed? Please visit the scientific FAQ (https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md). A summary is provided below:

The previous warm-up used Dreyer 2023, two-class motor imagery, batched windows and pooled balanced accuracy.
The new warm-up uses PROTEUS BCI Bordeaux with 41 EEG channels and three commands: motor imagery, mental subtraction and word generation.
Codabench provides four-second windows resampled to 120 Hz, with an input shape of (1, 41, 480).
Windows are evaluated one at a time, chronologically within each session, without shuffling.
Model state is reset when a new session begins.
predict() receives only EEG data—no participant, session, run, interface or label metadata.
Causal, label-free adaptation may be used within the current session.
The official ranking metric is balanced accuracy computed for each participant × session × interface cell and then averaged equally across cells. Higher is better.
Pooled balanced accuracy and plain accuracy remain available as diagnostic metrics.

What does this mean for your submission? Models developed for the legacy warm-up must be adapted for:
- 41 PROTEUS EEG channels;
- three output classes instead of two;
- four-second inputs at 120 Hz;
- batch size 1 during evaluation;
- chronological streaming;
- session-level state resets.

Legacy models hard-coded for Dreyer, two classes or batched inputs may therefore fail and need to be updated before resubmission.
We re-evaluated the submissions displayed on the legacy leaderboard and put them in the new leaderboard. The migration may have consumed today's daily submission allowance for some participants. The daily allowance resets tomorrow, and the total allowance for Update 1 has been reset to 100 submissions for everyone.

We have also added a comprehensive Scientific FAQ with detailed explanations of the dataset, preprocessing, chronological evaluation, adaptation, state handling and scoring.

Warm-up scores remain indicative; only the sealed phase determines the final competition ranking.

Questions and answers are handled through our Discord server. If you are not already a member, please join it using the invitation provided in your competition registration communications.

Thank you for your patience during this transition. We are excited to see your models evaluated on the new PROTEUS protocol!

Best regards,
The Neural Interfaces 2026 Organizing Team

## Resources

- Competition: https://www.codabench.org/competitions/17982/
- Dataset: https://nemar.org/dataset/nm000290
- Scientific FAQ: https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md
- Participant guide and baselines: https://neural-interfaces26.github.io/participant-guide.html#baseline-track-2
- Public competition kit: https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding
- NeuralBench tutorial: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html
- Updated competition website: https://neural-interfaces26.github.io/tracks.html
