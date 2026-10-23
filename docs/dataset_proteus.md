# PROTEUS BCI Bordeaux — Track 02 dataset

Public release for the 2026 warm-up. Source:
<https://nemar.org/dataset/nm000290> (NEMAR `nm000290`, v1.0.0, CC-BY-4.0,
published 2026-10-01).

## What it is

- 10 participants, 14 sessions, 112 runs (~1.48 GB)
- 41 EEG channels at 500 Hz (EEG-BIDS); task runs use the Graz and
  BrainHero interfaces with real/sham feedback
- 3 cued mental commands: **kinesthetic motor imagery**, **mental
  subtraction**, **word generation**
- Also includes resting baselines (eyes open / eyes closed)
- Full programme: 20 participants × 6 sessions; the public release has two
  participants with their first three sessions and eight with their first
  session. A distinct private counterpart is reserved for sealed evaluation
  (later sessions + new participants; composition undisclosed)

Recommended citation: Pauline, D., Manon, B., Loic, B., Simon, K., Sebastien,
V., Sebastien, R., Raphaëlle, R., & Fabien, L. (2026). *PROTEUS BCI Bordeaux
— EEG/EMG Foundation Challenge 2026, Track 02* (Version v1.0.0) [Data set].
NEMAR. <https://doi.org/10.82901/nemar.nm000290>

## Download

```bash
nemar dataset download nm000290                                   # clone + fetch in one step
nemar dataset download nm000290 --subjects sub-01,02             # subset
# or in two steps:
nemar dataset clone nm000290 && cd nm000290 && nemar dataset get <files>
```

Direct HTTPS for a single file also works, e.g. participants.tsv from
data.nemar.org/nm000290/v1.0.0/.

## Zarr copy (streaming access)

1. Start at `https://zarr.nemar.org/nm000290/zarr/index.json` (mandatory entry
   point; never hardcode a bucket path).
2. Pick a store entry (`stores[].zarr`, `stores[].groups[].name`).
3. Build the store URI: `s3://nemar/nm000290/zarr/{store.zarr}` (us-east-2,
   anonymous; use published `contract_base`/`s3_uri` when present).
4. Open read-only: `zarr.open_group(store=..., mode="r", zarr_format=3)`
   (zarr_format=3 required — anonymous ListBucket is denied, so v2 probes
   403).
5. Read level-0 array: `root[store.groups[0].name]["0"]` (full-rate signal;
   never use `view/` arrays for inference).
6. Dequantize: `physical = digital * scale + offset` (per-channel scale/offset
   attributes; unit on the group's `channels` attribute).
7. Slice, don't download: `signal[0:4, 0:500]`.

## Track 02 data context (from the tracks page)

| Study | Role | Participants | Channels | Rate | Size |
|---|---|---|---|---|---|
| Stieger 2021 | Alternative public data · historical reference | 62 | 60 | 1000 Hz | 399 GB |
| Dreyer 2023 | **Legacy warm-up** · alternative public data | 87 | 27 | 512 Hz | 19 GB |
| Zyma 2019 | — | 36 | 21 | 500 Hz | 0.18 GB |
| Scherer 2015 | — | 9 | 30 | 256 Hz | 1.1 GB |
| **PROTEUS BCI Bordeaux** | **Public training · warm-up evaluation** | 10 | 41 | 500 Hz | 1.48 GB |
| PROTEUS evaluation set | Sealed evaluation (private) | undisclosed | undisclosed | undisclosed | — |
