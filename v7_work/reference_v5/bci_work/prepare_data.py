"""Prepare PROTEUS nm000290 v1.0.0 using the Track 2 stream transform.

This reproduces the signal-transform implementation in neuroai v0.4.0,
commit 7aaccf34b11540e00cca2425c6b9094cae396bf6, without installing the full
NeuralBench stack. Inputs are EDF + EEG-BIDS sidecars; outputs are an mmap
window array and metadata. No fitted preprocessing is applied here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import train_test_split


CODES = {"mi": 0, "sub": 1, "word": 2}
INTERFACES = {
    "AcquisitionBH": "BrainHero",
    "OnlineRawBH": "BrainHero",
    "AcquisitionGraz": "Graz",
    "OnlineRawGraz": "Graz",
}
EXPECTED_SPLITS = {
    "train": ["P01", "P02", "P03", "P06", "P07", "P10"],
    "val": ["P05", "P08"],
    "test": ["P04", "P09"],
}


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def entities(path):
    return dict(item.split("-", 1) for item in Path(path).stem.split("_") if "-" in item)


def sidecar(path, suffix):
    return Path(str(path).replace("_eeg.edf", suffix))


def subject_split(subjects):
    # neuralset.events.transforms.splitting.SklearnSplit: test is split
    # first; val uses 0.2/(1-0.2) of the remaining, shuffled subject order.
    subjects = sorted(subjects)
    train_val, test = train_test_split(subjects, test_size=0.2, random_state=33)
    train, val = train_test_split(train_val, test_size=0.25, random_state=33)
    return {s: split for split, ids in (("train", train), ("val", val), ("test", test)) for s in ids}


def _fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_run(args):
    path, run_cache, canonical_channels = args
    path, run_cache = Path(path), Path(run_cache)
    out = run_cache / (path.stem + ".npz")
    cues = [r for r in read_tsv(sidecar(path, "_events.tsv")) if r["trial_type"] in CODES]
    if not cues:
        return None
    ent = entities(path)
    assert ent["task"] in INTERFACES, path
    if out.exists():
        return str(out)
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    channel_rows = read_tsv(sidecar(path, "_channels.tsv"))
    eeg_channels = [r["name"] for r in channel_rows if r["type"].upper() == "EEG"]
    assert len(eeg_channels) == 41, (path, len(eeg_channels))
    assert set(eeg_channels) == set(canonical_channels), path
    # read_raw_bids uses these types and bad flags. They are all EEG here;
    # bad channels are retained, as the official extractor drop_bads=False.
    raw.set_channel_types({r["name"]: r["type"].lower() for r in channel_rows}, verbose="ERROR")
    raw.info["bads"] = [r["name"] for r in channel_rows if r.get("status") == "bad"]
    raw.pick(eeg_channels)
    assert float(raw.info["sfreq"]) == 500.0, raw.info["sfreq"]
    # NeuralSet _notch_filter expands the configured base frequencies to
    # their harmonics below Nyquist, up to a 301 Hz ceiling.
    notch = []
    for freq in [50.0, 60.0]:
        notch.extend(np.arange(freq, min(raw.info["sfreq"] / 2, 301), freq).tolist())
    raw.notch_filter(notch, picks="all", phase="zero", filter_length="auto", n_jobs=1, verbose="ERROR")
    raw.filter(0.1, 75.0, picks="all", skip_by_annotation=("edge", "bad_acq_skip"), n_jobs=1, verbose="ERROR")
    raw.resample(120.0, n_jobs=1, verbose="ERROR")
    # MneTimedArray.from_native casts the resampled signal to float32;
    # scale_factor=1e6 is applied later to extracted windows.
    signal = raw.get_data().astype(np.float32)
    order = np.array([raw.ch_names.index(name) for name in canonical_channels])
    starts = np.array([int(round((float(r["onset"]) + 1.0) * 120.0)) for r in cues], dtype=np.int64)
    assert all(0 <= start and start + 480 <= signal.shape[1] for start in starts), path
    windows = np.stack([signal[order, start : start + 480] * np.float32(1e6) for start in starts])
    assert windows.shape == (len(cues), 41, 480)
    assert np.isfinite(windows).all(), path
    np.savez(out,
             X=windows,
             y=np.array([CODES[r["trial_type"]] for r in cues], dtype=np.int64),
             onset=np.array([float(r["onset"]) + 1.0 for r in cues]),
             cue_onset=np.array([float(r["onset"]) for r in cues]),
             start_sample=starts,
             source_sha256=np.array(_fingerprint(path)),
             source_file=np.array(str(path)),
             subject=np.array(ent["sub"]), session=np.array(ent["ses"]),
             task=np.array(ent["task"]), run=np.array(int(ent["run"])),
             interface=np.array(INTERFACES[ent["task"]]))
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--data-dir", type=Path, default=root / "data/nm000290/v1.0.0")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--partition", choices=["development", "test"], default="development")
    ap.add_argument("--freeze-manifest", type=Path, default=None)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_cache = args.output_dir / "runs"
    run_cache.mkdir(exist_ok=True)
    files = sorted(args.data_dir.glob("sub-*/ses-*/eeg/*_eeg.edf"))
    if args.partition == "test":
        assert args.freeze_manifest is not None and args.freeze_manifest.is_file(), "Test preprocessing requires frozen selection manifest"
    allowed = set(EXPECTED_SPLITS["train"] + EXPECTED_SPLITS["val"]) if args.partition == "development" else set(EXPECTED_SPLITS["test"])
    files = [p for p in files if entities(p)["sub"] in allowed]
    expected_runs = 80 if args.partition == "development" else 32
    assert len(files) == expected_runs, f"Expected {expected_runs} source runs, found {len(files)}"
    # Study.iter_timelines preserves lexicographic task order for ties at
    # run 0, and otherwise sorts by acquisition run number.
    files.sort(key=lambda p: (entities(p)["sub"], entities(p)["ses"], int(entities(p)["run"])))
    canonical_channels = [r["name"] for r in read_tsv(sidecar(files[0], "_channels.tsv")) if r["type"].upper() == "EEG"]
    split_of = subject_split({s for ids in EXPECTED_SPLITS.values() for s in ids})
    assert {split: sorted(s for s, v in split_of.items() if v == split) for split in EXPECTED_SPLITS} == EXPECTED_SPLITS
    params = [(str(p), str(run_cache), canonical_channels) for p in files]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(prepare_run, params), 1):
            if result:
                results.append(result)
            print(f"Prepared {i}/{len(files)} runs ({len(results)} cued)", flush=True)
    n_windows = 0
    for path in results:
        with np.load(path) as z:
            n_windows += len(z["y"])
    expected_windows = 1900 if args.partition == "development" else 755
    assert n_windows == expected_windows, f"Expected {expected_windows} windows, got {n_windows}"
    windows = np.lib.format.open_memmap(args.output_dir / "windows.npy", mode="w+", dtype=np.float32, shape=(n_windows, 41, 480))
    rows = []
    source_hashes = {}
    stream_ids = {}
    offset = 0
    for record_id, path in enumerate(results):
        with np.load(path) as z:
            n = len(z["y"])
            windows[offset:offset+n] = z["X"]
            ent = {k: z[k].item() for k in ("subject", "session", "task", "run", "interface")}
            stream_key = (ent["subject"], ent["session"])
            stream_id = stream_ids.setdefault(stream_key, len(stream_ids))
            source = Path(str(z["source_file"].item())).relative_to(args.data_dir)
            source_hashes[str(source)] = str(z["source_sha256"].item())
            for j in range(n):
                rows.append({**ent, "y": int(z["y"][j]), "onset": float(z["onset"][j]),
                             "cue_onset": float(z["cue_onset"][j]), "start_sample": int(z["start_sample"][j]),
                             "stream_id": stream_id, "record_id": record_id,
                             "split": split_of[ent["subject"]], "source_file": str(source)})
            offset += n
    windows.flush()
    meta = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(args.output_dir / "metadata.npz", **meta)
    split_counts = {s: int(np.sum(meta["split"] == s)) for s in EXPECTED_SPLITS}
    expected_counts = {"train": 1521, "val": 379, "test": 0} if args.partition == "development" else {"train": 0, "val": 0, "test": 755}
    assert split_counts == expected_counts, split_counts
    info = {
        "dataset": "NEMAR nm000290", "tag": "1.0.0", "license": "CC-BY-4.0",
        "dataset_doi": "https://doi.org/10.82901/nemar.nm000290",
        "competition_commit": "a66bc23d1a0e686b895c093a642b45c74970758b",
        "neuroai_release": "v0.4.0", "neuroai_commit": "7aaccf34b11540e00cca2425c6b9094cae396bf6",
        "mne_version": mne.__version__, "numpy_version": np.__version__,
        "input_shape": [41, 480], "sfreq": 120.0, "ch_names": canonical_channels,
        "units": "microvolts", "scale_factor": 1e6, "source_sfreq": 500.0,
        "notch_frequencies_hz": [50, 100, 150, 200, 60, 120, 180, 240],
        "bandpass_hz": [0.1, 75], "resample_method": "MNE default FFT",
        "window_start_seconds_after_cue": 1.0, "window_duration_seconds": 4.0,
        "window_start_index_rule": "round((cue_onset + 1) * 120)",
        "channel_order": "first-seen EEG channel order (NeuralSet unique map)",
        "label_mapping": CODES, "subjects_by_split": EXPECTED_SPLITS,
        "window_counts_by_split": split_counts, "n_windows": n_windows,
        "partition": args.partition, "subjects_present": sorted(allowed),
        "n_source_runs": len(files), "n_cued_runs": len(results), "n_streams": len(stream_ids),
        "source_sha256": source_hashes,
    }
    (args.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("n_windows", "n_source_runs", "n_cued_runs", "n_streams", "window_counts_by_split", "subjects_by_split")}))


if __name__ == "__main__":
    main()
