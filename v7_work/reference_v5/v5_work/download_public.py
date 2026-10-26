"""V5 option A: download all ten public development participants, checksum verified.
The historical warm-up test role is retired; sealed data are never accessed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def digest_bytes(data, algo):
    if algo == "git":
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return hashlib.new(algo, data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/nm000290/v1.0.0"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--partition", choices=["all-public-development"], default="all-public-development")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest_url = "https://data.nemar.org/nm000290/v1.0.0/manifest.json"
    data = urllib.request.urlopen(manifest_url, timeout=90).read()
    entries = json.loads(data)
    (args.destination / "download_manifest.json").write_bytes(data)
    allowed = {f"P{i:02d}" for i in range(1, 11)}
    def selected(entry):
        first = Path(entry["path"]).parts[0]
        return not first.startswith("sub-") or first.removeprefix("sub-") in allowed
    entries = [e for e in entries if selected(e)]
    started = time.monotonic()

    def fetch(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = args.destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            body = path.read_bytes()
            if len(body) == entry["size"] and digest_bytes(body, entry["checksum_algorithm"]) == entry["checksum"]:
                return entry["path"], len(body), "verified_existing"
        for attempt in range(4):
            try:
                request = urllib.request.Request(entry["url"], headers={"User-Agent": "NeuralInterfaces2026-reproducible-training/1.0"})
                body = urllib.request.urlopen(request, timeout=150).read()
                if len(body) != entry["size"]:
                    raise ValueError(f"Unexpected byte count for {path}: {len(body)} != {entry['size']}")
                checksum = digest_bytes(body, entry["checksum_algorithm"])
                if checksum != entry["checksum"]:
                    raise ValueError(f"Checksum mismatch for {path}: {checksum} != {entry['checksum']}")
                tmp = path.with_suffix(path.suffix + ".partial")
                tmp.write_bytes(body)
                os.replace(tmp, path)
                return entry["path"], len(body), "downloaded_verified"
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    records = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, entry): entry for entry in entries}
        for f in concurrent.futures.as_completed(futures):
            try:
                path, nbytes, status = f.result()
                records.append({"path": path, "size": nbytes, "status": status})
                if len(records) % 20 == 0 or path.endswith(".edf"):
                    print(json.dumps({"completed": len(records), "total": len(entries), "MB": round(sum(r['size'] for r in records) / 1e6, 1), "seconds": round(time.monotonic() - started, 1), "latest": path}), flush=True)
            except Exception as exc:
                entry = futures[f]
                errors.append({"path": entry["path"], "error": repr(exc)})
                print(json.dumps(errors[-1]), flush=True)
    report = {"dataset": "nm000290", "tag": "v1.0.0", "manifest_url": manifest_url,
              "partition": args.partition, "downloaded_subjects": sorted(allowed),
              "files_verified": len(records), "bytes_verified": sum(r["size"] for r in records),
              "elapsed_seconds": time.monotonic() - started, "errors": errors,
              "files": sorted(records, key=lambda r: r["path"])}
    (args.destination / ("download_verification_" + args.partition + ".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "files"}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
