# COMPLETE — 2026-10-25 11:10:35 UTC

The fixed v7 reconstruction round and all required null-result delivery work
are complete. Do not restart fitting or repeat transfers. Three complete
deliverables are local in v7_deliverables/ and privately uploaded to Bay:
transfers/neural-interfaces-2026-track02/v7-20261025.

The final archive was downloaded in303 independently hash-verified4MiB parts,
assembled, and matched SHA2567a101ef47340cbb4d2b9a331329cd35b61ca1fb8476c5aef8aece8632cb980ff.
All outer artifact hashes, both ZIP CRC checks, all2046 embedded manifest
hashes,460archivedCVartifacts, absence of submission/postselection artifacts,
and exact local independent reanalysis passed. final_delivery_check.py exited0.
Receipts: v7_deliverables/local_review.json, bay_transfer_receipt.json,
original_files_unchanged.json. v7_work/status.json now records complete.

Outcome: no candidate's pairedgain exceeded its bootstrapSE. No new deployment
or legacy submissionZIPs were built; no legacydiagnostic was run. Package-only
17contractchecks are explicitlyN/A in the report. The original v6 artifacts
remain unrecovered; this used the authorized reconstructed_v6 baseline.
Return the null-result table and local report/verification/reproduce links.

# Previous delivery notes (historical; all delivery completed above)

User explicitly authorized using shared cloudflare-storage/Bay for upload from
Dell and download on Mac. All three deliverables plus SHA256SUMS.json are now
uploaded privately to transfers/neural-interfaces-2026-track02/v7-20261025 at
the existing Bay endpoint. No public share was created. Dell upload completed.
The existing Mac login was passed over encrypted SSH stdin and only kept in
the upload child's environment; no credential file was written on Dell.

Bay returns malformed Content-Range headers (bytes NaN-1269782186/1269782187),
although requested segment bytes and Content-Length are correct. The ordinary
full-file connection also dropped. A robust workaround is NOW RUNNING:
`python3 -u v7_work/bay_transfer.py verified-parts`, unifiedexec64632.
It checks each 4MiB segment against SHA256 computed on the original Dell ZIP,
using v7_deliverables/reproduce_chunk_hashes.json (303segments), and stores only
verified chunks under v7_bay_verified_parts/. Last observed19/303 in14seconds.
Log: v7_work/bay_verified_download.log. Each complete chunk is resumable.
On completion it assembles reproduce.zip, verifies its full expected SHA256,
then invokes final_delivery_check.py (all outerhashes, ZIPCRCs, every inner
manifesthash,460CVartifacts, no submissionZIPs, exactlocalanalysis match).
It writes bay_transfer_receipt.json, updates local_review.json and status.json
to complete. Check these and the running process before doing anything else.
Do not rerun model fitting or upload archives again.

All four bci originals were rehashed and match their initial checksums;
receipt: v7_deliverables/original_files_unchanged.json.
Old rsync97594/97603 and parallel clients1100/2383 were stopped intentionally.
Temporary Dell tailnet transfer server2545971 was stopped. Their incomplete
local parts are not deliverables. Keep originals and complete archives intact.

# Prior delivery state — superseded by latest state above

The Dell original workflow completed overnight with a NULL result. It acquired
its own dataset, preprocessed, fit all 460 artifacts, replayed all 39,825
predictions, passed isolated reanalysis, and packaged three deliverables. DO NOT
run activate_prepared_data.py or restart the experiment. Mac alternative data
recovery was never activated on Dell and was not used for the measured run.

Measured reconstructed baseline 57.6982% +/-0.7423; SSL57.6729, nested57.4077,
expanded57.9641, mixup58.0473. Respective paired gains/SE in pp:
-0.0253/0.1934, -0.2905/0.1315, +0.2659/0.4880, +0.3491/0.5063.
No passes; no final/legacy candidate fitting or new submission ZIPs. The
historical v6 deployment remains missing. Do not claim recovered v6 identity.

Remote completed path: /mnt/18t/neural-interfaces-2026-track02-v7/v7_work/deliverables.
Local destination: v7_deliverables/.
- report.txt10243bytes SHA086bf05c1f7d3cf96349bfded7e13cae69328c218aeed7d6db6634996e5e62f4
- verification.zip28758bytes SHA43a44207f239dc264ef71e07f40d4dccb3ee361f2e86682758925a0bfb14bd92
- reproduce.zip1269782187bytes SHA7a101ef47340cbb4d2b9a331329cd35b61ca1fb8476c5aef8aece8632cb980ff

Report/verification are downloaded, hash/CRC verified. A 1.2MB analysis_evidence.zip
SHA5210b3f54dd9763989b6537f5aa0aa108479d7e037c34b064de0599ad6678b49 was also
downloaded; local independent reanalysis in v7_review/ exactly reproduced all
selection JSON, cell arrays, bootstrap arrays and null decision. Receipt is
v7_deliverables/local_review.json. Do not redo successful checks needlessly.

Full reproduce.zip transfer is INCOMPLETE. Active rsync unifiedexec14944,
parentPID97594 child97603, writes v7_deliverables/reproduce.zip (partial; last90MB).
SSH master socket /tmp/ni26-ssh-control-v3, Tailscale connection slow and unstable.
To improve transfer, a temporary read-only single-file HTTP Range server is on
Dell tailnet-only100.71.130.36:18767, PID2545971, project/transfer_archive_server.py.
It serves only /<reproduce_sha256>, auto shuts down after3600sec; stop it when done.
Local fetch_completed_archive.py downloads16 independent8MiB chunks into
v7_transfer_chunks/ and assembles reproduce.verified.zip only after fullhashmatch.
It was changed to ProxyHandler({}) because localhost HTTP proxy timed out for
the private address; this direct Tailscale connection was explicitly approved.
Old clientPID1100 stopped; newer process session will appear in live tools.
Ignore AppleDouble ._ files when counting chunks. Retain resumable real chunks.
When either transfer finishes, stop only the other task-owned process, publish
the hash-verified full ZIP as reproduce.zip, check CRC and embeddedSHA256SUMS,
update local_review.json/status, and return concise null-result table + links.
Do not call the task complete until the full artifact is downloaded or explicitly
state a remaining delivery failure. Preserve all original bci files.

# Earlier execution notes (historical, superseded by current state above)

The user explicitly authorized reconstruction after the original v6 artifacts
could not be recovered, and has repeatedly said to continue. Complete the
reconstruction + fixed v7 round and return actual deliverables. Do not ask for
that authorization again. No sub-agents are authorized by the active developer
instructions. No goal was created.

## Frozen experiment

- Plan SHA-256: `0448fed4aed8eb3b84e399204d1ec87b805f6e7824d3d3b02a41d176cb74950f`.
- Fitting implementation freeze SHA-256:
  `16c85f05472c629b8093721d6fd0d9d451bb0065ab3bac1fe904b110a252a139`.
- The first freeze is preserved as
  `implementation_freeze_before_download_resume.json`. The sole pre-fitting
  revision added resumable HTTP downloads after read timeouts. No EEG fitting
  had started. The model plan was not changed.
- Baseline is explicitly `reconstructed_v6`, not the lost verified v6 model.
  Never assert the historical 57.91%/ce6c36d9… artifact was reproduced.
- Four fixed candidates: masked SSL; replacement nested classical panel;
  expanded band/view grid; cross-subject covariance mixup. See plan.json.
- Actual old local files remain unchanged under bci/. The extracted v5 source
  lives in reference_v5/ with original byte hashes.

## Dell job

- Host: `dell@100.71.130.36` (user supplied authentication in conversation;
  do not put the password in files).
- Remote project: `/mnt/18t/neural-interfaces-2026-track02-v7`.
- Code/results: `<project>/v7_work`.
- Interpreter: `<project>/.venv/bin/python` (system packages from py312,
  GPU PyTorch 2.13.0+cu126; MNE1.10.2 and benchopt1.10.0 installed only in venv).
- GPU RTX3090 24GiB, about7.7GiB already allocated to unrelated services;
  leave those processes alone. 16 CPUs,64GiB RAM,8.6TB free on /mnt/18t.
- Full workflow was positively launched as PID1566794, current downloader
  PID1568236. Check actual process/status before trusting these stale IDs.
- `workflow.py` runs pinned benchopt setup; resumed data preparation;
  orchestrate.py; test_statistics.py; verify.py; finalize.py. It automatically
  runs GPU/CPU CV and conditional packaging once data is ready.
- Root logs: `workflow.log`, `data.log` (contains an old failed attempt before
  the current DOWNLOAD_START line). Work logs: `v7_work/neural.log`,
  `v7_work/classical.log`; authoritative current stage `v7_work/status.json`.
- Most recent observation: waiting_for_verified_data,160/587 files verified,
  ~372MB reused; four EDF partials are growing. Totalmanifest=1,593,504,588bytes.
  Downloads from the Dell are slow (~tens of KB/s per request), with retries.
  NO measured CV results yet.
- Old downloader1256906 failed on HTTP read timeout after150files; it is gone.
  New downloader keeps partials, HTTPRange resumes, four workers,12attempts.
- Latest transferred source archive `source-v3.tar.gz`309970bytes SHA-256
  `5800ad7f195643d58b37287a9a5f62e02a0f778d51919481702420c36fff3bb0` was
  verified and unpacked before launch. workflow.py was separately written to
  the remote with the same content as local (including skip-ready-data resume).

## Live terminal

The currently useful unified-exec session is **21507**, an authenticated SSH
TTY running the Dell py312 Python REPL. Its Python variables include `root`
(Path to remote project), `Path`, `os`, `subprocess`, `hashlib`, etc.

Use `tools.write_stdin` to this session with short Python statements ending in
newline. For multiline code use `exec(<JSON-encoded Python source string>)`.
TTY echoes are sometimes returned in fragments; poll before interpreting
partial output. This avoids very unreliable repeated SSH handshakes.
The remote background workflow remains independent of the control session.

Example: `print((root/'v7_work/status.json').read_text())`.
Example: `print((root/'data.log').read_text()[-1600:])`.
Never stop unrelated host processes. The job is not a null result if it fails.

## Verification already performed

- Local synthetic state/causality checks passed: batch/sequential probabilities,
  reset, deepcopy, causalprefix, clone isolation, empty/integer outputs,
  immutability, held-out input rejection, one-epoch fit, fit-only SSL.
- Synthetic bootstrap test passed: unequal cells (P04/P10 sixcells) retain
  equal-cell estimand, exact independent bootstrap formula, rejectzero gain,
  expected adoption, reject incomplete panels.
- Download mocktest passed: resumeafterinterruption,serverignoringRange reset,
  verified-existing-file reuse. Testfile was added locally afterv3transfer;
  no fitting source changes.
- Standalone packed inference import, safe weights_only checkpoint load and
  exact probability roundtrip matched unpackaged model on synthetic inputs.
- All currently written Python compiled. Core freeze validation passed.

## Required remaining work

1. Monitor/resume data acquisition as needed, then data/metadata verification.
2. Confirmactual GPU CV and classical CV start and finish. Resolve real bugs
   with logged, scoped corrections; preserve frozen scientific plan. Do not
   silently edit fitted sources or retain results from changed recipes.
3. Full expected CV audit:460checkpoints/reference artifacts. Replay39825
   actual one-window calls,210copies/resets,15official CVscore comparisons.
4. Pairedbootstrap andconditionalwinner; final/legacyfits onlyifpassing;
   all17contractchecks andsingle P04/P09diagnostic onlyafterbothZIPsfrozen.
5. Independent prediction-onlyreanalysis with originalworkspace opensdenied.
6. Finish report/verification/reproduce; inspect generated artifacts and copy
   completed deliverables back to this Mac (suggest v7_deliverables/) with
   SHA-256 verification. Null => no newsubmission ZIPs. Do not claimcomplete
   because the workflow was merely launched.

## Optional network investigation in progress

Agent Reach + Firecrawl skills were read for official alternate endpoint lookup.
Firecrawl has zero credits and search failed402; built-inweb fallback used.
Official NEMAR docs and nemar-py README describe public S3 content-addressed
objects. GitHub plugin verified the v1.0.0 EDF pointer:
`/annex/objects/SHA256E-s19142172--55290252415a2157f59feeb9c28383f7e8bb71cd5a76facd2814cce4da4e0d6f.edf`.
DirectpublicS3 probe onDell returned206,262144bytes in15.23s; not an obvious
speedup. Do not change data identity; every file must pass original manifest.

Mac curl speedprobe is pending unifiedexec **50393**. It reads first256KiB
from the same S3object to /tmp/ni26-nemar-speed-probe.bin and prints speed/time.
If Mac is much faster, consider preparing exact raw/windows locally and
transferring toDell, with matchingchecksums andactual preprocessing provenance.
No such alternate download has been launched yet.

Usefulprimarysources:
https://docs.nemar.org/platform/data-api/
https://docs.nemar.org/platform/for-agents/
https://github.com/nemarOrg/nemar-py
https://github.com/nemarDatasets/nm000290
