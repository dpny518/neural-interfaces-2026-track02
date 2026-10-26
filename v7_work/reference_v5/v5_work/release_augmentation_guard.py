"""Resume useful P07 work once P08 is ready, retaining a safe P10 barrier.

The first scheduler's pause is conservative until all assigned folds finish.
This separately authorized controller can resume its exact original PID when
P08/P09 are complete, then pause again with four P07 models still remaining if
P10 is not yet complete. No model values or scores are read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

from monitor_auxiliary_fits import ROOT, PLAN_SHA, ORIGINAL_PIDS, valid_original, fold_complete
from monitor_optional_ssl_tail import active_neural_processes


def running_neural_processes():
    result = []
    for value in active_neural_processes():
        try:
            state = Path(f"/proc/{value['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            continue
        if state not in ("T", "t", "Z"):
            result.append(dict(value, state=state))
    return result


def main():
    output = ROOT / "v5_work/augmentation_release_guard.json"
    if output.exists():
        raise RuntimeError("Release guard record already exists")
    pid = ORIGINAL_PIDS["augmentation"]
    state = dict(plan_sha256=PLAN_SHA, original_pid=pid, monitor_pid=os.getpid(),
                 scores_read=False, events=[])
    paused_by_this_controller = False
    resumed = False

    def record(event, **values):
        entry = dict(event=event, utc_epoch=time.time(), **values)
        state["events"].append(entry)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output)
        print(json.dumps(entry), flush=True)

    try:
        assert valid_original("augmentation")
        os.kill(pid, 0)
        record("signal_reachability_verified")
        while True:
            assert valid_original("augmentation")
            if fold_complete("augmentation", "P10"):
                # The original scheduler also resumes after its full target
                # checks; SIGCONT is idempotent if it got there first.
                os.kill(pid, signal.SIGCONT)
                paused_by_this_controller = False
                record("all_auxiliary_targets_ready_original_resumed")
                return
            if not resumed and fold_complete("augmentation", "P08") and fold_complete("augmentation", "P09"):
                active = [value for value in running_neural_processes() if value["pid"] != pid]
                if len(active) <= 3:
                    os.kill(pid, signal.SIGCONT)
                    resumed = True
                    record("original_resumed_for_P07", other_active_runner_count=len(active),
                           active_runners=active)
            if resumed and not paused_by_this_controller:
                # This completed first replicate leaves two full seed pairs of
                # the fixed P07 fit. The guard therefore acts well before the
                # original could advance through the already-complete P08/P09
                # folders into unfinished P10.
                first_pair = ROOT / "v5_work/neural_results/augmentation/hold_P07/seed42_epoch20.provenance.json"
                if first_pair.exists() and not fold_complete("augmentation", "P10"):
                    os.kill(pid, signal.SIGSTOP)
                    paused_by_this_controller = True
                    record("original_paused_with_two_P07_replicates_remaining")
            time.sleep(30)
    finally:
        if paused_by_this_controller and valid_original("augmentation"):
            os.kill(pid, signal.SIGCONT)
            record("original_resumed_in_finally")


if __name__ == "__main__":
    main()
