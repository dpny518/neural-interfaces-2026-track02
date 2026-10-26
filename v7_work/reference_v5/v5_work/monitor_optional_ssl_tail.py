"""Use one freed auxiliary slot for the identical SSL P10 fit, if safely ahead.

This optional scheduling change reads completion metadata only. It launches no
extra scientific experiment and never reads prediction values or scores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from monitor_auxiliary_fits import (ROOT, PLAN_SHA, ORIGINAL_PIDS, valid_original,
                                    fold_exists, fold_complete, summary_complete)

STATE_PATH = ROOT / "v5_work/auxiliary_ssl_tail_schedule_state.json"
MAIN_STATE_PATH = ROOT / "v5_work/auxiliary_schedule_state.json"


def active_neural_processes():
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            parts = [part.decode() for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError):
            continue
        if len(parts) < 2 or not parts[1].endswith("run_neural_cv.py"):
            continue
        assert "--threads" in parts and parts[parts.index("--threads") + 1] == "2", parts
        result.append(dict(pid=int(directory.name), command=parts))
    return result


def main():
    if STATE_PATH.exists():
        raise RuntimeError("Optional tail scheduling state already exists")
    process = None
    log_handle = None
    paused = False
    state = dict(plan_sha256=PLAN_SHA, policy="optional_ssl_P10_after_auxiliary_completion",
                 monitor_pid=os.getpid(), original_ssl_pid=ORIGINAL_PIDS["masked_ssl"],
                 poll_seconds=30, scores_read=False, decision="pending", events=[])

    def record(event, **values):
        entry = dict(event=event, utc_epoch=time.time(), **values)
        state["events"].append(entry)
        state["paused_original"] = paused
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(STATE_PATH)
        print(json.dumps(entry), flush=True)

    try:
        if not valid_original("masked_ssl"):
            state["decision"] = "skipped_original_not_running"
            record("optional_ssl_tail_skipped")
            return
        os.kill(ORIGINAL_PIDS["masked_ssl"], 0)
        record("signal_reachability_verified", original_ssl_pid=ORIGINAL_PIDS["masked_ssl"])
        while process is None:
            if fold_exists("masked_ssl", "P09") or fold_exists("masked_ssl", "P10"):
                state["decision"] = "skipped_insufficient_lead_or_existing_target"
                record("optional_ssl_tail_skipped")
                return
            if not valid_original("masked_ssl"):
                raise RuntimeError("Original SSL process ended before authorized auxiliary launch")
            if MAIN_STATE_PATH.exists():
                main_state = json.loads(MAIN_STATE_PATH.read_text())
                assert main_state["plan_sha256"] == PLAN_SHA
                # The existing monitor may choose SSL P10 as its second task.
                # Wait for that choice to resolve, preventing two schedulers
                # from simultaneously allocating the same target.
                main_resolved = main_state["decisions"]["adversarial_release"] != "pending"
                free_slot = any(job["status"] == "complete" for job in main_state["jobs"])
                ssl_already_assigned = any(job["recipe"] == "masked_ssl" for job in main_state["jobs"])
                if ssl_already_assigned:
                    state["decision"] = "skipped_ssl_tail_already_assigned"
                    record("optional_ssl_tail_skipped")
                    return
                if main_resolved and free_slot:
                    active = active_neural_processes()
                    if len(active) > 3:
                        raise RuntimeError("Auxiliary completion did not release a two-thread runner slot")
                    # Repeat the decisive guards immediately before launching.
                    assert valid_original("masked_ssl")
                    assert not fold_exists("masked_ssl", "P09")
                    assert not fold_exists("masked_ssl", "P10")
                    command = [str(ROOT / "bci-env/bin/python"), "v5_work/run_neural_cv.py",
                               "--recipe", "masked_ssl", "--subjects", "P10", "--threads", "2"]
                    log_path = ROOT / "v5_work/neural_masked_ssl_optional_tail.log"
                    log_handle = log_path.open("x")
                    environment = dict(os.environ)
                    environment.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
                                       MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                               stdout=log_handle, stderr=subprocess.STDOUT)
                    state.update(decision="launched", pid=process.pid, command=command,
                                 log=str(log_path), other_active_runners_at_launch=active)
                    record("optional_ssl_tail_launched", pid=process.pid,
                           other_active_runner_count=len(active))
                    break
            time.sleep(30)
        while True:
            code = process.poll()
            if code is not None:
                state["returncode"] = code
                if code != 0 or not summary_complete("masked_ssl", ["P10"]) or not fold_complete("masked_ssl", "P10"):
                    state["decision"] = "technical_failure"
                    record("optional_ssl_tail_failed", returncode=code)
                    raise RuntimeError("Optional SSL tail failed its completion guard")
                state["decision"] = "complete"
                record("optional_ssl_tail_complete", pid=process.pid)
                break
            if fold_exists("masked_ssl", "P09") and not paused and not fold_complete("masked_ssl", "P10"):
                assert valid_original("masked_ssl")
                os.kill(ORIGINAL_PIDS["masked_ssl"], signal.SIGSTOP)
                paused = True
                record("original_ssl_paused_before_overlap", pid=ORIGINAL_PIDS["masked_ssl"])
            time.sleep(30)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            record("unfinished_optional_tail_stopped", pid=process.pid, returncode=process.returncode)
        if paused:
            if valid_original("masked_ssl"):
                os.kill(ORIGINAL_PIDS["masked_ssl"], signal.SIGCONT)
            paused = False
            record("original_ssl_resumed_in_finally", pid=ORIGINAL_PIDS["masked_ssl"])
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
