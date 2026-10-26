"""Run only the two authorized, conditionally safe disjoint CV tail jobs.

Scheduling evidence only: this program reads completion metadata and file
presence, never predictions, labels or accuracy scores. It does not change a
training recipe or create additional models beyond the fixed ten-fold panel.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "v5_work/neural_results"
PLAN_SHA = "ce7264eaf7efba24d04ba7b694ba4ca51f76762d3f25cee0ce1dc132457cef59"
ORIGINAL_PIDS = {"baseline": 2954, "augmentation": 3204, "adversarial": 3216,
                 "masked_ssl": 2967}
STATE_PATH = ROOT / "v5_work/auxiliary_schedule_state.json"
SCHEDULING_POLICY = "expanded_two_preceding_original_folds_v2"


def emit(event, **values):
    print(json.dumps(dict(event=event, utc_epoch=time.time(), **values)), flush=True)


def valid_original(recipe):
    pid = ORIGINAL_PIDS[recipe]
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return False
    parts = [part.decode() for part in path.read_bytes().split(b"\0") if part]
    return (any(part.endswith("run_neural_cv.py") for part in parts)
            and "--recipe" in parts and parts[parts.index("--recipe") + 1] == recipe
            and "--subjects" not in parts and "--threads" in parts
            and parts[parts.index("--threads") + 1] == "2")


def fold_exists(recipe, subject):
    return (RESULTS / recipe / ("hold_" + subject)).exists()


def fold_complete(recipe, subject):
    folder = RESULTS / recipe / ("hold_" + subject)
    audit, prediction = folder / "isolation_check.json", folder / "predictions.npz"
    if not audit.is_file() or not prediction.is_file():
        return False
    try:
        metadata = json.loads(audit.read_text())
    except json.JSONDecodeError:
        return False
    digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
    if metadata.get("all_passed") is not True or metadata["prediction_sha256"] != digest:
        raise RuntimeError("Incomplete/changed auxiliary prediction evidence: " + str(folder))
    return True


def summary_complete(recipe, subjects=None):
    name = "summary.json" if subjects is None else "summary_" + "_".join(subjects) + ".json"
    path = RESULTS / recipe / name
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    expected = [f"P{i:02d}" for i in range(1, 11)] if subjects is None else subjects
    if not (value["recipe"] == recipe and value["subjects_completed"] == expected
            and value["n_models"] == 6 * len(expected)
            and value["plan_sha256"] == PLAN_SHA and value["selection_performed"] is False):
        raise RuntimeError("Unexpected completion summary: " + str(path))
    return True


def status():
    return {recipe: dict(all10_complete=summary_complete(recipe),
                         created=[f"P{i:02d}" for i in range(1, 11)
                                  if fold_exists(recipe, f"P{i:02d}")],
                         completed=[p.parent.name.removeprefix("hold_")
                                    for p in sorted((RESULTS / recipe).glob("hold_*/isolation_check.json"))])
            for recipe in ORIGINAL_PIDS}


def launch_lead_available(recipe, subjects, jobs):
    """Two original folds must precede the earliest absent auxiliary target."""
    if not valid_original(recipe) or any(fold_exists(recipe, subject) for subject in subjects):
        return False
    assigned = {subject for job in jobs if job["recipe"] == recipe for subject in job["subjects"]}
    original_created = [i for i in range(1, 11)
                        if f"P{i:02d}" not in assigned and fold_exists(recipe, f"P{i:02d}")]
    current = max(original_created, default=0)
    first = min(int(subject[1:]) for subject in subjects)
    return current > 0 and first - current >= 2


def main():
    if STATE_PATH.exists():
        raise RuntimeError("Scheduling state already exists; do not launch another scheduler")
    jobs, processes, handles, paused = [], {}, {}, set()
    decisions = {"baseline_release": "pending", "adversarial_release": "pending"}
    # Normal per-command PID sandboxes expose /proc host IDs without allowing
    # signals to them. Fail before any launch unless supported exec escalation
    # has made the exact authorized original processes reachable.
    for recipe in ("augmentation", "masked_ssl"):
        if not valid_original(recipe):
            raise RuntimeError("Original process missing before signal reachability check")
        os.kill(ORIGINAL_PIDS[recipe], 0)
    emit("signal_reachability_verified", original_pids=ORIGINAL_PIDS,
         scheduling_policy=SCHEDULING_POLICY, monitor_pid=os.getpid())

    def save():
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(plan_sha256=PLAN_SHA, jobs=jobs, decisions=decisions,
                                             paused_originals=sorted(paused),
                                             original_pids=ORIGINAL_PIDS, poll_seconds=30,
                                             monitor_pid=os.getpid(), scheduling_policy=SCHEDULING_POLICY,
                                             scores_read=False, updated_utc_epoch=time.time()), indent=2) + "\n")
        temporary.replace(STATE_PATH)

    def launch(key, recipe, subjects, log_name):
        if not launch_lead_available(recipe, subjects, jobs):
            raise RuntimeError("Auxiliary launch guard changed before launch")
        log_path = ROOT / "v5_work" / log_name
        handle = log_path.open("x")
        environment = dict(os.environ)
        environment.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           PYTHONUNBUFFERED="1")
        command = [str(ROOT / "bci-env/bin/python"), "v5_work/run_neural_cv.py",
                   "--recipe", recipe, "--subjects", *subjects, "--threads", "2"]
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        processes[key], handles[key] = process, handle
        job = dict(key=key, recipe=recipe, subjects=subjects, threads=2, pid=process.pid,
                   command=command, log=str(log_path), launched_utc_epoch=time.time(), status="running")
        jobs.append(job)
        decisions[key] = "launched"
        emit("auxiliary_launched", **job)
        save()

    try:
        previous = None
        while True:
            current = status()
            if current != previous:
                emit("fold_status", status=current)
                previous = current
            if decisions["baseline_release"] == "pending" and current["baseline"]["all10_complete"]:
                if launch_lead_available("augmentation", ["P09", "P10"], jobs):
                    launch("baseline_release", "augmentation", ["P09", "P10"],
                           "neural_augmentation_tail.log")
                else:
                    decisions["baseline_release"] = "skipped_insufficient_lead"
                    emit("auxiliary_skipped", key="baseline_release", reason="specified launch guard false")
            if decisions["adversarial_release"] == "pending" and current["adversarial"]["all10_complete"]:
                if launch_lead_available("augmentation", ["P08"], jobs):
                    launch("adversarial_release", "augmentation", ["P08"], "neural_augmentation_p08.log")
                elif launch_lead_available("masked_ssl", ["P10"], jobs):
                    launch("adversarial_release", "masked_ssl", ["P10"], "neural_masked_ssl_tail.log")
                else:
                    decisions["adversarial_release"] = "skipped_insufficient_lead"
                    emit("auxiliary_skipped", key="adversarial_release", reason="both specified launch guards false")
            for job in jobs:
                if job["status"] != "running":
                    continue
                code = processes[job["key"]].poll()
                if code is not None:
                    handles[job["key"]].close()
                    job.update(returncode=code, completed_utc_epoch=time.time())
                    if (code != 0 or not summary_complete(job["recipe"], job["subjects"])
                            or not all(fold_complete(job["recipe"], s) for s in job["subjects"])):
                        job["status"] = "failed"
                        emit("technical_failure", job=job)
                        raise RuntimeError("Auxiliary fit failed its completion guard")
                    job["status"] = "complete"
                    emit("auxiliary_complete", **job)
            for recipe in ("augmentation", "masked_ssl"):
                recipe_jobs = [job for job in jobs if job["recipe"] == recipe]
                targets = {s for job in recipe_jobs for s in job["subjects"]}
                unfinished = sorted(s for s in targets if not fold_complete(recipe, s))
                if unfinished:
                    first = int(unfinished[0][1:])
                    preceding = max((i for i in range(1, first) if f"P{i:02d}" not in targets), default=0)
                    if preceding and fold_exists(recipe, f"P{preceding:02d}") and recipe not in paused:
                        if not valid_original(recipe):
                            raise RuntimeError("Original process identity changed before overlap guard")
                        os.kill(ORIGINAL_PIDS[recipe], signal.SIGSTOP)
                        paused.add(recipe)
                        emit("original_paused_to_prevent_overlap", recipe=recipe,
                             pid=ORIGINAL_PIDS[recipe], unfinished_targets=unfinished,
                             preceding_original_fold=f"P{preceding:02d}")
                elif recipe in paused:
                    if valid_original(recipe):
                        os.kill(ORIGINAL_PIDS[recipe], signal.SIGCONT)
                    paused.remove(recipe)
                    emit("original_resumed", recipe=recipe, pid=ORIGINAL_PIDS[recipe])
            save()
            if all(value != "pending" for value in decisions.values()) and all(job["status"] == "complete" for job in jobs):
                emit("scheduling_complete", decisions=decisions, jobs=jobs)
                break
            time.sleep(30)
    finally:
        # On a technical failure, stop any unfinished auxiliary writer before
        # releasing an original process. The runner validates and resumes only
        # complete saved seeds, so interruption cannot create two active writers.
        for key, process in processes.items():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                emit("unfinished_auxiliary_stopped_before_original_resume", key=key,
                     pid=process.pid, returncode=process.returncode)
        for recipe in list(paused):
            if valid_original(recipe):
                os.kill(ORIGINAL_PIDS[recipe], signal.SIGCONT)
            paused.remove(recipe)
            emit("original_resumed_in_finally", recipe=recipe, pid=ORIGINAL_PIDS[recipe])
        for handle in handles.values():
            handle.close()
        save()


if __name__ == "__main__":
    main()
