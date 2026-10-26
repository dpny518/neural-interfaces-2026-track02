"""Durable job coordination; status records distinguish incomplete from null."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def state(**fields):
    fields['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    tmp=ROOT/'status.partial.json';tmp.write_text(json.dumps(fields,indent=2)+'\n')
    os.replace(tmp,ROOT/'status.json')
    print(json.dumps(fields),flush=True)


def run(script,*args):
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],check=True,cwd=ROOT.parent)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-pid',type=int,required=True);args=ap.parse_args()
    deadline=time.monotonic()+12*3600
    while not (ROOT/'cache/metadata_recovery_check.json').exists():
        state(stage='waiting_for_verified_data',training_started=False)
        try:os.kill(args.data_pid,0)
        except ProcessLookupError:raise RuntimeError('Data preparation stopped; inspect data.log')
        if time.monotonic()>deadline:raise TimeoutError('Data prerequisite timeout')
        time.sleep(30)
    state(stage='synthetic_validation',training_started=False)
    run('selfcheck.py')
    state(stage='preparing_window_local_features',training_started=False)
    run('run_cv.py','--part','features')
    processes={}
    for part in ['neural','classical']:
        log=open(ROOT/f'{part}.log','ab',buffering=0)
        processes[part]=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_cv.py'),'--part',part],
                                         cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
    while any(p.poll() is None for p in processes.values()):
        status={part:{'pid':p.pid,'returncode':p.poll()} for part,p in processes.items()}
        state(stage='cv_fitting',training_started=True,processes=status)
        if any(p.poll() not in [None,0] for p in processes.values()):
            # A failure cannot cause a partial panel to be analyzed or adopted.
            raise RuntimeError('A CV worker failed; other independent worker may still be running')
        time.sleep(30)
    assert all(p.returncode==0 for p in processes.values())
    state(stage='cv_fits_complete_pending_replay',training_started=True,cv_complete=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        state(stage='error',error=repr(e),null_result=False)
        raise
