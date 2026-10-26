"""Run the complete fixed round; any failed prerequisite stops adoption."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def run(script,*args):
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],check=True,cwd=ROOT.parent)
if __name__=='__main__':
    subprocess.run([sys.executable,'-m','pip','install','benchopt==1.10.0'],check=True)
    data=None
    if not (ROOT/'cache/metadata_recovery_check.json').exists():
        log=open(ROOT.parent/'data.log','ab',buffering=0)
        data=subprocess.Popen([sys.executable,'-u',str(ROOT/'prepare_data.py')],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        print('DATA_PID',data.pid,flush=True)
    run('orchestrate.py','--data-pid',str(data.pid if data else os.getpid()))
    if data is not None and data.wait()!=0:raise RuntimeError('Data preparation failed')
    run('test_statistics.py')
    run('verify.py')
    run('finalize.py')
