"""Evaluate a frozen submission through the official streaming implementation."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from evaluation import load_cache,cell_mean_score


def main():
    ap=argparse.ArgumentParser()
    root=Path(__file__).resolve().parent.parent
    ap.add_argument('--submission-dir',type=Path,default=root/'deliverables/submission')
    ap.add_argument('--output',type=Path,default=root/'deliverables/scores.json')
    ap.add_argument('--kit-dir',type=Path,default=root/'reference_audit/2026-competition')
    ap.add_argument('--splits',nargs='+',default=['val','test'])
    args=ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,str(args.kit_dir.resolve()))
    from benchmark_utils.streaming import predict_streams
    from benchmark_utils.metrics import balanced_accuracy,group_scores
    spec=importlib.util.spec_from_file_location('final_submission',args.submission_dir/'submission.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    _,_,dataset=load_cache()
    meta=dict(device='cpu',submission_dir=args.submission_dir,n_chans=41,n_times=480,
              n_classes=3,sfreq=120.,ch_names=dataset['ch_names'])
    model=module.Solver().load_model(meta)
    out=dict(submission_sha256=hashlib.sha256((args.submission_dir/'submission.py').read_bytes()).hexdigest(),
             weights_sha256=hashlib.sha256((args.submission_dir/'weights.pt').read_bytes()).hexdigest(),
             evaluator='official benchmark_utils.streaming.predict_streams',
             competition_commit=dataset['competition_commit'],device='cpu',batch_size=1,
             model_metadata_keys=list(meta),torch_version=torch.__version__,numpy_version=np.__version__,splits={})
    for split in args.splits:
        X,metadata,_=load_cache(split=split)
        def loader():
            for i in range(len(X)):
                yield (torch.from_numpy(np.array(X[i:i+1],copy=True)),torch.tensor(metadata['y'][i:i+1]),
                       dict(stream_id=metadata['stream_id'][i:i+1],record_id=metadata['record_id'][i:i+1],
                            onset=metadata['onset'][i:i+1],context_id=(metadata['interface'][i:i+1]=='BrainHero').astype(np.int64)))
        started=time.perf_counter()
        true,pred,stream_info=predict_streams(model,loader())
        elapsed=time.perf_counter()-started
        pred=pred.astype(np.int64)
        metrics=cell_mean_score(true,pred,metadata)
        official=float(group_scores(balanced_accuracy,true,pred,np.stack([stream_info['stream_id'],stream_info['context_id']],axis=1)).mean())
        assert abs(official-metrics['balanced_accuracy_cell_mean'])<1e-12
        metrics['official_metric_exact_match']=True
        metrics['inference_seconds']=elapsed
        metrics['mean_milliseconds_per_window']=elapsed*1000/len(X)
        out['splits'][split]=metrics
        np.savez_compressed(args.output.parent/(split+'_predictions.npz'),y_true=true,y_pred=pred,
                            subject=metadata['subject'],session=metadata['session'],interface=metadata['interface'],
                            run=metadata['run'],onset=metadata['onset'],stream_id=metadata['stream_id'])
        args.output.write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(split=split,**metrics)),flush=True)


if __name__=='__main__':main()
