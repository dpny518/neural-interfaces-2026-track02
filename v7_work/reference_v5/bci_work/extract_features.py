"""Window-local feature cache for reproducible offline training.

This computes no cross-window statistics and never loads class labels.
Source references and supervised parameters are fitted by training scripts.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'methods_draft'))
from riemann_features import DEFAULT_BANDS,bank_covariances,static_features


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,default=Path(__file__).resolve().parent/'cache')
    ap.add_argument('--all-views',action='store_true',help='Also extract whole-window caches for ablations')
    args=ap.parse_args()
    threadpool_limits(limits=2)
    X=np.load(args.cache/'windows.npy',mmap_mode='r')
    parts=[]
    for name,lo,hi in [('early',0,240),('late',240,480)]:
        c=bank_covariances(X[:,:,lo:hi])
        np.save(args.cache/f'raw_{name}_log_features.npy',static_features(c,'logcov',.05).astype(np.float32))
        parts.append(c)
        print('Extracted',name,c.shape,flush=True)
    np.save(args.cache/'temporal_band_covs.npy',np.concatenate(parts,axis=1))
    del parts
    if args.all_views:
        c=bank_covariances(X)
        np.save(args.cache/'band_covs.npy',c)
        np.save(args.cache/'raw_log_features.npy',static_features(c,'logcov',.05).astype(np.float32))
        np.save(args.cache/'raw_power_features.npy',static_features(c,'logpower',.05).astype(np.float32))
    info=dict(sfreq=120.,bands=[list(x) for x in DEFAULT_BANDS],crops=[[0,240],[240,480]],
              window='Hann',shrinkage=.05,labels_used=False,cross_window_statistics=False)
    (args.cache/'feature_config_reproducible.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':main()
