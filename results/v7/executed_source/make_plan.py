"""Declare the reconstruction and fixed v7 panel before any EEG fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent
base = json.loads((ROOT / 'reference_v5/v5_work/plan.json').read_text())
plan = {
    'schema': 'track2-v7-reconstructed-v6-v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'baseline': 'reconstructed_v6',
    'original_v6_recovered': False,
    'historical_v6_cv_percent_unverified': 57.91,
    'historical_v6_sha256_unverified': 'ce6c36d951c07d41db28179a541bac56865ddda717861a91d413bd198d80daa4',
    'no_score_matching_or_reconstruction_search': True,
    'public_subjects': [f'P{i:02d}' for i in range(1, 11)],
    'seed_pairs': [[33, 42], [133, 142], [233, 242]],
    'fixed_epochs': [75, 20],
    'final_seed_pair': [33, 42],
    'supervised': base['neural_definitions']['supervised'],
    'ssl': base['neural_definitions']['recipes']['masked_ssl'],
    'neural_alignment': base['neural_alignment'],
    'classical_definition': base['classical_definition'],
    'bands': [[1, 4], [4, 8], [8, 12], [12, 18], [18, 26], [26, 40], [4, 40]],
    'views': [[0, 240], [240, 480], [0, 480]],
    'model': {'n_filters': 16, 'n_chans': 41, 'n_classes': 3,
              'include_diagonal': True, 'dropout': 0.5, 'hidden': 128,
              'batchnorm_momentum': 0.05,
              'shared_spatial_weights': 'one matrix per frequency band, shared across views',
              'head': 'z=dropout(BatchNorm(features)); Linear(z,3)+Linear(dropout(GELU(Linear(z,128))),3)',
              'initialization': 'PyTorch defaults; normalized random spatial weights; both head branches initialized normally'},
    'reconstruction_inferences': ['exact residual-head topology/initialization was lost and is fixed above',
        'v5 optimizer, label smoothing, gains, BN momentum, losses and epoch schedule retained',
        'classical v6 nested recipe was lost; replacement below is newly declared, not recovered'],
    'candidates': ['reconstructed_v6', 'masked_ssl', 'nested_classical', 'expanded_grid', 'cross_subject_mixup'],
    'expanded_grid': {'bands': [[1,4],[4,8],[8,12],[12,18],[18,26],[26,40],[4,40],
                              [8,10],[10,13],[13,20],[20,30]],
                      'views': [[0,240],[240,480],[0,480],[120,360]],
                      'broadband_band_index': 6,
                      'classical_branch': 'unchanged original 14-bank covariance ridge'},
    'cross_subject_mixup': {'alpha': 0.2, 'probability': 0.5, 'seed_offset': 300000,
                           'partners': 'uniformly drawn different-subject examples in fitting minibatch',
                           'loss': 'lambda*w_i*CE(y_i)+(1-lambda)*w_j*CE(y_j)',
                           'rationale': 'smooth decision boundaries across fitting-subject covariance mixtures'},
    'nested_classical': {'replacement_for_missing_v6_recipe': True,
        'neural_branches': 'exactly shared freshly fitted reconstructed_v6 branches',
        'weights': [0.45,0.45,0.05,0.05],
        'members': ['original covariance ridge', 'inner-selected aligned log-power member'],
        'grid': [{'kind':'linear','alpha':10.0}, {'kind':'rbf','alpha':1.0,'gamma':'1/feature_dimension'}],
        'features': '14-bank causally Euclidean-aligned log diagonal covariance; training-only weighted coordinate standardization',
        'reference': 'equal fitting-session arithmetic raw covariance; prior_count=8,ema=.02,shrinkage=.05,clip=10',
        'target': 'weighted centered one-hot labels', 'temperature':0.125,
        'inner_protocol': 'LOSO among outer fitting subjects; exclude outer and inner held-out subjects from every fit/reference/scale',
        'selection': 'largest inner equal-cell balanced accuracy; linear before rbf for exact ties',
        'final_selection': 'repeat inner LOSO on the authorized final or legacy fitting population only'},
    'metric': 'equal mean balanced accuracy over subject/session/interface cells',
    'bootstrap': {'draws':10000,'seed':20261024,
        'unit':'participant retaining all sessions/interfaces',
        'paired_replicates':'average matched cell differences across three seed pairs before resampling',
        'weighting':'sum sampled cell differences / sampled cell count; not equal subject means'},
    'adoption': 'mean paired gain > participant-cluster bootstrap sample SE',
    'winner': 'largest gain minus SE among passes; lexicographic tie break',
    'historical_test_role_retired': True,
    'final_subjects': base['final_subjects'],
    'legacy_diagnostic': base['legacy_diagnostic'],
    'null_result': 'no candidate final/legacy fit or new submission ZIP; reconstruction is not asserted identical to missing v6',
    'compute': {'device':'cuda:0','precision':'float32, no AMP/TF32',
                'deterministic_algorithms':True,'cpu_threads':2},
    'competition_commit':base['competition_commit'],
    'source_freeze': 'all active implementation files hashed before first fit; bug-only changes recorded and affected work rerun',
}
out = ROOT / 'plan.json'
if out.exists():
    raise SystemExit('Refusing to overwrite the declared plan')
raw = (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode()
out.write_bytes(raw)
(ROOT / 'plan.sha256').write_text(hashlib.sha256(raw).hexdigest() + '\n')
print(hashlib.sha256(raw).hexdigest())
