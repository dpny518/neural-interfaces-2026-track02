"""Write the short report and a self-contained reproducibility bundle."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'


def main():
    scores=json.loads((OUT/'scores.json').read_text())
    contract=json.loads((OUT/'contract_check.json').read_text())
    selection=json.loads((ROOT/'bci_work/selection_final.json').read_text())
    metric_check=json.loads((OUT/'streaming_metric_check.json').read_text())
    info=json.loads((ROOT/'bci_work/cache/dataset_info.json').read_text())
    val,test=scores['splits']['val'],scores['splits']['test']
    sha=hashlib.sha256((OUT/'submission.zip').read_bytes()).hexdigest()
    assert contract['all_passed'] and len(contract['checks'])==16
    assert scores['weights_sha256']==contract['weights_sha256']
    assert scores['submission_sha256']==contract['submission_sha256']
    assert val['balanced_accuracy_cell_mean']==selection['validation_cell_mean']
    assert metric_check['subject_disjoint']
    for split, result in (('val',val),('test',test)):
        assert abs(metric_check['splits'][split]['streaming_cell_mean_balanced_accuracy']-result['balanced_accuracy_cell_mean']) < 1e-12
    cells='\n'.join(f"  {c['subject']} / session {c['session']} / {c['interface']:<9}: {100*c['balanced_accuracy']:.2f}% ({c['n_windows']} windows)" for c in test['cells'])
    report=f'''NEURAL INTERFACES 2026 — TRACK 2 SUBMISSION REPORT
23 October 2026

RESULT
The frozen submission achieved {100*test['balanced_accuracy_cell_mean']:.4f}% test streaming balanced accuracy: +{100*test['balanced_accuracy_cell_mean']-36.1:.4f} percentage points over the user-reported 36.1% baseline. Validation was {100*val['balanced_accuracy_cell_mean']:.4f}% (+{100*val['balanced_accuracy_cell_mean']-63.9:.4f} points over the reported 63.9%). Training-subject leave-one-subject-out (LOSO) cell-mean accuracy was {100*selection['LOSO_cell_mean']:.4f}%.

All headline values use balanced accuracy per participant × session × interface cell, averaged equally. The final ZIP was scored through the actual competition benchmark_utils.streaming.predict_streams function, with B=1, fresh deepcopy/reset per participant-session, and chronological run/window order. Official metric helpers agreed exactly with the local report. Test was evaluated after selection and weights were frozen, and was not used for tuning. These are local public warm-up scores; no Codabench upload was performed. The supplied baseline weights were unavailable, so the baseline numbers are the user's comparison, not a rerun.

An independent recomputation from the saved streaming predictions also matches the official cell-mean scores exactly. Pooled balanced accuracy is a separate diagnostic: validation {100*metric_check['splits']['val']['pooled_balanced_accuracy_diagnostic']:.4f}%, test {100*metric_check['splits']['test']['pooled_balanced_accuracy_diagnostic']:.4f}%. Neither pooled value is used for the headline comparison. See streaming_metric_check.json for the disjoint split assertion and every cell score.

WHAT CHANGED
The user-described v2 baseline applies per-window, per-channel z-scoring and averages EEGNet x3 plus ShallowFBCSPNet x3. This submission replaces that six-network raw-signal ensemble with two learned spatial log-variance networks and a covariance ridge branch, while adding causal, session-local covariance alignment. It uses broadband window power normalization inside its feature model instead of per-channel z-scoring, preserving relative spatial powers. The official input preprocessing remains the same.

The model analyzes the first and second two-second halves of each input window separately. Each half yields seven Hann/FFT covariance bands: 1–4, 4–8, 8–12, 12–18, 18–26, 26–40 and 4–40 Hz. This preserves temporal differences around the feedback boundary while using only the current supplied window.

The final probability ensemble is 45% learned spatial log-variance network (seed 33, checkpoint epoch 75), 45% the same architecture (seed 42, epoch 20), and 10% a Euclidean-aligned matrix-log covariance ridge classifier. Each neural network learns 16 spatial filters per band and uses channel log powers, training-only BatchNorm statistics, dropout 0.5, sensor-gain augmentation and label smoothing. Training losses balance class counts within each scoring cell. The ridge branch uses 5% covariance shrinkage and alpha=10,000. Its native one-hot regression scores use softmax temperature 0.125.

Causal covariance references start from equal-session training-only priors, adapt from already-seen windows, and reset completely at each session. Current features use the prior reference; the current covariance updates it afterward. Running updates use a 0.02 minimum rate and trace-ratio clipping. The neural prior count is 16; the ridge count is 8. No labels or subject/session/interface IDs reach predict, and no state crosses sessions. Whole-recording test statistics are never used.

MODEL SELECTION
Temporal log-centered ridge improved validation from 56.85% without adaptation to 60.49% with it; corresponding LOSO improved from 49.76% to 52.97%. Temporal Euclidean ridge reached 61.74% validation. The neural temporal models reached 64.70% and 64.14%; averaging them reached 65.48%. A predefined 27-candidate coarse ensemble grid selected the final 65.77%/54.36% validation/LOSO model by their equal-weight mean. Separate interface experts did not improve selection metrics and were excluded. Completed raw EEGNet exploratory variants scored 49.57% and 48.06%; the incomplete additional Shallow experiment was stopped and excluded. No claim is made to have exhaustively searched larger architectures.

DATA AND PREPROCESSING
PROTEUS BCI Bordeaux, NEMAR nm000290 v1.0.0: all 587 files / 112 EDFs / 1,593,504,588 bytes verified against the published manifest. Exact subject lists (reference train_test_split seeds 33/33):
  Train: sub-P01, sub-P02, sub-P03, sub-P06, sub-P07, sub-P10 (1,521 windows).
  Validation: sub-P05, sub-P08 (379 windows).
  Test: sub-P04, sub-P09 (755 windows; 4 participant-sessions; 8 interface cells).
Cache metadata omit the sub- prefix, e.g. P04 means source participant sub-P04. The held-out test subjects therefore match the user's stated v2 test subjects exactly. Independent verification with the user's local preprocessing and harness is still appropriate.

The official source transform was reproduced: notch bases 50/60 Hz with the implementation's harmonics below Nyquist; 0.1–75 Hz bandpass at 500 Hz; resample 120 Hz; float32 conversion; 480-sample windows starting at round((cue+1)*120); multiply by 1e6. No preprocessing re-reference, scaling or clamping was added. A 36-window run read through MNE-BIDS and exact official filter methods matched bit-for-bit, maximum difference 0.0 microvolts. The model's window-local feature transformations follow this fixed input pipeline.

TEST CELLS
{cells}

CONTRACT AND RUNTIME
All 16 executed checks passed: actual CompetSolver inheritance and forced local fallback; load_model; int64 predictions for B=0,1,4; valid classes; finite zero/large input handling; deepcopy before and after adaptation; reset replay; batch-versus-sequential equivalence; causal prefix invariance under future-suffix changes; independent clone state; unchanged template; noncontiguous input; and input immutability. Checks use training windows and synthetic signals. CPU was tested; no GPU was available here.

The ZIP contains exactly submission.py and weights.pt at its root, is {int((OUT/'submission.zip').stat().st_size):,} bytes, and requires only NumPy, PyTorch and Python's standard library at inference. Nothing is installed or downloaded at evaluation. Test inference took {test['inference_seconds']:.2f} seconds for 755 streamed windows ({test['mean_milliseconds_per_window']:.2f} ms/window in this CPU run).

submission.zip SHA-256: {sha}

REPRODUCIBILITY AND SOURCES
reproduce.zip includes training/evaluation/contract scripts, selected source checkpoints, frozen selection, complete score/cell reports, OOF predictions and search tables. contract_check.json is the executed contract evidence; streaming_metric_check.json independently recomputes the equal-cell metric from saved predictions.
Competition kit commit: {info['competition_commit']}
NeuroAI v0.4.0 commit: {info['neuroai_commit']}
Official contract: https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md
Submission guide: https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md
Dataset: Dreyer Pauline, Bourdil Manon, Bechon Loic, Kojima Simon, Velut Sebastien, Rimbert Sebastien, Roy Raphaëlle, Lotte Fabien. PROTEUS BCI Bordeaux, v1.0.0, CC-BY-4.0. https://doi.org/10.82901/nemar.nm000290
Alignment concept: He & Wu, Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach. https://arxiv.org/abs/1808.05464
The specific model, causal update variant and scores above are this run's experiments. Published EEGNet/CTNet/REVE reference figures use session-mean scores; they are not presented as exact cell-metric comparisons here.
'''
    (OUT/'report.txt').write_text(report)
    requirements='''--extra-index-url https://download.pytorch.org/whl/cpu
numpy==2.3.5
scipy==1.17.0
scikit-learn==1.8.0
torch==2.14.1+cpu
mne==1.10.2
mne-bids==0.17.0
benchopt==1.10.0
'''
    (ROOT/'bci_work/requirements-training.txt').write_text(requirements)
    readme='''REPRODUCE THE TRACK 2 SUBMISSION

The complete frozen, measured submission is deliverables/submission.zip.
Upload that ZIP directly; its only root files are submission.py and weights.pt.
No packages are installed by inference. This bundle does not include EEG data.

Python used: 3.12.14. CPU training dependencies are pinned in
bci_work/requirements-training.txt. All source priors and supervised weights
use the six training participants. Validation and train-subject LOSO select
the ensemble; test labels do not select parameters. Final checkpoint epochs:
seed33=75, seed42=20; original cosine scheduler horizon=120.

EXACT SUBJECT SPLIT (reference sklearn train_test_split seeds 33/33)
Train: sub-P01, sub-P02, sub-P03, sub-P06, sub-P07, sub-P10
Validation: sub-P05, sub-P08
Test: sub-P04, sub-P09
Cache metadata store P01, P02, etc., without the source directory's sub- prefix.
These sets are disjoint. Window counts are 1,521 / 379 / 755, respectively.
The test contains 4 participant-session streams and 8 interface cells.
All headline scores are equal averages of balanced accuracy within each
participant x session x interface cell, after B=1 chronological inference
with deepcopy/reset at every participant-session boundary.

1. Set up a Python environment and install training/checking dependencies:
   python -m venv .venv
   .venv/bin/pip install -r bci_work/requirements-training.txt
   Use that environment's python for every command below.

2. Download exactly NEMAR nm000290 v1.0.0 with manifest checksum validation:
   python experiments/download_data.py

3. Reproduce the official window cache, then window-local spectral features:
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bci_work/prepare_data.py --workers 4
   OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python bci_work/extract_features.py --all-views

4. Reproduce selected neural-family training (default modes may also run static ablations):
   OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python bci_work/train_neural_cov.py --covariances bci_work/cache/temporal_band_covs.npy --output bci_work/neural_results/temporal --modes causal --seeds 33 42 --epochs 120

5. Reproduce the selected ridge family:
   OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python bci_work/euclidean_sweep.py --temporal
   The selected checkpoint is euclidean_temporal/euclidean_002.npz.

6. Build from frozen choices, or retain the supplied exact measured ZIP:
   python bci_work/build_submission.py --selection bci_work/selection_final.json --out deliverables
   The build requires the original selected checkpoints, which are included.
   Re-training may produce tiny floating-point differences across environments.
   selection_final.json records the SHA-256 of the original selected sources.

7. Clone the official kit for the real base class and actual streaming evaluator:
   git clone https://github.com/neural-interfaces26/2026-competition.git reference_audit/2026-competition
   git -C reference_audit/2026-competition checkout a66bc23d1a0e686b895c093a642b45c74970758b
   MPLCONFIGDIR=/tmp/bci-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bci_work/contract_check.py --submission-dir deliverables/submission.zip --output deliverables/contract_check.json
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bci_work/evaluate_submission.py --submission-dir deliverables/submission --output deliverables/scores.json
   build_submission creates the unpacked submission directory. If using only
   the original ZIP, extract it into deliverables/submission first.

Independent full protocol parity additionally needs NeuroAI v0.4.0 source:
   git clone --branch v0.4.0 https://github.com/facebookresearch/neuroai.git reference_audit/neuroai
   python bci_work/verify_protocol.py
The executed evidence is already in bci_work/cache/protocol_check.json.

Optional robustness/search reproduction:
   python bci_work/temporal_sweep.py
   python bci_work/classical_sweep.py
   python bci_work/crossvalidate.py
   python bci_work/euclidean_loso.py
   python bci_work/crossvalidate_neural.py --seed 33 --epochs 75 --schedule-epochs 120
   python bci_work/crossvalidate_neural.py --seed 42 --epochs 20 --schedule-epochs 120
   python bci_work/select_ensemble.py
Use OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 for these CPU runs.

Evidence:
- deliverables/scores.json: official evaluator scores, all cells, latency.
- deliverables/contract_check.json: 16 executed contract checks and file hashes.
- deliverables/streaming_metric_check.json: independent cell-mean recomputation,
  exact subject lists and separately labeled pooled diagnostic scores.
- bci_work/cv_results/: OOF and validation probabilities, exact ensemble grid,
  fold provenance, and all relevant selection summaries.
- bci_work/cache/dataset_info.json: split, signal transform, channels, source hashes.
- data_provenance/: original published manifest and completed checksum report.

All dataset attributions and public source URLs are in deliverables/report.txt.
The reported baseline was supplied by the user and was not independently rerun.
No Codabench upload was performed. The supplied submission has the exact frozen
weights that produced the scores; no test-based refinement was made.
'''
    (ROOT/'bci_work/REPRODUCE.txt').write_text(readme)
    include=[]
    for folder in ('bci_work','methods_draft','experiments'):
        include.extend((ROOT/folder).glob('*.py'))
    include += [ROOT/'bci_work/requirements-training.txt', ROOT/'bci_work/selection_final.json']
    for name in ('dataset_info.json','protocol_check.json','feature_config.json'):
        p=ROOT/'bci_work/cache'/name
        if p.exists():include.append(p)
    for name in ('classical','temporal_results','euclidean','euclidean_temporal','logreg'):
        for filename in ('results.json','scores.json'):
            p=ROOT/'bci_work'/name/filename
            if p.exists():include.append(p)
    include.extend((ROOT/'bci_work/cv_results').glob('*.json'))
    for name in ('neural_temporal_seed33_oof.npz','neural_temporal_seed42_oof.npz',
                 'euclidean_temporal_001_oof.npz','euclidean_temporal_002_oof.npz',
                 'selected_ensemble_probabilities.npz'):
        include.append(ROOT/'bci_work/cv_results'/name)
    for m in selection['members']:
        include.append(ROOT/'bci_work'/m['file'])
    for folder in ('neural_results','neural_results/temporal','neural_results/csp','raw_results'):
        for name in ('summary.json','training_results.json'):
            p=ROOT/'bci_work'/folder/name
            if p.exists():include.append(p)
    for name in ('submission.zip','report.txt','contract_check.json','streaming_metric_check.json','scores.json','build_info.json',
                 'val_predictions.npz','test_predictions.npz'):
        include.append(OUT/name)
    with zipfile.ZipFile(OUT/'reproduce.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.write(ROOT/'bci_work/REPRODUCE.txt','README.txt')
        for p in sorted(set(include)):
            assert p.is_file(),p
            z.write(p,p.relative_to(ROOT).as_posix())
        for name in ('download_manifest.json','download_verification.json'):
            z.write(ROOT/'data/nm000290/v1.0.0'/name,'data_provenance/'+name)
    with zipfile.ZipFile(OUT/'reproduce.zip') as z:
        assert z.testzip() is None
        print('Reproduction archive entries:',len(z.namelist()))
    print(json.dumps(dict(validation=val['balanced_accuracy_cell_mean'],test=test['balanced_accuracy_cell_mean'],
                         improvement_points=100*test['balanced_accuracy_cell_mean']-36.1,
                         zip_sha256=sha,report_words=len(report.split()),
                         files={p.name:p.stat().st_size for p in [OUT/'submission.zip',OUT/'report.txt',OUT/'contract_check.json',OUT/'reproduce.zip']}),indent=2))


if __name__=='__main__':main()
