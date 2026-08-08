"""
Post-fMRIPrep cleaning for ds005917 (NIMH Ketamine dataset).

Applies nuisance regression, band-pass filtering, and spatial smoothing to
fMRIPrep output, matching the preprocessing pipeline in Daws et al. 2022.
Saves the cleaned NIfTI to the raw data directory used by react.sh and preprocess.py.

The confound set stays close to the psilodep pipeline (6 motion
parameters + WM + CSF, mirroring its M_V_WMlocal2 stage)

Usage (run from project root):
    python -m preprocessing.ketamine.postprocess --bids-id sub-MOA101 --s-id S01
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
from utils.files import project_root, raw_data_dir

STUDY = 'ds005917'
SESSION = 'before'       # maps to ses-b0 in BIDS
BIDS_SESSION = 'ses-b0'
SPACE = 'MNI152NLin6Asym'  # FSL MNI152 2mm -- must match fmriprep.sh --output-spaces
TR = 2.5                 # seconds; GE Signa HDxt (from BIDS sidecar)
SMOOTHING_FWHM = 6.0     # mm; standard for resting-state
LOW_PASS = 0.08          # Hz
HIGH_PASS = 0.01         # Hz

# Confound columns to regress out (from fMRIPrep confounds TSV)
CONFOUND_COLS = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'white_matter', 'csf',
]

# One-hot regressor families emitted by fMRIPrep. 
#   non_steady_state_outlierXX : auto-detected T1-saturation volumes at scan onset
#   motion_outlierXX           : volumes over --fd-spike-threshold 0.5
CONFOUND_PREFIXES = ['non_steady_state_outlier', 'motion_outlier']


def find_runs(fmriprep_dir, bids_id):
    """
    Locate every preprocessed resting-state run for a subject.

    Most subjects have one. sub-MOA101 and sub-MOA201 have two complete 192-volume
    baseline runs, with no metadata saying which is preferred.
    """
    func_dir = os.path.join(fmriprep_dir, bids_id, BIDS_SESSION, 'func')
    bold_pattern = os.path.join(
        func_dir,
        f'{bids_id}_{BIDS_SESSION}_task-rest_*space-{SPACE}_res-2_desc-preproc_bold.nii.gz'
    )
    bold_files = sorted(glob.glob(bold_pattern))
    if not bold_files:
        raise FileNotFoundError(f'No {SPACE} BOLD found for {bids_id} in {func_dir}')

    runs = []
    for bold_file in bold_files:
        run_tag = os.path.basename(bold_file).split('_space')[0].split(f'{bids_id}_')[1]
        runs.append({
            'bold': bold_file,
            'run_tag': run_tag,
            'confounds': os.path.join(
                func_dir, f'{bids_id}_{run_tag}_desc-confounds_timeseries.tsv'),
            'mask': os.path.join(
                func_dir, f'{bids_id}_{run_tag}_space-{SPACE}_res-2_desc-brain_mask.nii.gz'),
        })
    return runs


def select_run(runs):
    """Pick the run with the lowest mean framewise displacement."""
    if len(runs) == 1:
        return runs[0]

    for run in runs:
        df = pd.read_csv(run['confounds'], sep='\t')
        run['mean_fd'] = float(df['framewise_displacement'].fillna(0.0).mean())

    best = min(runs, key=lambda r: r['mean_fd'])
    print(f'  {len(runs)} runs found; selecting by mean FD:')
    for run in runs:
        marker = ' <-- selected' if run is best else ''
        print(f'    {run["run_tag"]}: mean FD {run["mean_fd"]:.4f} mm{marker}')
    return best


def load_confounds(confounds_file):
    """Load and clean the confounds matrix from the fMRIPrep TSV."""
    df = pd.read_csv(confounds_file, sep='\t')

    missing = [c for c in CONFOUND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing confound columns: {missing}')

    # Expand the one-hot regressor families actually present for this run.
    outlier_cols = sorted(
        c for c in df.columns if any(c.startswith(p) for p in CONFOUND_PREFIXES)
    )
    cols = CONFOUND_COLS + outlier_cols

    n_nss = len([c for c in outlier_cols if c.startswith('non_steady_state_outlier')])
    n_spike = len([c for c in outlier_cols if c.startswith('motion_outlier')])
    print(f'  Confounds: {len(CONFOUND_COLS)} nuisance + {n_nss} non-steady-state '
          f'+ {n_spike} motion-spike = {len(cols)} regressors')

    if len(cols) > 0.25 * len(df):
        print(f'  WARNING: {len(cols)} regressors for {len(df)} volumes '
              f'({100 * len(cols) / len(df):.0f}% of DoF) -- this run is heavily censored')

    confounds = df[cols].copy()

    # fMRIPrep puts NaN in the first row of derivative columns; fill with 0
    confounds = confounds.fillna(0.0)

    return confounds.values


def postprocess(bids_id, s_id, fmriprep_dir, output_base_dir):
    runs = find_runs(fmriprep_dir, bids_id)
    run = select_run(runs)
    print(f'  BOLD:      {run["bold"]}')
    print(f'  Confounds: {run["confounds"]}')
    print(f'  Mask:      {run["mask"]}')

    confounds = load_confounds(run['confounds'])

    masker = NiftiMasker(
        mask_img=run['mask'],
        t_r=TR,
        low_pass=LOW_PASS,
        high_pass=HIGH_PASS,
        smoothing_fwhm=SMOOTHING_FWHM,
        standardize=False,
        detrend=True,
    )

    signals = masker.fit_transform(run['bold'], confounds=confounds)
    cleaned_img = masker.inverse_transform(signals)

    out_dir = os.path.join(output_base_dir, STUDY, SESSION, s_id)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{SESSION}_rest_preproc.nii.gz')
    nib.save(cleaned_img, out_file)
    print(f'  Saved → {out_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-id', required=True, help='BIDS subject ID, e.g. sub-MOA101')
    parser.add_argument('--s-id', required=True, help='Sequential study ID, e.g. S01')
    parser.add_argument(
        '--fmriprep-dir',
        default=None,
        help='Path to fMRIPrep derivatives directory. '
             'Defaults to {project_root}/data/preprocessed/ds005917'
    )
    args = parser.parse_args()

    fmriprep_dir = args.fmriprep_dir or os.path.join(
        project_root(), 'data', 'preprocessed', 'ds005917'
    )
    output_base_dir = raw_data_dir()

    print(f'Postprocessing {args.bids_id} → {args.s_id}')
    postprocess(args.bids_id, args.s_id, fmriprep_dir, output_base_dir)
    print('Done.')


if __name__ == '__main__':
    main()
