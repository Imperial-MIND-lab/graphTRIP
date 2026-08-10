"""
Quality control for the ds005917 (NIMH Ketamine) preprocessing pipeline.

Runs the checks that catch the failure modes of stages 1-5 before they become
36 wasted array jobs or, worse, silently wrong node/edge features.

Usage (run from project root):

    # Stage 0a: is the BIDS tree complete enough for fMRIPrep to start?
    python -m preprocessing.ketamine.qc bids

    # Stage 0b: cheapest possible check -- do the template grids agree?
    #           Run this on ONE subject before submitting anything else.
    python -m preprocessing.ketamine.qc grid --s-id S01

    # Per-subject report (stages 1-5): motion, cleaning, registration,
    # REACT maps, parcellated features.
    python -m preprocessing.ketamine.qc subject --s-id S01 --atlas schaefer100

    # Cohort report: motion distribution, QC-FC, feature comparability with
    # psilodep1 (the transfer target).
    python -m preprocessing.ketamine.qc cohort --atlas schaefer100

Figures are written to outputs/qc_ds005917/.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2026-08-03
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import plotting as nlplot
from nilearn import image as nlimage
from nilearn.maskers import NiftiMasker

from utils.files import project_root, raw_data_dir, get_subject_id
from preprocessing.metrics import get_atlas, parcellate

STUDY = 'ds005917'
SESSION = 'before'
BIDS_SESSION = 'ses-b0'
TR = 2.5
LOW_PASS, HIGH_PASS = 0.08, 0.01
FD_THRESH = 0.5          # mm, matches --fd-spike-threshold
FD_MEAN_EXCLUDE = 0.3    # mm, conventional subject-level exclusion
FD_PCT_EXCLUDE = 20.0    # % of volumes over FD_THRESH

# Recessive grid, thin marks -- diagnostic plots, not presentation charts.
sns.set_theme(style='ticks', rc={'axes.grid': True, 'grid.alpha': 0.25,
                                 'grid.linewidth': 0.5, 'axes.linewidth': 0.8})
SEQ_CMAP = sns.color_palette('YlGnBu', as_cmap=True)   # magnitude
DIV_CMAP = 'RdBu_r'                                    # polarity, neutral midpoint
INK = '#2b2b2b'
ACCENT = '#1f6f8b'
FLAG = '#b4432c'


# ------------------------------------------------------------------ paths --

def qc_dir():
    d = os.path.join(project_root(), 'outputs', 'qc_ds005917')
    os.makedirs(d, exist_ok=True)
    return d


def subject_map():
    return pd.read_csv(os.path.join(project_root(), 'preprocessing',
                                    'ketamine', 'subject_map.csv'))


def bids_id_of(s_id):
    smap = subject_map()
    row = smap[smap['s_id'] == s_id]
    if row.empty:
        raise ValueError(f'{s_id} not in subject_map.csv')
    return row['bids_id'].iloc[0]


def fmriprep_func_dir(bids_id):
    return os.path.join(project_root(), 'data', 'preprocessed', STUDY,
                        bids_id, BIDS_SESSION, 'func')


def cleaned_nifti(s_id):
    return os.path.join(raw_data_dir(), STUDY, SESSION, s_id,
                        f'{SESSION}_rest_preproc.nii.gz')


def confounds_tsv(bids_id):
    hits = sorted(glob.glob(os.path.join(
        fmriprep_func_dir(bids_id),
        f'{bids_id}_{BIDS_SESSION}_task-rest_*desc-confounds_timeseries.tsv')))
    if not hits:
        raise FileNotFoundError(f'No confounds TSV for {bids_id}')
    return hits[0]


def fmriprep_bold(bids_id):
    hits = sorted(glob.glob(os.path.join(
        fmriprep_func_dir(bids_id),
        f'{bids_id}_{BIDS_SESSION}_task-rest_*space-MNI152NLin*_res-2_desc-preproc_bold.nii.gz')))
    if not hits:
        raise FileNotFoundError(f'No MNI BOLD for {bids_id}')
    return hits[0]


def react_dir(s_id, receptor_set='Believeau-5'):
    return os.path.join(project_root(), 'data', 'raw', STUDY, SESSION,
                        'MNI_2mm', f'REACT_{receptor_set}', s_id)


def gm_mask_file():
    return os.path.join(project_root(), 'data', 'raw', 'react_data',
                        'masks', 'gm_mask.nii.gz')


def feature_dir(s_id, atlas):
    return os.path.join(project_root(), 'data', 'raw', STUDY, SESSION, atlas, s_id)


# -------------------------------------------------------- stage 0: bids --

def check_bids():
    """
    The aws sync in ketamine_dataset.md cannot fetch root-level files (each --include
    pattern requires a sub-*/, phenotype/ or participants* prefix), so these are the
    files most likely to be absent.
    """
    bids_dir = os.path.join(project_root(), 'data', 'raw', STUDY)
    ok = True

    print(f'BIDS root: {bids_dir}\n')
    print('Required root files:')
    for fname, why in [
            ('dataset_description.json', 'required at the root of every BIDS dataset'),
            ('task-rest_bold.json', 'carries TaskName; per-run sidecars do not'),
            ('task-rest_physio.json', 'sidecar for the *_physio.tsv.gz files'),
            ('.bidsignore', 'excludes phenotype/ from validation')]:
        present = os.path.exists(os.path.join(bids_dir, fname))
        ok &= present
        print(f'  [{"OK " if present else "MISSING"}] {fname:28s} {why}')

    # TaskName must reach every functional run, via inheritance from the root sidecar.
    root_bold = os.path.join(bids_dir, 'task-rest_bold.json')
    if os.path.exists(root_bold):
        import json
        with open(root_bold) as fh:
            has_taskname = 'TaskName' in json.load(fh)
        ok &= has_taskname
        print(f'  [{"OK " if has_taskname else "MISSING"}] '
              f'{"-> TaskName field":28s} mandatory for BIDS func runs')

    # graphTRIP writes annotations.csv and before/ into the BIDS root; the validator
    # rejects both and fMRIPrep aborts before doing any work.
    print('\nNon-BIDS files in the BIDS root (must be listed in .bidsignore):')
    ignore_path = os.path.join(bids_dir, '.bidsignore')
    ignored = []
    if os.path.exists(ignore_path):
        with open(ignore_path) as fh:
            ignored = [line.strip() for line in fh if line.strip()]

    for entry, pattern in [('annotations.csv', 'annotations.csv'),
                           (SESSION, f'{SESSION}/'),
                           ('phenotype', 'phenotype/')]:
        if not os.path.exists(os.path.join(bids_dir, entry)):
            continue
        covered = pattern in ignored or entry in ignored
        ok &= covered
        print(f'  [{"OK " if covered else "MISSING"}] {entry:28s} '
              f'{"covered by .bidsignore" if covered else f"add `{pattern}` to .bidsignore"}')

    print('\nPer-subject inputs (from subject_map.csv):')
    smap = subject_map()
    problems = []
    for _, r in smap.iterrows():
        s_id, bids_id = r['s_id'], r['bids_id']
        bold = glob.glob(os.path.join(bids_dir, bids_id, BIDS_SESSION, 'func',
                                      f'{bids_id}_{BIDS_SESSION}_task-rest_*_bold.nii.gz'))
        # T1w may live in any session -- fMRIPrep's anatomical workflow is subject-level,
        # which is how sub-MOA136 (no ses-b0 T1w) is handled.
        t1w = glob.glob(os.path.join(bids_dir, bids_id, 'ses-*', 'anat', '*_T1w.nii.gz'))
        if not bold or not t1w:
            problems.append((s_id, bids_id, len(bold), len(t1w)))
        elif len(bold) > 1 or not any(BIDS_SESSION in f for f in t1w):
            sessions = sorted({os.path.basename(f).split('_')[1] for f in t1w})
            print(f'  [NOTE] {s_id} ({bids_id}): {len(bold)} rest run(s), '
                  f'T1w in {", ".join(sessions)}')

    if problems:
        ok = False
        for s_id, bids_id, nb, nt in problems:
            print(f'  [FAIL] {s_id} ({bids_id}): {nb} rest run(s), {nt} T1w')
    print(f'  {len(smap) - len(problems)}/{len(smap)} subjects have both a '
          f'rest run and a T1w')

    print('\n' + ('BIDS tree looks ready for fMRIPrep.\n' if ok else
                  'BIDS tree is INCOMPLETE -- see the README for the download commands.\n'))
    return ok


# ------------------------------------------------------- stage 0: grids --

def describe_grid(path_or_img, label):
    img = nib.load(path_or_img) if isinstance(path_or_img, str) else path_or_img
    return {'label': label, 'shape': tuple(img.shape[:3]),
            'zooms': tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]),
            'affine': np.round(img.affine, 3)}


def check_grids(s_id=None, atlas='schaefer100'):
    """
    REACT's PET atlas and GM mask, and nilearn's Schaefer parcellation, all live
    on the FSL MNI152 2 mm grid (91x109x91). If fMRIPrep writes
    MNI152NLin2009cAsym:res-2 instead (97x115x97), REACT will not run and the
    parcellation will be silently resampled onto the wrong template.
    """
    grids = [describe_grid(gm_mask_file(), 'REACT gm_mask (reference)')]

    pet = os.path.join(project_root(), 'data', 'raw', 'react_data',
                       '5-HT_atlas_2mm', 'concatenated', 'Believeau-5', 'pet_atlas.nii.gz')
    if os.path.exists(pet):
        grids.append(describe_grid(pet, 'REACT pet_atlas'))
    else:
        print(f'  MISSING: {pet}\n  (check the spelling of the concatenated/ '
              f'subdirectory -- the code expects "Believeau", not "Believau")')

    grids.append(describe_grid(get_atlas(atlas)['maps'], f'nilearn {atlas}'))

    if s_id:
        for path, label in [(cleaned_nifti(s_id), f'{s_id} cleaned BOLD'),
                            (None, None)]:
            if path and os.path.exists(path):
                grids.append(describe_grid(path, label))
            elif path:
                print(f'  (not yet produced: {path})')
        try:
            fp = fmriprep_bold(bids_id_of(s_id))
            grids.append(describe_grid(fp, f'{s_id} fMRIPrep BOLD'))
        except FileNotFoundError:
            print('  (fMRIPrep output not found yet)')

    ref = grids[0]
    print(f'\n{"grid":<32} {"shape":<18} {"voxel size":<18} matches reference')
    print('-' * 92)
    ok = True
    for g in grids:
        same = (g['shape'] == ref['shape']
                and np.allclose(g['affine'], ref['affine'], atol=1e-3))
        if g is not ref and not same:
            ok = False
        print(f'{g["label"]:<32} {str(g["shape"]):<18} {str(g["zooms"]):<18} '
              f'{"--" if g is ref else ("YES" if same else "NO  <-- MISMATCH")}')
    print('-' * 92)
    if ok:
        print('All grids agree. REACT and parcellation will operate on a common space.\n')
    else:
        print('MISMATCH. Fix --output-spaces before running stages 3-5.\n'
              'Use MNI152NLin6Asym:res-2 (the FSL MNI152 template these atlases '
              'were built in), not MNI152NLin2009cAsym:res-2.\n')
    return ok


# ------------------------------------------------------- motion helpers --

def motion_summary(bids_id):
    """Mean/max FD and % of volumes over threshold, from the fMRIPrep confounds."""
    df = pd.read_csv(confounds_tsv(bids_id), sep='\t')
    fd = df['framewise_displacement'].fillna(0.0).values
    n_nss = len([c for c in df.columns if c.startswith('non_steady_state_outlier')])
    return {'fd': fd,
            'mean_fd': float(np.mean(fd)),
            'max_fd': float(np.max(fd)),
            'pct_over': float(100 * np.mean(fd > FD_THRESH)),
            'n_vols': len(fd),
            'n_nonsteady': n_nss,
            'dvars': df['std_dvars'].fillna(0.0).values if 'std_dvars' in df else None}


# ----------------------------------------------- stage 1-5: per subject --

def qc_subject(s_id, atlas='schaefer100', receptor_set='Believeau-5'):
    bids_id = bids_id_of(s_id)
    print(f'QC for {s_id} ({bids_id})')

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(6, 2, hspace=0.45, wspace=0.25,
                          height_ratios=[1, 1.4, 1, 1, 1.3, 1.3])
    notes = []

    # --- 1. Motion (stage 1) -------------------------------------------- #
    ax = fig.add_subplot(gs[0, :])
    try:
        m = motion_summary(bids_id)
        t = np.arange(m['n_vols']) * TR
        ax.plot(t, m['fd'], lw=1.0, color=ACCENT)
        ax.axhline(FD_THRESH, ls='--', lw=1.0, color=FLAG)
        ax.fill_between(t, 0, m['fd'], where=m['fd'] > FD_THRESH,
                        color=FLAG, alpha=0.35, step='mid')
        ax.set_xlabel('time (s)'); ax.set_ylabel('FD (mm)')
        ax.set_title(f'Framewise displacement — mean {m["mean_fd"]:.3f} mm, '
                     f'max {m["max_fd"]:.2f} mm, {m["pct_over"]:.1f}% of volumes > {FD_THRESH} mm',
                     loc='left', color=INK)
        if m['mean_fd'] > FD_MEAN_EXCLUDE:
            notes.append(f'HIGH MOTION: mean FD {m["mean_fd"]:.3f} > {FD_MEAN_EXCLUDE} mm')
        if m['pct_over'] > FD_PCT_EXCLUDE:
            notes.append(f'HIGH MOTION: {m["pct_over"]:.1f}% of volumes > {FD_THRESH} mm')
        if m['n_nonsteady'] > 0:
            notes.append(f'{m["n_nonsteady"]} non-steady-state volume(s) detected and '
                         f'NOT removed by postprocess.py')
    except Exception as e:
        ax.text(0.5, 0.5, f'motion unavailable: {e}', ha='center', transform=ax.transAxes)

    # --- 2. Carpet plots, before vs after cleaning (stages 1 -> 2) ------- #
    for col, (path, label) in enumerate([
            (lambda: fmriprep_bold(bids_id), 'fMRIPrep output (uncleaned)'),
            (lambda: cleaned_nifti(s_id), 'after nuisance + bandpass + smoothing')]):
        ax = fig.add_subplot(gs[1, col])
        try:
            img = path()
            nlplot.plot_carpet(img, axes=ax, detrend=False)
            ax.set_title(label, loc='left', fontsize=10, color=INK)
        except Exception as e:
            ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8,
                    transform=ax.transAxes)
            ax.set_axis_off()

    # --- 3. Power spectrum: did the band-pass actually happen? ---------- #
    ax = fig.add_subplot(gs[2, 0])
    try:
        masker = NiftiMasker(mask_strategy='epi', standardize=True)
        pre = masker.fit_transform(fmriprep_bold(bids_id)).mean(axis=1)
        post = NiftiMasker(mask_img=masker.mask_img_, standardize=True
                           ).fit_transform(cleaned_nifti(s_id)).mean(axis=1)
        freqs = np.fft.rfftfreq(len(pre), d=TR)
        for sig, lab, c in [(pre, 'uncleaned', '#9aa5ab'), (post, 'cleaned', ACCENT)]:
            ax.semilogy(freqs, np.abs(np.fft.rfft(sig)) ** 2 + 1e-12, lw=1.5,
                        label=lab, color=c)
        ax.axvline(HIGH_PASS, ls='--', lw=1.0, color=FLAG)
        ax.axvline(LOW_PASS, ls='--', lw=1.0, color=FLAG)
        ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('power')
        ax.set_title('Global signal spectrum — power outside the dashed band\n'
                     'should collapse after cleaning', loc='left', fontsize=10, color=INK)
        ax.legend(frameon=False)
        band = (freqs > LOW_PASS * 1.5)
        if band.any():
            leak = np.abs(np.fft.rfft(post))[band].max() / (np.abs(np.fft.rfft(post)).max() + 1e-12)
            if leak > 0.1:
                notes.append(f'Band-pass looks ineffective: {leak:.0%} residual power '
                             f'above {LOW_PASS} Hz')
    except Exception as e:
        ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8, transform=ax.transAxes)

    # --- 4. Registration / coverage ------------------------------------- #
    ax = fig.add_subplot(gs[2, 1])
    try:
        mean_img = nlimage.mean_img(cleaned_nifti(s_id))
        nlplot.plot_roi(gm_mask_file(), bg_img=mean_img, axes=ax,
                        display_mode='z', cut_coords=5, alpha=0.35,
                        cmap='autumn', black_bg=False,
                        title='REACT GM mask on subject mean BOLD')
    except Exception as e:
        ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8, transform=ax.transAxes)
        ax.set_axis_off()

    # --- 5. REACT stage-2 maps (stage 4) -------------------------------- #
    ax = fig.add_subplot(gs[3, :])
    try:
        maps = sorted(glob.glob(os.path.join(react_dir(s_id, receptor_set),
                                             '*_react_stage2_map*.nii.gz')))
        if not maps:
            raise FileNotFoundError('no *_react_stage2_map*.nii.gz')
        stats = []
        for mp in maps:
            d = nib.load(mp).get_fdata()
            nz = d[d != 0]
            stats.append({'map': os.path.basename(mp).split('_react_stage2_')[-1].replace('.nii.gz', ''),
                          'n_nonzero': int(nz.size), 'mean': float(np.mean(nz)) if nz.size else np.nan,
                          'sd': float(np.std(nz)) if nz.size else np.nan,
                          'n_nan': int(np.isnan(d).sum())})
        sdf = pd.DataFrame(stats)
        parts = ax.violinplot([nib.load(mp).get_fdata()[nib.load(mp).get_fdata() != 0]
                               for mp in maps], showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(ACCENT); pc.set_alpha(0.55)
        ax.set_xticks(range(1, len(maps) + 1)); ax.set_xticklabels(sdf['map'])
        ax.set_ylabel('stage-2 beta'); ax.axhline(0, lw=0.8, color=INK, alpha=0.4)
        ax.set_title(f'REACT {receptor_set} stage-2 maps — distributions should be '
                     'unimodal, centred near 0, non-degenerate',
                     loc='left', fontsize=10, color=INK)
        for _, r in sdf.iterrows():
            if r['n_nonzero'] == 0 or not np.isfinite(r['sd']) or r['sd'] == 0:
                notes.append(f'REACT map {r["map"]} is empty or constant')
            if r['n_nan'] > 0:
                notes.append(f'REACT map {r["map"]} contains {r["n_nan"]} NaNs')
        print(sdf.to_string(index=False))
    except Exception as e:
        ax.text(0.5, 0.5, f'REACT unavailable:\n{e}', ha='center', fontsize=8,
                transform=ax.transAxes)

    # --- 6. Parcellated features (stage 5) ------------------------------ #
    fdir = feature_dir(s_id, atlas)
    ax = fig.add_subplot(gs[4, 0])
    try:
        bold = np.genfromtxt(os.path.join(fdir, 'bold.csv'), delimiter=',')
        n_flat = int(np.sum(np.std(bold, axis=0) < 1e-8))
        n_nan = int(np.isnan(bold).sum())
        sd = np.std(bold, axis=0)
        ax.bar(np.arange(len(sd)), sd, color=ACCENT, width=1.0)
        ax.set_xlabel('parcel'); ax.set_ylabel('temporal SD')
        ax.set_title(f'Parcel signal, {atlas} — {bold.shape[0]} vols x {bold.shape[1]} parcels\n'
                     f'{n_flat} flat parcel(s), {n_nan} NaN(s)', loc='left',
                     fontsize=10, color=INK)
        if n_flat or n_nan:
            notes.append(f'{n_flat} flat and {n_nan} NaN entries in bold.csv — '
                         'preprocess.py patches these by interpolating neighbouring '
                         'parcels, which is silent and wrong if coverage is the cause')
    except Exception as e:
        ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8, transform=ax.transAxes)

    ax = fig.add_subplot(gs[4, 1])
    try:
        edge = pd.read_csv(os.path.join(fdir, 'edge.csv'))
        n = int(np.sqrt(len(edge)))
        fc = edge['functional_connectivity'].values.reshape(n, n)
        im = ax.imshow(fc, cmap=DIV_CMAP, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, label='r')
        ax.set_title(f'Functional connectivity ({atlas})', loc='left',
                     fontsize=10, color=INK)
    except Exception as e:
        ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8, transform=ax.transAxes)
        ax.set_axis_off()

    # --- 7. Comparability with psilodep1 (the transfer target) ---------- #
    ax = fig.add_subplot(gs[5, :])
    try:
        node = pd.read_csv(os.path.join(fdir, 'node.csv'))
        ref_f = os.path.join(project_root(), 'data', 'raw', 'psilodep1', 'before',
                             atlas, 'S01', 'node.csv')
        ref = pd.read_csv(ref_f)
        shared = [c for c in node.columns if c in ref.columns]
        long = pd.concat([
            node[shared].melt(var_name='feature', value_name='value').assign(dataset='ds005917'),
            ref[shared].melt(var_name='feature', value_name='value').assign(dataset='psilodep1 S01'),
        ])
        sns.violinplot(data=long, x='feature', y='value', hue='dataset', ax=ax,
                       split=True, inner='quart', linewidth=0.8,
                       palette=[ACCENT, '#9aa5ab'])
        ax.set_title('Node features vs psilodep1 — distributions must be on a '
                     'comparable scale for transfer to mean anything',
                     loc='left', fontsize=10, color=INK)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(frameon=False, title=None)
        missing = [c for c in ref.columns if c not in node.columns]
        if missing:
            notes.append(f'node.csv is missing features present in psilodep1: {missing}')
    except Exception as e:
        ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8, transform=ax.transAxes)

    fig.suptitle(f'ds005917 QC — {s_id} ({bids_id})', fontsize=15, y=0.995, color=INK)
    out = os.path.join(qc_dir(), f'qc_{s_id}_{atlas}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')

    print('\n' + ('=' * 70))
    if notes:
        print('FLAGS:')
        for n in notes:
            print(f'  - {n}')
    else:
        print('No flags raised.')
    print('=' * 70)
    return notes


# ------------------------------------------------------ cohort-level QC --

def qc_cohort(atlas='schaefer100'):
    smap = subject_map()
    rows, fcs, fds = [], [], []
    for _, r in smap.iterrows():
        s_id, bids_id = r['s_id'], r['bids_id']
        rec = {'s_id': s_id, 'bids_id': bids_id, 'group': r['group']}
        try:
            m = motion_summary(bids_id)
            rec.update(mean_fd=m['mean_fd'], max_fd=m['max_fd'],
                       pct_over=m['pct_over'], n_vols=m['n_vols'],
                       n_nonsteady=m['n_nonsteady'])
        except Exception:
            rec.update(mean_fd=np.nan, max_fd=np.nan, pct_over=np.nan,
                       n_vols=np.nan, n_nonsteady=np.nan)
        edge_f = os.path.join(feature_dir(s_id, atlas), 'edge.csv')
        if os.path.exists(edge_f):
            e = pd.read_csv(edge_f)['functional_connectivity'].values
            n = int(np.sqrt(len(e)))
            fc = e.reshape(n, n)
            fcs.append(fc[np.triu_indices(n, k=1)])
            fds.append(rec['mean_fd'])
            rec['has_features'] = True
        else:
            rec['has_features'] = False
        rows.append(rec)
    df = pd.DataFrame(rows)
    csv_out = os.path.join(qc_dir(), 'cohort_qc.csv')
    df.to_csv(csv_out, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.45, wspace=0.25)

    # Mean FD per subject, ranked, with the exclusion line.
    ax = axes[0, 0]
    d = df.dropna(subset=['mean_fd']).sort_values('mean_fd')
    cols = [FLAG if v > FD_MEAN_EXCLUDE else ACCENT for v in d['mean_fd']]
    ax.bar(range(len(d)), d['mean_fd'], color=cols)
    ax.axhline(FD_MEAN_EXCLUDE, ls='--', lw=1.0, color=FLAG)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d['s_id'], rotation=90, fontsize=6)
    ax.set_ylabel('mean FD (mm)')
    n_ex = int((d['mean_fd'] > FD_MEAN_EXCLUDE).sum())
    ax.set_title(f'Mean framewise displacement — {n_ex} subject(s) over '
                 f'{FD_MEAN_EXCLUDE} mm', loc='left', color=INK)

    # % of volumes over the spike threshold.
    ax = axes[0, 1]
    d2 = df.dropna(subset=['pct_over']).sort_values('pct_over')
    cols = [FLAG if v > FD_PCT_EXCLUDE else ACCENT for v in d2['pct_over']]
    ax.bar(range(len(d2)), d2['pct_over'], color=cols)
    ax.axhline(FD_PCT_EXCLUDE, ls='--', lw=1.0, color=FLAG)
    ax.set_xticks(range(len(d2))); ax.set_xticklabels(d2['s_id'], rotation=90, fontsize=6)
    ax.set_ylabel(f'% volumes FD > {FD_THRESH} mm')
    ax.set_title('Volume-level motion burden', loc='left', color=INK)

    # QC-FC: is residual motion still driving connectivity?
    ax = axes[1, 0]
    if len(fcs) > 5:
        F = np.vstack(fcs)
        fdv = np.asarray(fds, dtype=float)
        ok = np.isfinite(fdv)
        r = np.array([np.corrcoef(F[ok, j], fdv[ok])[0, 1] for j in range(F.shape[1])])
        ax.hist(r, bins=60, color=ACCENT)
        ax.axvline(0, lw=1.0, color=INK, alpha=0.5)
        ax.axvline(np.median(r), ls='--', lw=1.2, color=FLAG)
        ax.set_xlabel('corr(edge FC, mean FD)'); ax.set_ylabel('edges')
        ax.set_title(f'QC-FC — median r = {np.median(r):+.3f}\n'
                     'a distribution shifted off zero means motion survived cleaning',
                     loc='left', color=INK)
    else:
        ax.text(0.5, 0.5, 'needs >5 subjects with features', ha='center',
                transform=ax.transAxes); ax.set_axis_off()

    # Mean FC matrix vs psilodep1 -- the transfer sanity check.
    ax = axes[1, 1]
    ref_edges = None
    try:
        ref_files = sorted(glob.glob(os.path.join(project_root(), 'data', 'raw',
                                                  'psilodep1', 'before', atlas,
                                                  'S*', 'edge.csv')))
        ref = np.vstack([pd.read_csv(f)['functional_connectivity'].values for f in ref_files])
        n = int(np.sqrt(ref.shape[1]))
        iu = np.triu_indices(n, k=1)
        ref_edges = np.vstack([row.reshape(n, n)[iu] for row in ref])
        ref_mean = ref.mean(axis=0).reshape(n, n)[iu]
        ket_mean = np.vstack(fcs).mean(axis=0)
        rr = np.corrcoef(ref_mean, ket_mean)[0, 1]
        ax.hexbin(ref_mean, ket_mean, gridsize=60, cmap=SEQ_CMAP, mincnt=1)
        lim = [min(ref_mean.min(), ket_mean.min()), max(ref_mean.max(), ket_mean.max())]
        ax.plot(lim, lim, ls='--', lw=1.0, color=INK, alpha=0.5)
        ax.set_xlabel(f'psilodep1 mean FC (n={len(ref_files)})')
        ax.set_ylabel(f'ds005917 mean FC (n={len(fcs)})')
        ax.set_title(f'Group-mean FC agreement — r = {rr:.3f}\n'
                     'below ~0.7 suggests a preprocessing or space problem',
                     loc='left', color=INK)
    except Exception as e:
        ax.text(0.5, 0.5, f'unavailable:\n{e}', ha='center', fontsize=8,
                transform=ax.transAxes); ax.set_axis_off()

    # Edge-weight distributions. A shift here is the covariate shift that breaks
    # transfer, and is the signature of a mismatched confound strategy.
    ax = axes[2, 0]
    if ref_edges is not None and fcs:
        ket_edges = np.vstack(fcs)
        for edges, lab, c in [(ref_edges, f'psilodep1 (n={len(ref_edges)})', '#9aa5ab'),
                              (ket_edges, f'ds005917 (n={len(ket_edges)})', ACCENT)]:
            ax.hist(edges.ravel(), bins=120, density=True, histtype='step',
                    lw=1.8, label=lab, color=c)
        ax.axvline(0, lw=1.0, color=INK, alpha=0.4)
        ax.set_xlabel('edge weight (r)'); ax.set_ylabel('density')
        ax.set_title('Pooled edge-weight distribution', loc='left', color=INK)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, 'needs both cohorts', ha='center',
                transform=ax.transAxes); ax.set_axis_off()

    # Per-subject summary stats -- separates a genuine cohort shift from a few outliers.
    ax = axes[2, 1]
    stats_rows = []
    if ref_edges is not None and fcs:
        ket_edges = np.vstack(fcs)
        for edges, lab, c in [(ref_edges, 'psilodep1', '#9aa5ab'),
                              (ket_edges, 'ds005917', ACCENT)]:
            m = edges.mean(axis=1)
            fneg = (edges < 0).mean(axis=1)
            ax.scatter(m, fneg, s=28, alpha=0.75, color=c, label=lab,
                       edgecolor='white', linewidth=0.8)
            stats_rows.append({'cohort': lab, 'n': len(edges),
                               'mean': m.mean(), 'sd': edges.std(axis=1).mean(),
                               'frac_neg': fneg.mean()})
        ax.set_xlabel('subject mean edge weight'); ax.set_ylabel('fraction of negative edges')
        ax.set_title('Per-subject FC summary', loc='left', color=INK)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, 'needs both cohorts', ha='center',
                transform=ax.transAxes); ax.set_axis_off()

    out = os.path.join(qc_dir(), f'qc_cohort_{atlas}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    print(f'Saved {csv_out}')

    if stats_rows:
        print('\nFC edge-weight comparison (subject-level means):')
        print(pd.DataFrame(stats_rows).to_string(index=False,
                                                 float_format=lambda v: f'{v:.4f}'))

    print('\nSubjects with no features produced:')
    miss = df[~df['has_features']]['s_id'].tolist()
    print('  ' + (', '.join(miss) if miss else 'none'))
    return df


# ------------------------------------------------------------------ cli --

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['bids', 'grid', 'subject', 'cohort'])
    p.add_argument('--s-id', default=None, help='e.g. S01')
    p.add_argument('--atlas', default='schaefer100')
    p.add_argument('--receptor-set', default='Believeau-5')
    args = p.parse_args()

    if args.stage == 'bids':
        sys.exit(0 if check_bids() else 1)
    elif args.stage == 'grid':
        ok = check_grids(args.s_id, args.atlas)
        sys.exit(0 if ok else 1)
    elif args.stage == 'subject':
        if not args.s_id:
            p.error('--s-id is required for the subject stage')
        qc_subject(args.s_id, args.atlas, args.receptor_set)
    else:
        qc_cohort(args.atlas)


if __name__ == '__main__':
    main()
