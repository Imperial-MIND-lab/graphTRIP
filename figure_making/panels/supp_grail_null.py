"""
Supplementary: the null-model GRAIL negative control.

This panel runs GRAIL on the permutation-null weights, trained on shuffled labels
(scripts/permutation_null.py), and compares the reported alignments against the
resulting distribution.

Author: Hanna M. Tolle
Date: 2026-08-29
License: BSD 3-Clause
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from utils.plotting import NEUTRAL, NEUTRAL2, PSILO

from figure_making.paths import output_dir, MissingInput
from figure_making.registry import register
from figure_making.panels.supp_permutation_null import perm_dirs
from scripts.grail_validation import load_cohort, load_alignments, identified_biomarkers


GRAIL_DIR = ('graphtrip', 'grail')            # the reported alignments
NULL_DIR = ('graphtrip', 'permutation_null')  # perm_*/grail/seed_*/mean_alignments.csv
ALIGNMENT_FILE = 'mean_alignments.csv'
IDENTIFIED_THRESH = 0.5   # the threshold behind the reported biomarker set
SEEDS_PER_PERM = 10
NCOLS = 5


# Loading ----------------------------------------------------------------------------

def observed_profile(feat):
    '''
    The reported alignment of each biomarker: averaged over fold models to give one value
    per patient, then averaged over patients.
    '''
    align, seeds = load_alignments(output_dir(*GRAIL_DIR), feat)
    per_patient = np.stack([align[s] for s in seeds]).mean(axis=(0, 1))  # [subject, feature]
    return per_patient.mean(axis=0), len(seeds)


def null_draws(feat):
    '''
    One null draw per perm_seed, collapsed the same way as the observed statistic.
    '''
    base = output_dir(*NULL_DIR)
    rows, n_runs = [], []
    for perm_dir in perm_dirs(base):
        files = sorted(glob.glob(os.path.join(perm_dir, 'grail', 'seed_*', ALIGNMENT_FILE)))
        if not files:
            continue
        runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        per_patient = runs.groupby('subject')[feat].mean()
        rows.append({'perm_seed': int(os.path.basename(perm_dir).split('_')[-1]),
                     **per_patient.mean(axis=0).to_dict()})
        n_runs.append(len(files))

    if not rows:
        raise MissingInput(os.path.join(base, 'perm_*', 'grail'))
    draws = pd.DataFrame(rows).sort_values('perm_seed').reset_index(drop=True)
    return draws, np.array(n_runs)


# Inference --------------------------------------------------------------------------

def rank_stats(observed, null):
    '''
    Two-sided rank p against the null draws:

        p = (1 + #{|null| >= |observed|}) / (1 + n_draws),  floor 1/(1 + n_draws)

    This is the absolute-value form, not the 2*min(p_greater, p_less) that
    utils.statsalg.compute_permutation_stats computes for the spin test; the two differ by
    a factor of two in their floor. The sign is kept in observed and null_mean.
    '''
    n = len(null)
    exceed = int((np.abs(null) >= abs(observed)).sum())
    return {'observed': observed,
            'null_mean': float(np.mean(null)),
            'null_sd': float(np.std(null, ddof=1)),
            'null_min': float(np.min(null)),
            'null_max': float(np.max(null)),
            'n_draws': n,
            'rank_p': (1 + exceed)/(1 + n),
            'rank_p_floor': 1/(1 + n)}


# Panels -----------------------------------------------------------------------------

def _save(fig, out, name):
    save_path = out.fig(name)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def null_histograms(summary, draws, out, name='grail_null_histograms'):
    '''One null distribution per reported biomarker, with the observed value marked.'''
    reported = summary[summary['identified']]
    ncols = min(NCOLS, len(reported))
    nrows = int(np.ceil(len(reported)/ncols))
    fig, axes2d = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.6*nrows),
                               constrained_layout=True, squeeze=False)
    axes = axes2d.ravel()
    for ax in axes[len(reported):]:
        ax.remove()

    for i, (ax, (_, row)) in enumerate(zip(axes, reported.iterrows())):
        null = draws[row['biomarker']].values
        ax.hist(null, bins=20, color=NEUTRAL, edgecolor=NEUTRAL2, linewidth=0.5)
        ax.axvline(0, color=NEUTRAL2, lw=0.8, ls=':')             # zero
        ax.axvline(row['null_mean'], color=NEUTRAL2, lw=1.0, ls='--')  # null mean
        ax.axvline(row['observed'], color=PSILO, lw=1.8)          # observed
        ax.set_title(f"{row['biomarker']}\np = {row['rank_p']:.3f}, q = {row['fdr_q']:.3f}",
                     fontsize=8, pad=6)
        ax.set_xlabel('mean alignment', fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel('null draws', fontsize=8)
        ax.tick_params(labelsize=7)
    _save(fig, out, name)


def null_profile(summary, out, name='grail_null_profile'):
    '''
    The whole alignment profile: observed against the null mean, over every candidate.
    '''
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    other = summary[~summary['identified']]
    ident = summary[summary['identified']]
    ax.axhline(0, color=NEUTRAL2, lw=0.8, ls=':')
    ax.axvline(0, color=NEUTRAL2, lw=0.8, ls=':')
    ax.scatter(other['observed'], other['null_mean'], s=18, color=NEUTRAL,
               edgecolor=NEUTRAL2, linewidth=0.4, label='other')
    ax.scatter(ident['observed'], ident['null_mean'], s=30, color=PSILO,
               edgecolor=NEUTRAL2, linewidth=0.4, label='reported')
    r = stats.pearsonr(summary['observed'], summary['null_mean'])
    ax.set_xlabel('observed mean alignment')
    ax.set_ylabel('null mean alignment')
    ax.set_title(f'r = {r[0]:.3f} over {len(summary)} biomarkers', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, name)
    return r[0]


# Target -----------------------------------------------------------------------------

@register('grail_null', group='supp', subdir='SUPPLEMENTARY/grail_null')
def grail_null(ctx, out):
    '''
    Are the reported GRAIL alignments larger than models trained on shuffled outcomes
    produce by chance?
    '''
    feat, _, _, _ = load_cohort()
    draws, n_runs = null_draws(feat)       # raises MissingInput until the array lands
    observed, n_seeds = observed_profile(feat)

    majority = identified_biomarkers(thresh=IDENTIFIED_THRESH)
    summary = pd.DataFrame([{'biomarker': f,
                             'identified': f in majority,
                             'category': majority.get(f, ''),
                             **rank_stats(observed[i], draws[f].values)}
                            for i, f in enumerate(feat)])

    # FDR over the reported biomarkers only
    summary['fdr_q'] = np.nan
    reported = summary['identified'].values
    summary.loc[reported, 'fdr_q'] = fdrcorrection(
        summary.loc[reported, 'rank_p'], alpha=0.05)[1]

    null_histograms(summary, draws, out)
    profile_r = null_profile(summary, out)

    out.table('grail_null_draws', draws)
    out.table('grail_null_stats', summary)

    # Report ---------------------------------------------------------------------------
    out.log(f'GRAIL null: {len(draws)} permutation draws, '
            f'{n_runs.min()}-{n_runs.max()} training seeds each '
            f'(expected {SEEDS_PER_PERM}); observed from {n_seeds} seeds.')
    if (n_runs != SEEDS_PER_PERM).any():
        incomplete = int((n_runs != SEEDS_PER_PERM).sum())
        out.log(f'WARNING: {incomplete} permutation(s) have an incomplete set of seeds.')
    out.log(f'Null mean over all {len(feat)} biomarkers: '
            f'{summary["null_mean"].mean():+.4f} '
            f'(the null is not assumed to be centred on zero).')
    out.log(f'Observed vs null-mean profile: r = {profile_r:.3f}.')
    n_sig = int((summary.loc[reported, 'fdr_q'] < 0.05).sum())
    out.log(f'{n_sig}/{int(reported.sum())} reported biomarkers exceed their null '
            f'at FDR < 0.05 (rank p floor {summary["rank_p_floor"].iloc[0]:.4f}).')
    out.log()
    cols = ['biomarker', 'category', 'observed', 'null_mean', 'null_sd', 'rank_p', 'fdr_q']
    out.log_df('Reported biomarkers', summary.loc[reported, cols].round(4))
    out.log_df('All candidates', summary[cols].round(4))
