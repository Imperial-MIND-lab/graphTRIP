"""
Supplementary: the biomarker selection validated on patients it never saw.

Every panel comes from the nested cross-validation (scripts/nested_cv.py and
scripts/nested_cv_validation.py), in which one outer fold of patients is held out of
both training and biomarker selection.

Four panels: whether the inner models still predict at the reduced training size, whether
the selected biomarkers predict the held-out patients against random candidate sets of the
same size, whether their learned direction holds out, and how stable the selection is
across outer folds.

Author: Hanna M. Tolle
Date: 2026-09-12
License: BSD 3-Clause
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

from utils.plotting import NEUTRAL, NEUTRAL2

from figure_making.common import fmt_p
from figure_making.paths import output_dir, require, MissingInput
from figure_making.registry import register


# The same red as the other supplementary panels, for the observed and identified values.
DARK_RED = '#AA0000'

ANALYSIS_DIR = ('graphtrip', 'nested_cv', 'analysis')
PUBLISHED_GRAIL = ('graphtrip', 'grail')


def _read(name):
    '''One analysis table, or a skip if the nested runs have not been made.'''
    return pd.read_csv(require(os.path.join(output_dir(*ANALYSIS_DIR), name)))


@register('nested_cv', group='supp', subdir='SUPPLEMENTARY/nested_cv')
def nested_cv(ctx, out):

    # a. Do the inner models still predict at the reduced training size? ------------------
    performance = _read('nested_cv_performance.csv')
    _inner_performance(performance, out)

    out.log(f'Nested cross-validation: {performance["outer_fold"].nunique()} outer folds x '
            f'{performance["seed"].nunique()} seeds, {len(performance)} inner fold models.')
    out.log(f'  inner test rho: median {performance["rho"].median():+.4f}, '
            f'{(performance["rho"] > 0).sum()}/{len(performance)} above zero')
    out.log()

    # b. Do the selected biomarkers predict the held-out patients? ------------------------
    try:
        ridge = _read('nested_cv_ridge.csv')
        predictions = _read('nested_cv_ridge_predictions.csv')
        null = np.load(require(os.path.join(output_dir(*ANALYSIS_DIR),
                                            'nested_cv_ridge_null.npy')))
        _ridge_panels(ridge, predictions, null, out)
        out.log_df('Ridge on the held-out patients', ridge.round(4))
        out.log()
    except (MissingInput, FileNotFoundError) as error:
        out.log(f'No held-out prediction panel: {error}')
        out.log()

    # c. Does the learned direction hold in the held-out patients? ------------------------
    try:
        per_fold = _read('nested_cv_directional_per_fold.csv')
        summary = _read('nested_cv_directional_summary.csv')
        _directional_panel(per_fold, summary, out)
        out.log_df('Held-out direction, one value per outer fold', summary.round(4))
        out.log()
    except (MissingInput, FileNotFoundError) as error:
        out.log(f'No held-out direction panel: {error}')
        out.log()

    # d. How stable is the selection across outer folds? ----------------------------------
    try:
        stability = _read('nested_cv_stability.csv')
        selection = _read('nested_cv_selection.csv')
        _stability_panel(stability, selection, out)
        out.table('nested_cv_stability', stability)
        out.log(f'{len(stability)} biomarkers are selected in at least one outer fold; '
                f'{int((stability["frac_outer_folds"] == 1).sum())} in all of them.')
        out.log()
    except (MissingInput, FileNotFoundError) as error:
        out.log(f'No selection stability panel: {error}')
        out.log()


def _published_rho():
    '''The published fold models' test rho, for the training-size comparison.'''
    paths = sorted(glob.glob(os.path.join(output_dir(*PUBLISHED_GRAIL), 'seed_*',
                                          'fold_performances.csv')))
    if not paths:
        return None
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)['rho']


def _inner_performance(performance, out, name='nested_cv_inner_performance'):
    '''
    Compare performances of the nested inner fold models against the main model.
    Nested models were trained on less data.
    '''
    published = _published_rho()
    groups = [('nested\n(~31 training)', performance['rho'].values, DARK_RED)]
    if published is not None:
        groups.insert(0, ('published\n(36 training)', published.values, NEUTRAL2))

    fig, ax = plt.subplots(figsize=(3.6, 4.6), constrained_layout=True)
    for i, (label, values, colour) in enumerate(groups):
        ax.scatter(np.full(len(values), i + 1), values, color=colour, s=22,
                   edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
        ax.hlines(np.median(values), i + 0.78, i + 1.22, color=colour, lw=2, zorder=4)

    ax.axhline(0, color='lightgray', ls='--', lw=1, zorder=0)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([label for label, _, _ in groups])
    ax.set_xlim(0.5, len(groups) + 0.5)
    ax.set_ylabel("fold model's test rho")
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Inner fold performance\n(line: median)', fontsize=9, linespacing=1.35)

    _save(fig, out, name)


def _ridge_panels(ridge, predictions, null, out):
    '''
    The held-out predictions of the selected biomarkers, and the null they are judged by.
    The null is 1000 random candidate sets of the same size.
    '''
    observed = predictions.dropna(subset=['prediction'])
    r, p = pearsonr(observed['label'], observed['prediction'])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)

    axes[0].scatter(observed['label'], observed['prediction'], color=DARK_RED, s=26,
                    edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
    slope, intercept = np.polyfit(observed['label'], observed['prediction'], 1)
    line = np.array([observed['label'].min(), observed['label'].max()])
    axes[0].plot(line, slope*line + intercept, color=NEUTRAL2, lw=1.5, zorder=2)
    axes[0].set_xlabel('true outcome')
    axes[0].set_ylabel('predicted outcome')
    axes[0].set_title(f'Held-out patients (n = {len(observed)})\n'
                      f'r = {r:.4f}, p = {fmt_p(p)}', fontsize=9, linespacing=1.35)

    row = ridge[ridge['set'].str.startswith('selected')].iloc[0]
    if len(null):
        axes[1].hist(null, bins=30, color=NEUTRAL, edgecolor=NEUTRAL2, linewidth=0.5,
                     label=f'random sets of the same size (n = {len(null)})')
        axes[1].axvline(row['null_mean'], color=NEUTRAL2, lw=1.0, ls='--', label='null mean')
    axes[1].axvline(row['observed'], color=DARK_RED, lw=1.8, label='selected biomarkers')
    axes[1].set_xlabel('pooled held-out r')
    axes[1].set_ylabel('count')
    axes[1].set_title(f"z = {row['z_score']:+.2f}, p = {fmt_p(row['p_value'])}",
                      fontsize=9)
    axes[1].legend(loc='upper left', fontsize=6.5, frameon=False, handlelength=1.2)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

    _save(fig, out, 'nested_cv_ridge')


def _directional_panel(per_fold, summary, out, name='nested_cv_direction'):
    '''
    The selected biomarkers against the other candidates, one point per outer fold.

    Seeds are averaged within an outer fold first: they share its held-out patients, so
    they are not independent replicates.
    '''
    statistics = [s for s in ('signed_r', 'pearson')
                  if f'identified_{s}' in per_fold.columns]
    fig, axes = plt.subplots(1, len(statistics), figsize=(3.2*len(statistics), 4.8),
                             constrained_layout=True, squeeze=False)

    for ax, statistic in zip(axes[0], statistics):
        columns = [f'identified_{statistic}', f'other_{statistic}']
        values = per_fold[columns].values
        low, high = np.nanmin(values), np.nanmax(values)
        pad = 0.1*(high - low) if high > low else 0.1
        x = np.array([1, 2])

        ax.plot(x, values.T, color=NEUTRAL, lw=0.6, zorder=1)
        for xi, column, colour in zip(x, columns, (DARK_RED, NEUTRAL2)):
            ax.scatter(np.full(len(per_fold), xi), per_fold[column], color=colour, s=28,
                       edgecolor='white', linewidth=0.4, zorder=3)
            ax.hlines(per_fold[column].mean(), xi - 0.22, xi + 0.22, color=colour, lw=2,
                      zorder=2)

        pval = summary.loc[summary['statistic'] == statistic,
                           'p_identified_gt_other'].iloc[0]
        if not np.isnan(pval) and pval < 0.05:
            ax.text(1.5, high + 1.0*pad, '*', color=DARK_RED, fontsize=22,
                    ha='center', va='center', fontweight='bold')

        ax.axhline(0, color='lightgray', ls='--', lw=1, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(['selected', 'other'])
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(low - pad, high + 1.9*pad)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel(f'{statistic} over the held-out patients')
        # The test needs at least 5 outer folds; until then the panel shows the values only
        label = f'p = {fmt_p(pval)}' if not np.isnan(pval) else 'too few outer folds to test'
        ax.set_title(f'{statistic}\npaired Wilcoxon over {len(per_fold)} outer folds\n'
                     f'{label}', fontsize=9, linespacing=1.35)

    _save(fig, out, name)


def _stability_panel(stability, selection, out, name='nested_cv_stability'):
    '''
    How often each biomarker is reselected when the outer fold changes.

    A selection that changes completely between outer folds cannot be expected to
    generalise, so this panel is read alongside the two statistics.
    '''
    n_folds = selection['outer_fold'].nunique()
    top = stability.head(25).iloc[::-1]
    colours = [DARK_RED if published else NEUTRAL2
               for published in top['in_published_set']]

    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.22*len(top) + 1.2)),
                           constrained_layout=True)
    ax.barh(range(len(top)), top['n_outer_folds'], color=colours, edgecolor=NEUTRAL2,
            linewidth=0.4, alpha=0.9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['biomarker'], fontsize=7)
    ax.set_xlabel(f'outer folds selecting it (of {n_folds})')
    ax.set_xlim(0, n_folds)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title('Selection stability\n(red: in the published graphTRIP set)',
                 fontsize=9, linespacing=1.35)

    _save(fig, out, name)


def _save(fig, out, name):
    save_path = out.fig(name)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
