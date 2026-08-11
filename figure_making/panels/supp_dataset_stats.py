"""
Supplementary: descriptive statistics of the psilodep2 and psilodep1 datasets.

Shows pre- versus post-treatment QIDS and BDI scores of both studies, the association
between baseline and post-treatment QIDS, compares the two treatment arms of psilodep2
on baseline, final and change scores, and tests whether SSRI discontinuation was
balanced across arms.

Author: Hanna M. Tolle
Date: 2026-08-11
License: BSD 3-Clause
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from utils.annotations import load_annotations
from utils.plotting import ALPHA_SCATTER, BOX_COLOR, ESCIT, NEUTRAL2, PSILO

from figure_making.registry import register


# Post-treatment timepoint per study: psilodep2 patients are scored at the final
# integration session, psilodep1 patients one week after the second dose. These are the
# prediction targets of experiments/configs/{graphtrip,psilodep1_finetuning}.json.
STUDIES = [
    {'study': 'psilodep2', 'post_label': 'Final\nIntegration',
     'measures': [('QIDS', 'QIDS_Before', 'QIDS_Final_Integration'),
                  ('BDI', 'BDI_Before', 'BDI_Final_Integration')]},
    {'study': 'psilodep1', 'post_label': '1 week\npost-dose',
     'measures': [('QIDS', 'QIDS_Before', 'QIDS_1week'),
                  ('BDI', 'BDI_Before', 'BDI_1week')]},
]

# Arm comparisons; only psilodep1 has a single (psilocybin-only) arm.
ARM_COMPARISONS = ['QIDS_Before', 'BDI_Before',
                   'QIDS_Final_Integration', 'BDI_Final_Integration',
                   'Delta_QIDS', 'Delta_BDI']

# Condition_numeric is coded identically in both studies: 1 = psilocybin, -1 = escitalopram.
CONDITION_STYLES = [(1, 'Psilocybin', 'd', PSILO),
                    (-1, 'Escitalopram', 'o', ESCIT)]

JITTER_SD = 0.06
MARKER_SIZE = 28


def _compare_arms(df, column):
    '''Two-sample t-test and Cohen's d between the escitalopram and psilocybin arms.'''
    e = df[df['Condition'] == 'E'][column]
    p = df[df['Condition'] == 'P'][column]
    t, pval = stats.ttest_ind(e, p)
    d = (e.mean() - p.mean()) / np.sqrt((e.var() + p.var()) / 2)
    return {'measure': column,
            'escitalopram_mean': e.mean(), 'escitalopram_std': e.std(),
            'psilocybin_mean': p.mean(), 'psilocybin_std': p.std(),
            't': t, 'p': pval, 'cohen_d': d}


def _condition_legend(ax, conditions):
    '''Adds a marker legend for the treatment arms present in conditions.'''
    handles = [Line2D([], [], linestyle='none', marker=marker, color=color, label=label)
               for value, label, marker, color in CONDITION_STYLES if (conditions == value).any()]
    ax.legend(handles=handles, frameon=False, fontsize='small', loc='best')


def _scatter_by_condition(ax, x, y, conditions, **kwargs):
    '''Scatters x against y with one marker style and colour per treatment arm.'''
    for value, _, marker, color in CONDITION_STYLES:
        mask = conditions == value
        if mask.any():
            ax.scatter(x[mask], y[mask], marker=marker, color=color, edgecolor=color,
                       s=MARKER_SIZE, alpha=ALPHA_SCATTER, **kwargs)


def _paired_data(df, before_col, after_col):
    '''Returns the before scores, after scores and conditions of complete pairs.'''
    sub = df.dropna(subset=[before_col, after_col])
    return (sub[before_col].to_numpy(dtype=float),
            sub[after_col].to_numpy(dtype=float),
            sub['Condition_numeric'].to_numpy(dtype=float))


def _paired_panel(ax, before, after, conditions, post_label, ylabel, rng):
    '''
    Boxplots of the before and after scores with the individual patients on top.

    Each patient contributes one jittered marker per timepoint, styled by treatment arm,
    and a thin line connecting the two. The connecting lines use the same jittered
    x-positions as the markers, so they meet the dots they belong to.
    '''
    positions = np.array([0., 1.])
    scores = pd.DataFrame({'score': np.concatenate([before, after]),
                           'timepoint': ['before'] * len(before) + ['after'] * len(after)})
    sns.boxplot(data=scores, x='timepoint', y='score', order=['before', 'after'],
                color=BOX_COLOR, width=0.5, showfliers=False, ax=ax, zorder=1)

    x_jitter = positions[:, None] + rng.normal(0, JITTER_SD, size=(2, len(before)))
    for i in range(len(before)):
        ax.plot(x_jitter[:, i], [before[i], after[i]],
                color=NEUTRAL2, linewidth=0.5, alpha=0.5, zorder=2)

    _scatter_by_condition(ax, x_jitter[0], before, conditions, zorder=3)
    _scatter_by_condition(ax, x_jitter[1], after, conditions, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Before', post_label])
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)


def _paired_test(study, measure, before, after):
    '''Paired t-test between the before and after scores of one measure.'''
    t, pval = stats.ttest_rel(before, after)
    diff = after - before
    return {'study': study, 'measure': measure, 'n': len(before),
            'before_mean': before.mean(), 'before_std': before.std(ddof=1),
            'after_mean': after.mean(), 'after_std': after.std(ddof=1),
            'delta_mean': diff.mean(), 'delta_std': diff.std(ddof=1),
            't': t, 'p': pval, 'cohen_dz': diff.mean() / diff.std(ddof=1)}


def _add_study_separator(fig, left_ax, right_ax):
    '''Draws a dashed vertical line between the panels of the two studies.'''
    x = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2
    fig.add_artist(Line2D([x, x], [0.02, 0.98], transform=fig.transFigure,
                          color=NEUTRAL2, linestyle='--', linewidth=1))


@register('dataset_stats', group='supp', subdir='SUPPLEMENTARY/dataset_stats')
def dataset_stats(ctx, out):
    rng = np.random.default_rng(ctx.cfg.seed)
    annotations = {spec['study']: load_annotations(spec['study'], filter={'Exclusion': 0})
                   for spec in STUDIES}

    # Sample composition ----------------------------------------------------------------
    for study, df in annotations.items():
        out.log(f'=== {study} (n={len(df)}) ===')
        for value, label, _, _ in CONDITION_STYLES:
            arm = df[df['Condition_numeric'] == value]
            if len(arm) == 0:
                continue
            n_females = len(arm[arm['Gender'] == 'F'])
            out.log(f'Number of patients in {label.lower()} condition: '
                    f'{len(arm)} ({n_females} female)')
        out.log()

    # Pre- versus post-treatment scores --------------------------------------------------
    panels = [(spec, measure) for spec in STUDIES for measure in spec['measures']]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.5))

    rows = []
    for ax, (spec, (measure, before_col, after_col)) in zip(axes, panels):
        df = annotations[spec['study']]
        before, after, conditions = _paired_data(df, before_col, after_col)
        _paired_panel(ax, before, after, conditions, spec['post_label'],
                      f'{measure} Score', rng)
        ax.set_title(f"{spec['study']}: {measure}")
        rows.append(_paired_test(spec['study'], measure, before, after))

    # The legend goes on the first panel that shows both arms.
    _condition_legend(axes[0], annotations[STUDIES[0]['study']]['Condition_numeric'].to_numpy())

    plt.tight_layout()
    _add_study_separator(fig, axes[len(STUDIES[0]['measures']) - 1],
                         axes[len(STUDIES[0]['measures'])])
    save_path = out.fig('qids_bdi_pre_vs_post')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    paired_tests = pd.DataFrame(rows)
    out.table('pre_vs_post_tests', paired_tests)
    out.log_df('Pre- versus post-treatment scores (paired t-tests)', paired_tests)

    # Baseline versus post-treatment QIDS -------------------------------------------------
    fig, axes = plt.subplots(1, len(STUDIES), figsize=(4.5 * len(STUDIES), 4))

    rows = []
    for ax, spec in zip(axes, STUDIES):
        measure, before_col, after_col = spec['measures'][0]
        before, after, conditions = _paired_data(annotations[spec['study']],
                                                 before_col, after_col)
        r, pval = stats.pearsonr(before, after)

        slope, intercept = np.polyfit(before, after, 1)
        x_line = np.array([before.min(), before.max()])
        ax.plot(x_line, slope * x_line + intercept, color='darkred', alpha=0.6, zorder=1)
        _scatter_by_condition(ax, before, after, conditions, zorder=2)

        ax.set_xlabel(before_col)
        ax.set_ylabel(after_col)
        ax.set_title(f"{spec['study']}: r={r:.2f}, p={pval:.3f}")
        rows.append({'study': spec['study'], 'x': before_col, 'y': after_col,
                     'n': len(before), 'r': r, 'p': pval})

    _condition_legend(axes[0], annotations[STUDIES[0]['study']]['Condition_numeric'].to_numpy())

    plt.tight_layout()
    save_path = out.fig('qids_before_vs_post')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    correlations = pd.DataFrame(rows)
    out.table('qids_before_vs_post_correlations', correlations)
    out.log_df('Baseline versus post-treatment QIDS', correlations)

    # QIDS and BDI comparisons between the psilodep2 treatment arms ----------------------
    df = annotations['psilodep2']
    comparisons = pd.DataFrame([_compare_arms(df, column) for column in ARM_COMPARISONS])
    out.table('condition_comparisons', comparisons)
    out.log_df('psilodep2 treatment arm comparisons', comparisons)

    # SSRI discontinuation balance -------------------------------------------------------
    contingency = pd.crosstab(df['Condition'], df['Stop_SSRI'])
    chi2, p_value = stats.chi2_contingency(contingency)[:2]

    report = ['Contingency table of Stop_SSRI vs Condition:',
              contingency.to_string(),
              '',
              'Chi-square test results:',
              f'chi2 = {chi2:.3f}',
              f'p-value = {p_value:.3f}']
    out.text('ssri_chi2', '\n'.join(report))
    for line in report:
        out.log(line)
