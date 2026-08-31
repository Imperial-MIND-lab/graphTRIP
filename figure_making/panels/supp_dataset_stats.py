"""
Supplementary: descriptive statistics of the psilodep2 and psilodep1 datasets.

Author: Hanna M. Tolle
Date: 2026-08-11
License: BSD 3-Clause
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from utils.plotting import ALPHA_SCATTER, BOX_COLOR, ESCIT, NEUTRAL2, PSILO

from figure_making.common import (
    JITTER_SD, MARKER_SIZE, baseline_severity_panels, fmt_p_floor, study_annotations)
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

# Baseline vs post-treatment QIDS scatters. The first two panels are each study's
# prediction target; the third is the later psilodep1 timepoint in the same 16 patients.
QIDS_SCATTER_PANELS = [
    {'study': 'psilodep2', 'before_col': 'QIDS_Before', 'after_col': 'QIDS_Final_Integration'},
    {'study': 'psilodep1', 'before_col': 'QIDS_Before', 'after_col': 'QIDS_1week'},
    {'study': 'psilodep1', 'before_col': 'QIDS_Before', 'after_col': 'QIDS_3months'},
]

# Condition_numeric is coded identically in both studies: 1 = psilocybin, -1 = escitalopram.
CONDITION_STYLES = [(1, 'Psilocybin', 'd', PSILO),
                    (-1, 'Escitalopram', 'o', ESCIT)]



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


def _regression_line(ax, x, y, color):
    '''Draws the least-squares fit of y on x across the observed range of x.'''
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, color=color, alpha=0.6, zorder=1)


def _arm_difference_tests(spec, before, after, conditions):
    '''
    Tests whether the baseline-outcome association differs between the treatment arms.
    (Fisher z contrast, and interaction term of pooled regression.
    '''
    psilo, escit = CONDITION_STYLES[0][0], CONDITION_STYLES[1][0]
    r_p, _ = stats.pearsonr(before[conditions == psilo], after[conditions == psilo])
    r_e, _ = stats.pearsonr(before[conditions == escit], after[conditions == escit])
    n_p = int((conditions == psilo).sum())
    n_e = int((conditions == escit).sum())

    z = (np.arctanh(r_e) - np.arctanh(r_p)) / np.sqrt(1 / (n_p - 3) + 1 / (n_e - 3))
    rows = [{'study': spec['study'],
             'test': 'Fisher z (escitalopram vs psilocybin r)',
             'estimate': np.nan, 'statistic': z, 'p': 2 * stats.norm.sf(abs(z)),
             'df': np.nan, 'n': n_p + n_e}]

    design = sm.add_constant(np.column_stack([before, conditions, before * conditions]),
                             has_constant='add')
    fit = sm.OLS(after, design).fit()
    rows.append({'study': spec['study'],
                 'test': f"OLS interaction {spec['before_col']} x Condition",
                 'estimate': fit.params[3], 'statistic': fit.tvalues[3],
                 'p': fit.pvalues[3], 'df': fit.df_resid, 'n': len(after)})
    return rows


def _add_study_separator(fig, left_ax, right_ax):
    '''Draws a dashed vertical line between the panels of the two studies.'''
    x = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2
    fig.add_artist(Line2D([x, x], [0.02, 0.98], transform=fig.transFigure,
                          color=NEUTRAL2, linestyle='--', linewidth=1))


@register('dataset_stats', group='supp', subdir='SUPPLEMENTARY/dataset_stats')
def dataset_stats(ctx, out):
    rng = np.random.default_rng(ctx.cfg.seed)
    annotations = {spec['study']: study_annotations(spec['study'])
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

    # Baseline severity of the psilocybin patients of both studies ------------------------
    baseline_tests = baseline_severity_panels(out, 'qids_bdi_baseline_by_study', rng)
    out.table('baseline_by_study_tests', baseline_tests)
    out.log_df('Baseline severity, psilodep2 psilocybin arm versus psilodep1 '
               "(Welch's t-tests, uncorrected)", baseline_tests)

    # Baseline versus post-treatment QIDS -------------------------------------------------
    fig, axes = plt.subplots(1, len(QIDS_SCATTER_PANELS),
                             figsize=(4.5 * len(QIDS_SCATTER_PANELS), 4))

    rows = []
    arm_test_rows = []
    for ax, spec in zip(axes, QIDS_SCATTER_PANELS):
        before_col, after_col = spec['before_col'], spec['after_col']
        before, after, conditions = _paired_data(annotations[spec['study']],
                                                 before_col, after_col)

        # The two psilodep2 arms are fitted separately, because a pooled line would mix
        # two different baseline-outcome associations. psilodep1 is psilocybin-only and
        # keeps a single fit over all its patients.
        arms = [(value, label, color) for value, label, _, color in CONDITION_STYLES
                if (conditions == value).any()]
        if len(arms) > 1:
            title_lines = []
            for value, label, color in arms:
                mask = conditions == value
                r, pval = stats.pearsonr(before[mask], after[mask])
                _regression_line(ax, before[mask], after[mask], color)
                title_lines.append(f'{label} r={r:.2f}, {fmt_p_floor(pval)}')
                rows.append({'study': spec['study'], 'condition': label,
                             'x': before_col, 'y': after_col,
                             'n': int(mask.sum()), 'r': r, 'p': pval})

            # Also save pooled baseline vs post-treatment QIDS correlation
            r_pooled, p_pooled = stats.pearsonr(before, after)
            rows.append({'study': spec['study'], 'condition': 'all',
                         'x': before_col, 'y': after_col,
                         'n': len(before), 'r': r_pooled, 'p': p_pooled})
            title_lines.append(f'pooled r={r_pooled:.2f}, {fmt_p_floor(p_pooled)}')

            arm_test_rows.extend(_arm_difference_tests(spec, before, after, conditions))
            ax.set_title('\n'.join([f"{spec['study']}: {title_lines[0]}"] + title_lines[1:]))
        else:
            r, pval = stats.pearsonr(before, after)
            _regression_line(ax, before, after, 'darkred')
            ax.set_title(f"{spec['study']}: r={r:.2f}, {fmt_p_floor(pval)}")
            rows.append({'study': spec['study'], 'condition': 'all',
                         'x': before_col, 'y': after_col,
                         'n': len(before), 'r': r, 'p': pval})

        _scatter_by_condition(ax, before, after, conditions, zorder=2)
        ax.set_xlabel(before_col)
        ax.set_ylabel(after_col)

    _condition_legend(axes[0], annotations[STUDIES[0]['study']]['Condition_numeric'].to_numpy())

    plt.tight_layout()
    save_path = out.fig('qids_before_vs_post')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    correlations = pd.DataFrame(rows)
    out.table('qids_before_vs_post_correlations', correlations)
    out.log_df('Baseline versus post-treatment QIDS', correlations)

    arm_tests = pd.DataFrame(arm_test_rows)
    out.table('baseline_outcome_arm_tests', arm_tests)
    out.log_df('Does the baseline-outcome association differ between arms?', arm_tests)

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
