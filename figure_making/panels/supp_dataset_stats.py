"""
Supplementary: descriptive statistics of the psilodep2 and psilodep1 datasets.

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

from datasets import get_default_prefilter
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

# Baseline severity of the psilocybin patients of both studies: the psilodep2
# escitalopram arm is dropped so that both groups received psilocybin.
BASELINE_GROUPS = [
    {'study': 'psilodep2', 'label': 'psilodep2\n(psilocybin arm)', 'condition': 1},
    {'study': 'psilodep1', 'label': 'psilodep1\n(all psilocybin)', 'condition': None},
]
BASELINE_MEASURES = [('QIDS', 'QIDS_Before'), ('BDI', 'BDI_Before')]

JITTER_SD = 0.06
MARKER_SIZE = 28
SIG_ALPHA = 0.05


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


def _fmt_p(pval, decimals=3):
    '''Formats a p-value for a panel title, without rounding small values to zero.'''
    floor = 10 ** -decimals
    return f'p<{floor:.{decimals}f}' if pval < floor else f'p={pval:.{decimals}f}'


def _load_study_annotations(study):
    '''
    Loads the annotations of the patients a study contributes to the models.

    The filter is datasets.get_default_prefilter(), the same one BrainGraphDataset
    applies, so that every panel here describes exactly the patients that are trained
    and tested on: all non-excluded psilodep2 patients, and the non-excluded psilodep1
    patients that have a pre-treatment scan.
    '''
    return load_annotations(study, filter=get_default_prefilter(study))


def _baseline_samples(group, column):
    '''Returns the baseline scores of one study group, dropping patients without a score.'''
    df = _load_study_annotations(group['study'])
    if group['condition'] is not None:
        df = df[df['Condition_numeric'] == group['condition']]
    return df[column].dropna().to_numpy(dtype=float)


def _compare_studies(measure, column, groups, samples):
    '''
    Welch's t-test and Cohen's d between the baseline scores of two independent groups.

    Welch rather than Student, because the two studies differ in sample size and the
    equality of their variances is not established.
    '''
    a, b = samples
    t, pval = stats.ttest_ind(a, b, equal_var=False)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    return {'measure': measure, 'column': column,
            'group1': groups[0]['study'], 'n1': len(a),
            'mean1': a.mean(), 'std1': a.std(ddof=1),
            'group2': groups[1]['study'], 'n2': len(b),
            'mean2': b.mean(), 'std2': b.std(ddof=1),
            't': t, 'p': pval, 'cohen_d': (a.mean() - b.mean()) / pooled_sd}


def _group_panel(ax, samples, labels, ylabel, rng):
    '''Boxplots of the baseline scores of each group, with the individual patients on top.'''
    scores = pd.DataFrame({'score': np.concatenate(samples),
                           'group': np.repeat(labels, [len(s) for s in samples])})
    sns.boxplot(data=scores, x='group', y='score', order=labels,
                color=BOX_COLOR, width=0.5, showfliers=False, ax=ax, zorder=1)

    # Every patient shown here received psilocybin, hence a single marker style.
    _, _, marker, color = CONDITION_STYLES[0]
    for position, values in enumerate(samples):
        x_jitter = position + rng.normal(0, JITTER_SD, size=len(values))
        ax.scatter(x_jitter, values, marker=marker, color=color, edgecolor=color,
                   s=MARKER_SIZE, alpha=ALPHA_SCATTER, zorder=2)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel)


def _add_significance_marker(ax, pval, positions=(0., 1.)):
    '''Marks a significant group difference with an asterisk centred above the boxes.'''
    if pval >= SIG_ALPHA:
        return
    ax.text(np.mean(positions), 0.97, '*', transform=ax.get_xaxis_transform(),
            color='red', fontsize=16, fontweight='bold', ha='center', va='top')


def _regression_line(ax, x, y, color):
    '''Draws the least-squares fit of y on x across the observed range of x.'''
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, color=color, alpha=0.6, zorder=1)


def _add_study_separator(fig, left_ax, right_ax):
    '''Draws a dashed vertical line between the panels of the two studies.'''
    x = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2
    fig.add_artist(Line2D([x, x], [0.02, 0.98], transform=fig.transFigure,
                          color=NEUTRAL2, linestyle='--', linewidth=1))


@register('dataset_stats', group='supp', subdir='SUPPLEMENTARY/dataset_stats')
def dataset_stats(ctx, out):
    rng = np.random.default_rng(ctx.cfg.seed)
    annotations = {spec['study']: _load_study_annotations(spec['study'])
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
    fig, axes = plt.subplots(1, len(BASELINE_MEASURES), figsize=(3 * len(BASELINE_MEASURES), 3.5))

    rows = []
    for ax, (measure, column) in zip(axes, BASELINE_MEASURES):
        samples = [_baseline_samples(group, column) for group in BASELINE_GROUPS]
        labels = [f"{group['label']}\nn={len(sample)}"
                  for group, sample in zip(BASELINE_GROUPS, samples)]
        _group_panel(ax, samples, labels, f'{measure} Score', rng)

        test = _compare_studies(measure, column, BASELINE_GROUPS, samples)
        _add_significance_marker(ax, test['p'])
        ax.set_title(f"{measure}: t={test['t']:.2f}, {_fmt_p(test['p'])}, "
                     f"d={test['cohen_d']:.2f}")
        rows.append(test)

    plt.tight_layout()
    save_path = out.fig('qids_bdi_baseline_by_study')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    baseline_tests = pd.DataFrame(rows)
    out.table('baseline_by_study_tests', baseline_tests)
    out.log_df('Baseline severity, psilodep2 psilocybin arm versus psilodep1 '
               "(Welch's t-tests, uncorrected)", baseline_tests)

    # Baseline versus post-treatment QIDS -------------------------------------------------
    fig, axes = plt.subplots(1, len(STUDIES), figsize=(4.5 * len(STUDIES), 4))

    rows = []
    for ax, spec in zip(axes, STUDIES):
        measure, before_col, after_col = spec['measures'][0]
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
                title_lines.append(f'{label} r={r:.2f}, {_fmt_p(pval)}')
                rows.append({'study': spec['study'], 'condition': label,
                             'x': before_col, 'y': after_col,
                             'n': int(mask.sum()), 'r': r, 'p': pval})
            ax.set_title('\n'.join([f"{spec['study']}: {title_lines[0]}"] + title_lines[1:]))
        else:
            r, pval = stats.pearsonr(before, after)
            _regression_line(ax, before, after, 'darkred')
            ax.set_title(f"{spec['study']}: r={r:.2f}, {_fmt_p(pval)}")
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
