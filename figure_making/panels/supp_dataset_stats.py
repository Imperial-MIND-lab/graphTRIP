"""
Supplementary: descriptive statistics of the primary dataset (psilodep2).

Compares the two treatment arms on baseline, final and change scores for QIDS and BDI,
and tests whether SSRI discontinuation was balanced across arms.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from utils.annotations import load_annotations

from figure_making.registry import register


# (panel name, QIDS column, BDI column, title)
COMPARISONS = [
    ('qids_bdi_before', 'QIDS_Before', 'BDI_Before', 'Before Treatment'),
    ('qids_bdi_final', 'QIDS_Final_Integration', 'BDI_Final_Integration', 'Final Integration'),
    ('qids_bdi_delta', 'Delta_QIDS', 'Delta_BDI', 'change (post - pre)'),
]


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


@register('dataset_stats', group='supp', subdir='SUPPLEMENTARY/dataset_stats')
def dataset_stats(ctx, out):
    df = load_annotations("psilodep2", filter={'Exclusion': 0})

    # Sample composition ----------------------------------------------------------------
    for label, condition in [('escitalopram', 'E'), ('psilocybin', 'P')]:
        n_total = len(df[df['Condition'] == condition])
        n_females = len(df[(df['Condition'] == condition) & (df['Gender'] == 'F')])
        out.log(f'Number of patients in {label} condition: {n_total} ({n_females} female)')
    out.log()

    # QIDS and BDI comparisons between treatment arms -----------------------------------
    rows = []
    for name, qids_col, bdi_col, title in COMPARISONS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

        for ax, column, ylabel in [(ax1, qids_col, 'QIDS Score'), (ax2, bdi_col, 'BDI Score')]:
            result = _compare_arms(df, column)
            rows.append(result)
            sns.boxplot(data=df, x='Condition', y=column, ax=ax)
            ax.set_xlabel('Condition')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{column}\nt={result['t']:.2f}, p={result['p']:.3f}, "
                         f"d={result['cohen_d']:.2f}")

        plt.tight_layout()
        save_path = out.fig(name)
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)

    comparisons = pd.DataFrame(rows)
    out.table('condition_comparisons', comparisons)
    out.log_df('Treatment arm comparisons', comparisons)

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
