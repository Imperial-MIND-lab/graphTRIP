"""
Supplementary: agreement between cross-validation fold models, and what the GRAIL
biomarkers capture.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils.helpers import get_groups
from utils.statsalg import min_significant_r
from utils.plotting import (
    ESCIT, PSILO, plot_colormap_stack, plot_histogram, plot_legend,
    plot_biomarker_heatmap, regression_scatter2, true_vs_pred_scatter)

from figure_making.common import load_biomarker_categories, biomarker_palette
from figure_making.paths import output_dir, posthoc_dir, require
from figure_making.registry import register


# (model, analysis, agreement filename, panel name, colour index)
AGREEMENT_PANELS = [
    ('graphtrip', 'grail', 'grail_agreement_scores.csv',
     'graphTRIP_grail_agreement_histogram', 3),
    ('graphtrip', 'regional_attributions', 'regional_attributions_agreement_scores.csv',
     'graphTRIP_regional_attributions_agreement_histogram', 5),
    ('medusa_graphtrip', 'grail', 'grail_agreement_scores.csv',
     'medusa_graphTRIP_grail_agreement_histogram', 3),
    ('medusa_graphtrip', 'regional_attributions', 'regional_attributions_agreement_scores.csv',
     'medusa_graphTRIP_regional_attributions_agreement_histogram', 5),
]

CONDITION_ORDER = ['Global', 'P', 'E']
MARKER_STYLES = ['o', 's', '^', 'v', 'D', '>', '<', 'p', '*', 'h']


@register('cv_model_agreement', group='supp', subdir='SUPPLEMENTARY/cv_model_agreement')
def cv_model_agreement(ctx, out):
    colors = plot_colormap_stack('YlGnBu', 10, make_plot=False)

    for model, analysis, filename, name, color_idx in AGREEMENT_PANELS:
        results_dir = require(posthoc_dir(model, analysis))
        agreement = pd.read_csv(os.path.join(results_dir, filename))['consistency_score'].values
        r_min = min_significant_r(len(agreement))

        plot_histogram({'consistency': agreement},
                       vline=r_min,
                       save_path=out.fig(name),
                       figsize=(3, 4),
                       alpha=0.65,
                       palette={'consistency': colors[color_idx]})

        out.log(f'{model} / {analysis}: n = {len(agreement)}, '
                f'mean consistency = {np.mean(agreement):.3f}, '
                f'significance threshold = {r_min:.3f}, '
                f'above threshold = {(agreement > r_min).sum()}')


@register('GRAIL_biomarkers', group='supp', subdir='SUPPLEMENTARY/GRAIL_biomarkers')
def grail_biomarkers(ctx, out):

    # a. Alignment captures univariate correlations --------------------------------------
    grail_dir = require(posthoc_dir('graphtrip', 'grail'))
    mean_alignments = pd.read_csv(os.path.join(grail_dir, 'weighted_mean_alignments.csv'))
    feature_names = mean_alignments.columns.tolist()

    biomarker_dir = require(output_dir('graphtrip', 'test_biomarkers'))
    feature_corrs = pd.read_csv(os.path.join(biomarker_dir, 'feature_correlations.csv'))
    feature_corrs = feature_corrs[feature_corrs['feature'].isin(feature_names)]
    feature_corrs = feature_corrs.set_index('feature').loc[feature_names].reset_index()

    # One colour per biomarker category
    groups = [group for group in get_groups(feature_names) if len(group) > 0]
    group_colors = plot_colormap_stack('nipy_spectral', len(groups), make_plot=False)
    palette = {feature: group_colors[i] for i, group in enumerate(groups) for feature in group}

    group_mean_alignments = mean_alignments[feature_names].mean(axis=0)
    plot_df = pd.DataFrame({'feature_correlation': feature_corrs['corr'].values,
                            'group_mean_alignment': group_mean_alignments.values,
                            'feature': feature_names})
    regression_scatter2(plot_df,
                        xcol='feature_correlation',
                        ycol='group_mean_alignment',
                        ylim=None,
                        xlim=None,
                        featcol='feature',
                        palette=palette,
                        equal_aspect=False,
                        save_path=out.fig('group_mean_alignment_vs_feature_corrs'))
    plot_legend(palette, orientation='horizontal', size=(15, 1), label=None,
                save_path=out.fig('group_mean_alignment_vs_feature_corrs_legend'))

    r, p = pearsonr(plot_df['feature_correlation'], plot_df['group_mean_alignment'])
    out.log(f'Candidate biomarkers: {len(feature_names)}')
    out.log(f'Group-mean alignment vs feature correlation: r = {r:.4f}, p = {p:.4e}')
    out.log()

    # b. graphTRIP learns more than simple univariate correlations -----------------------
    ridge = pd.read_csv(os.path.join(biomarker_dir, 'ridge_cv_predict.csv'))
    ridge['Condition'] = ctx.conditions
    true_vs_pred_scatter(ridge, save_path=out.fig('ridge_cv_predict'))

    r, p = pearsonr(ridge['label'], ridge['prediction'])
    out.log(f'Ridge regression on biomarkers: r = {r:.4f}, p = {p:.4e}')
    out.log()

    # c. All biomarker categories ---------------------------------------------------------
    palette = biomarker_palette()
    all_categories, _, all_sorted = load_biomarker_categories(thresh=1.0)
    plot_biomarker_heatmap(all_categories[all_sorted], palette,
                           save_path=out.fig('all_biomarker_cats_heatmap'))

    # d. Identified biomarkers reflect univariate and drug-interaction relationships ------
    _, majority_cat, sorted_biomarkers = load_biomarker_categories(thresh=0.5)

    biomarker_values = pd.read_csv(os.path.join(biomarker_dir, 'feature_values.csv'))
    biomarker_values = biomarker_values.sort_values('sub')
    biomarker_values['Condition'] = ctx.conditions
    biomarker_values = biomarker_values.drop(columns=['sub'])

    cat_corrs = _biomarker_correlations(biomarker_values, sorted_biomarkers, majority_cat)
    out.table('identified_biomarker_correlations', cat_corrs)

    _plot_biomarker_correlations(ctx, out, cat_corrs)


def _biomarker_correlations(biomarker_values, sorted_biomarkers, majority_cat):
    '''Correlates each biomarker with the outcome, globally and within each drug arm.'''
    rows = {'Category': [], 'Biomarker': [], 'Condition': [], 'r': [], 'p': []}
    cond_dict = {'P': 1, 'E': -1}

    for biomarker in sorted_biomarkers:
        df = biomarker_values[[biomarker, 'Condition', 'y']].copy()

        r, p = pearsonr(df[biomarker], df['y'])
        rows['Category'].append(majority_cat[biomarker])
        rows['Biomarker'].append(biomarker)
        rows['Condition'].append('Global')
        rows['r'].append(r)
        rows['p'].append(p)

        for cond_name, cond_val in cond_dict.items():
            df_cond = df[df['Condition'] == cond_val]
            r, p = pearsonr(df_cond[biomarker], df_cond['y'])
            rows['Category'].append(majority_cat[biomarker])
            rows['Biomarker'].append(biomarker)
            rows['Condition'].append(cond_name)
            rows['r'].append(r)
            rows['p'].append(p)

    return pd.DataFrame(rows)


def _plot_biomarker_correlations(ctx, out, cat_corrs):
    '''
    Plots biomarker-outcome correlations grouped by category, split by drug condition.

    Point positions are jittered with the seeded generator on the context, so the panel
    is reproducible (the notebook used an unseeded np.random.normal).
    '''
    cat_order = list(cat_corrs['Category'].unique())
    unique_biomarkers = list(cat_corrs['Biomarker'].unique())
    marker_map = {bm: MARKER_STYLES[i % len(MARKER_STYLES)]
                  for i, bm in enumerate(unique_biomarkers)}

    width = 0.2
    cat_positions = np.arange(len(cat_order))
    cond_offsets = {'Global': -width, 'P': 0, 'E': width}
    cond_colors = {'Global': (0.5, 0.5, 0.5), 'P': PSILO, 'E': ESCIT}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cat in enumerate(cat_order):
        for cnd in CONDITION_ORDER:
            group = cat_corrs[(cat_corrs['Category'] == cat) & (cat_corrs['Condition'] == cnd)]
            if group.empty:
                continue
            pos = cat_positions[i] + cond_offsets[cnd]
            jitter = ctx.rng.normal(0, 0.04, size=len(group))
            for y, offset, bm in zip(group['r'].values, jitter, group['Biomarker'].values):
                ax.scatter(pos + offset, y,
                           color=cond_colors[cnd],
                           edgecolor=cond_colors[cnd],
                           s=50,
                           marker=marker_map[bm],
                           alpha=0.7,
                           zorder=3)

    ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
    ax.set_xticks(cat_positions)
    ax.set_xticklabels(cat_order, rotation=20)
    ax.set_xlabel('Category')
    ax.set_ylabel('Correlation (r)')
    ax.set_title("Grouped Biomarker Correlation by Category and Condition")

    legend_handles = [plt.Line2D([0], [0], color=cond_colors[cnd], marker='o', linestyle='',
                                 markersize=10, label=cnd, markeredgewidth=2)
                      for cnd in CONDITION_ORDER]
    ax.add_artist(ax.legend(handles=legend_handles, title="Condition", loc='upper left'))

    save_path = out.fig('identified_biomarker_correlations')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    # The biomarker marker styles are shown as a separate legend panel
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.axis('off')
    biomarker_handles = [plt.Line2D([0], [0], color='k', marker=marker_map[bm], linestyle='',
                                    markersize=10, label=bm, markerfacecolor='k',
                                    markeredgewidth=0)
                         for bm in unique_biomarkers]
    ax2.add_artist(ax2.legend(handles=biomarker_handles, title="Biomarker", loc='center',
                              ncol=1, frameon=False))

    save_path = out.fig('biomarker_legend')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig2)
