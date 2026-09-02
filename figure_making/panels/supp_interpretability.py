"""
Supplementary: agreement between cross-validation fold models, what the GRAIL biomarkers
capture, and the permutation-null control on them.

The null panels were previously drawn by the grail_null target; that target now only
computes and reports the statistics behind them (scripts/grail_model_null.py, and
section 3 of biomarker_selection_pipeline.md for the construction and its caveats).

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde

from utils.statsalg import min_significant_r
from utils.plotting import (
    ALPHA_SCATTER, ESCIT, PSILO, NEUTRAL, NEUTRAL2, plot_colormap_stack, plot_histogram,
    plot_biomarker_heatmap, true_vs_pred_scatter)

from figure_making.common import (
    load_biomarker_categories, biomarker_palette, fmt_p_floor)
from figure_making.paths import output_dir, posthoc_dir, require, MissingInput
from figure_making.registry import register
from scripts.grail_model_null import (
    TREES, load_candidates, load_all_trees, group_means, load_reported, load_sign_claims,
    profile_agreement, per_biomarker)


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

# The identified biomarkers and every observed value are marked in dark red, so that they
# read against the grey candidates and grey nulls. Same red as supp_permutation_null.
DARK_RED = '#AA0000'

NULL_NCOLS = 5          # biomarker panels per row
PROFILE_NCOLS = 2       # tree panels per row, giving a 2 x 2 grid for the four trees
PROFILE_BINS = np.linspace(-1, 1, 41)


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

    # The identified biomarkers and their marker symbols are shared by the alignment
    # scatter and the correlation panel, so that one legend serves both.
    _, majority_cat, sorted_biomarkers = load_biomarker_categories(thresh=0.5)
    marker_map = {bm: MARKER_STYLES[i % len(MARKER_STYLES)]
                  for i, bm in enumerate(sorted_biomarkers)}

    # a. Alignment captures univariate correlations --------------------------------------
    grail_dir = require(posthoc_dir('graphtrip', 'grail'))
    mean_alignments = pd.read_csv(os.path.join(grail_dir, 'weighted_mean_alignments.csv'))
    feature_names = mean_alignments.columns.tolist()

    biomarker_dir = require(output_dir('graphtrip', 'test_biomarkers'))
    feature_corrs = pd.read_csv(os.path.join(biomarker_dir, 'feature_correlations.csv'))
    feature_corrs = feature_corrs[feature_corrs['feature'].isin(feature_names)]
    feature_corrs = feature_corrs.set_index('feature').loc[feature_names].reset_index()

    group_mean_alignments = mean_alignments[feature_names].mean(axis=0)
    plot_df = pd.DataFrame({'feature_correlation': feature_corrs['corr'].values,
                            'group_mean_alignment': group_mean_alignments.values,
                            'feature': feature_names})
    identified = [bm for bm in sorted_biomarkers if bm in feature_names]
    fits = _alignment_vs_correlation(plot_df, identified, marker_map, out)

    out.log(f'Candidate biomarkers: {len(feature_names)}, '
            f'of which identified: {len(identified)}')
    for row in fits:
        out.log(f'Group-mean alignment vs feature correlation, {row["set"]} '
                f'(n = {row["n"]}): r = {row["r"]:.4f}, p = {row["p"]:.4e}')
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
    biomarker_values = pd.read_csv(os.path.join(biomarker_dir, 'feature_values.csv'))
    biomarker_values = biomarker_values.sort_values('sub')
    biomarker_values['Condition'] = ctx.conditions
    biomarker_values = biomarker_values.drop(columns=['sub'])

    cat_corrs = _biomarker_correlations(biomarker_values, sorted_biomarkers, majority_cat)
    out.table('identified_biomarker_correlations', cat_corrs)

    _plot_biomarker_correlations(ctx, out, cat_corrs, marker_map)

    # e. The permutation-null control -----------------------------------------------------
    # Skipped rather than fatal: the null arrays are the one input of this target that is
    # produced by a separate set of training runs.
    try:
        _grail_null_panels(out)
    except (MissingInput, FileNotFoundError, ValueError) as error:
        out.log(f'No GRAIL permutation-null panels: {error}')


def _alignment_vs_correlation(plot_df, identified, marker_map, out,
                              name='group_mean_alignment_vs_feature_corrs'):
    '''
    Group-mean alignment against univariate outcome correlation, one point per candidate.

    Two regression lines: over all candidates, and over the identified biomarkers alone,
    which is the subset the manuscript reports. The identified points carry the marker
    symbols of the biomarker_legend panel, so no separate colour legend is needed. The
    axes are square in the panel, not in data units: the two quantities are not on a
    common scale, so an equal aspect would flatten the alignments into a line.

    Returns the fit of each line, for the statistics report.
    '''
    is_identified = plot_df['feature'].isin(identified)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax.set_box_aspect(1)

    rest = plot_df[~is_identified]
    ax.scatter(rest['feature_correlation'], rest['group_mean_alignment'],
               marker='o', s=40, color=NEUTRAL, edgecolor=NEUTRAL2, linewidth=0.4,
               alpha=ALPHA_SCATTER, zorder=2)
    for _, row in plot_df[is_identified].iterrows():
        ax.scatter(row['feature_correlation'], row['group_mean_alignment'],
                   marker=marker_map[row['feature']], s=70, color=DARK_RED,
                   alpha=ALPHA_SCATTER, zorder=3)

    fits = []
    for subset, colour, label in [(plot_df, NEUTRAL2, 'all candidates'),
                                  (plot_df[is_identified], DARK_RED, 'identified')]:
        x = subset['feature_correlation'].values
        y = subset['group_mean_alignment'].values
        r, p = pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, slope*x_line + intercept, color=colour, linewidth=1.8,
                alpha=0.8, zorder=4,
                label=f'{label} (n = {len(subset)}): r = {r:.2f}, {fmt_p_floor(p)}')
        fits.append({'set': label, 'n': len(subset), 'r': r, 'p': p})

    ax.set_xlabel('Feature correlation with outcome (r)')
    ax.set_ylabel('Group-mean alignment')
    ax.grid(True)

    # Below the axes, so that neither fit label covers a point. Constrained layout
    # reserves the space, which plain savefig would otherwise clip.
    fig.legend(loc='outside lower center', fontsize=8, frameon=False)

    _save(fig, out, name)
    return fits


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


def _plot_biomarker_correlations(ctx, out, cat_corrs, marker_map):
    '''
    Plots biomarker-outcome correlations grouped by category, split by drug condition.

    Point positions are jittered with the seeded generator on the context, so the panel
    is reproducible (the notebook used an unseeded np.random.normal).
    '''
    cat_order = list(cat_corrs['Category'].unique())
    unique_biomarkers = list(cat_corrs['Biomarker'].unique())

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


# The permutation-null control -------------------------------------------------------

def _save(fig, out, name):
    save_path = out.fig(name)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def _grid(n, ncols, width, height):
    '''A grid of axes with the unused cells removed.'''
    ncols = min(ncols, n)
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.remove()
    return fig, axes[:n], ncols


def _grail_null_panels(out):
    '''
    Draws the two permutation-null panels from the same arrays the grail_null target
    reports on; scripts.grail_model_null caches the load, so a full figure run reads the
    thousands of GRAIL tables once.
    '''
    feat = load_candidates()
    reported, claims = load_reported(), load_sign_claims()
    observed, null, _ = load_all_trees(feat)
    observed_group, null_group = group_means(observed, null)

    rows = []
    for tree in TREES:
        within, between = profile_agreement(observed[tree], null[tree])
        rows += [{'tree': tree, 'comparison': 'observed_split_half', 'r': r} for r in within]
        rows += [{'tree': tree, 'comparison': 'observed_vs_null', 'r': r} for r in between]
    _profile_histograms(pd.DataFrame(rows), out)

    _null_histograms(per_biomarker(observed_group, null_group, feat, reported, claims),
                     null_group, feat, out)


def _profile_histograms(profile, out, name='grail_null_profile_agreement'):
    '''
    One axis per tree: how well the real ensemble agrees with itself, against how well a
    null ensemble agrees with it.

    Each histogram is scaled to its own peak. The two have very different spreads -- the
    split-half correlations sit in a single bin -- so plotting raw counts would flatten the
    null distribution into the axis. No density fit here for the same reason: a curve
    through a single occupied bin would be a fabrication.
    '''
    fig, axes, ncols = _grid(len(TREES), PROFILE_NCOLS, 3.4, 2.8)
    series = [('observed_vs_null', NEUTRAL, 'null ensemble vs observed'),
              ('observed_split_half', DARK_RED, 'observed, split by seed')]

    for i, (ax, tree) in enumerate(zip(axes, TREES)):
        for comparison, colour, label in series:
            values = profile.loc[(profile['tree'] == tree)
                                 & (profile['comparison'] == comparison), 'r'].values
            counts, _ = np.histogram(values, bins=PROFILE_BINS)
            ax.bar(PROFILE_BINS[:-1], counts/counts.max(), align='edge',
                   width=np.diff(PROFILE_BINS), color=colour, edgecolor=NEUTRAL2,
                   linewidth=0.4, alpha=0.85, label=f'{label} ({values.mean():+.2f})')
        ax.axvline(0, color=NEUTRAL2, lw=0.8, ls=':')
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.55)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(tree, fontsize=9, pad=4)
        ax.set_xlabel('profile correlation (r)', fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel('frequency (peak-scaled)', fontsize=8)
        ax.legend(loc='upper left', fontsize=6.5, frameon=False, handlelength=1.0)
        ax.tick_params(labelsize=7)

    _save(fig, out, name)


def _null_histograms(table, null_group, feat, out, name='grail_null_histograms'):
    '''
    One null distribution per reported biomarker, with the observed alignment marked.

    The draws come from the biomarker's primary tree, so each panel is the test the
    per-biomarker table reports. Bars are densities so that the Gaussian kernel density
    estimate can be drawn over them. Titles are red where the biomarker beats its null at
    FDR < 0.05 across the reported set.
    '''
    fig, axes, ncols = _grid(len(table), NULL_NCOLS, 3.0, 2.8)

    for i, (ax, (_, row)) in enumerate(zip(axes, table.iterrows())):
        null = null_group[row['primary_tree']][:, feat.index(row['biomarker'])]

        # Keep the observed line off the frame, wherever it falls relative to the draws
        lo, hi = min(null.min(), row['observed']), max(null.max(), row['observed'])
        pad = 0.08*(hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 400)
        density = gaussian_kde(null)(grid)

        ax.hist(null, bins=20, density=True, color=NEUTRAL, edgecolor=NEUTRAL2,
                linewidth=0.5, label='null draws')
        ax.fill_between(grid, density, color=NEUTRAL2, alpha=0.25, linewidth=0)
        ax.plot(grid, density, color=NEUTRAL2, lw=1.4, label='Gaussian KDE')
        ax.axvline(row['null_mean'], color=NEUTRAL2, lw=1.0, ls='--', label='null mean')
        ax.axvline(row['observed'], color=DARK_RED, lw=1.8, label='observed')
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(bottom=0)

        significant = row['fdr_q'] < 0.05
        ax.set_title(f"{row['biomarker']}  ({row['primary_tree']})\n"
                     f"z = {row['z']:+.2f}, p = {row['rank_p']:.3f}, q = {row['fdr_q']:.3f}",
                     fontsize=8, pad=6,
                     color=DARK_RED if significant else 'black',
                     fontweight='bold' if significant else 'normal')
        ax.set_xlabel('mean alignment', fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel('density', fontsize=8)
        if i == 0:
            ax.legend(loc='upper left', fontsize=6, frameon=False, handlelength=1.0)
        ax.tick_params(labelsize=7)

    _save(fig, out, name)
