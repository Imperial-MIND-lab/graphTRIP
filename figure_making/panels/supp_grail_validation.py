"""
Supplementary: what the GRAIL biomarkers capture, and the permutation-null control on
them.

One target, GRAIL_biomarkers, holds every panel: the alignment-versus-correlation
scatter, the linear-model benchmark, the biomarker categories and their outcome
correlations, and the two permutation-null panels that ask whether any of it survives
models trained on shuffled outcomes.

The permutation-null models are retrained on outcomes shuffled across the whole cohort
(scripts/permutation_null.py), and GRAIL is run on their weights. They carry every source
of structure the real models carry -- the same graphs, the same architecture, the same
splits, the same candidate features -- except the outcome. The null runs have no spin test,
so only mean alignments exist for them, and every comparison here is built on the mean
alignment and nothing downstream of it. Observed and null are constructed identically: all
seeds, all folds, all patients, with no rho>0 fold filter and no performance weighting on
either side.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import glob
import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

from utils.files import add_project_root
from utils.plotting import (
    ALPHA_SCATTER, CMAP_DEFAULT, ESCIT, PSILO, NEUTRAL, NEUTRAL2,
    plot_biomarker_heatmap, true_vs_pred_scatter)

from figure_making.common import load_biomarker_categories, biomarker_palette, fmt_p
from figure_making.paths import (
    output_dir, posthoc_dir, perm_dirs, require, MissingInput)
from figure_making.registry import register


CONDITION_ORDER = ['Global', 'P', 'E']
MARKER_STYLES = ['o', 's', '^', 'v', 'D', '>', '<', 'p', '*', 'h']

# The identified biomarkers and every observed value are marked in dark red, so that they
# read against the grey candidates and grey nulls. Same red as supp_permutation_null.
DARK_RED = '#AA0000'

NULL_NCOLS = 5          # biomarker panels per row
PROFILE_NCOLS = 2       # tree panels per row, giving a 2 x 2 grid for the four trees
PROFILE_BINS = np.linspace(-1, 1, 41)

# The four GRAIL trees the biomarker categories are built from. Each is one model and one
# prediction target: (observed GRAIL run, permutation-null tree, grail_mode within it).
# Medusa writes all three of its heads into one table, tagged by grail_mode; graphTRIP has
# a single head and no such column.
TREES = {
    'Shared':       (('graphtrip', 'grail'),
                     ('graphtrip', 'permutation_null'), None),
    'Psilocybin':   (('medusa_graphtrip', 'grail_psilocybin'),
                     ('medusa_graphtrip', 'permutation_null'), 'psilocybin'),
    'Escitalopram': (('medusa_graphtrip', 'grail_escitalopram'),
                     ('medusa_graphtrip', 'permutation_null'), 'escitalopram'),
    'ITE':          (('medusa_graphtrip', 'grail'),
                     ('medusa_graphtrip', 'permutation_null'), 'ite'),
}

# The tree whose claim defines each category, and is therefore the one the biomarker is
# tested against.
PRIMARY_TREE = {
    'Shared_response': 'Shared',        'Shared_resistance': 'Shared',
    'E_response_P_resistance': 'ITE',   'P_response_E_resistance': 'ITE',
    'E_response': 'Escitalopram',       'E_resistance': 'Escitalopram',
    'P_response': 'Psilocybin',         'P_resistance': 'Psilocybin',
}

# The sign each tree's alignment must have per category. Not under outputs/, so it is the
# one path here that is not built by figure_making.paths.
SYNERGY_FILE = 'experiments/configs/biomarker_synergies.csv'

# The out-of-fold correlation check: graphTRIP's GRAIL run and the folds behind it.
MIN_TRAIN, MIN_TEST = 5, 3

# Folds whose model failed to predict are dropped, matching the rho > 0 inclusion
# criterion analyse_grail_results applies to the published pipeline.
MIN_RHO = 0.0
RHO_VMIN, RHO_VMAX = 0.0, 1.0


# Loading GRAIL results ------------------------------------------------------------------

def _load_cohort(biomarker_dir):
    '''
    The candidate biomarker values and the outcome, one row per patient.

    Returns (values, feat): the frame in subject order, and the candidate names in the
    order the GRAIL tables use.
    '''
    values = pd.read_csv(os.path.join(biomarker_dir, 'feature_values.csv'))
    values = values.sort_values('sub').drop(columns=['sub'])
    return values, [c for c in values.columns if c != 'y']


# graphTRIP's GRAIL run is read by both the out-of-fold check and the null control, and is
# some three thousand small CSVs; the memo means a figure run reads it once. Candidate
# names are passed as a tuple so that the call is hashable.
@lru_cache(maxsize=None)
def _observed_alignments(grail_dir, feat):
    '''
    Observed mean alignments of one GRAIL run, as [seed, fold, subject, biomarker].

    Returns (alignments, seeds), where seeds names the seed directories in the order of
    the first axis. Subjects and folds are counted from the run itself rather than assumed.
    '''
    seed_dirs = sorted(glob.glob(os.path.join(require(grail_dir), 'seed_*')))
    if not seed_dirs:
        raise MissingInput(os.path.join(grail_dir, 'seed_*'))

    n_sub = len(glob.glob(os.path.join(seed_dirs[0], 'sub_*')))
    n_fold = len(pd.read_csv(os.path.join(seed_dirs[0], 'fold_performances.csv')))
    alignments = np.stack([np.stack([np.stack([
        pd.read_csv(os.path.join(seed_dir, f'sub_{i}',
                                 f'k{k}_mean_alignments.csv'))[list(feat)].values[0]
        for i in range(n_sub)]) for k in range(n_fold)]) for seed_dir in seed_dirs])
    return alignments, [os.path.basename(d) for d in seed_dirs]


def _null_alignments(null_dir, feat, modes, n_seeds):
    '''
    Null mean alignments per grail_mode, as {mode: [permutation, subject, biomarker]}.

    Medusa writes all three of its heads into one table, so each permutation's files are
    read once and split by mode rather than re-read per tree. Seeds are averaged inside the
    loader because every statistic below collapses that axis anyway, which keeps the null
    arrays an order of magnitude smaller.

    Permutations with fewer than n_seeds training seeds are dropped: a draw built from
    fewer seeds carries more seed noise and is not the same statistic as the observed one.
    '''
    draws = {mode: [] for mode in modes}
    dropped = []

    for perm_dir in perm_dirs(require(null_dir)):
        files = sorted(glob.glob(os.path.join(perm_dir, 'grail', 'seed_*',
                                              'mean_alignments.csv')))
        if len(files) != n_seeds:
            dropped.append(os.path.basename(perm_dir))
            continue
        tables = [pd.read_csv(path) for path in files]
        for mode in modes:
            per_seed = [t if mode is None else t[t['grail_mode'] == mode] for t in tables]
            draws[mode].append(np.mean([t.groupby('subject')[feat].mean().values
                                        for t in per_seed], axis=0))

    if not all(draws.values()):
        raise FileNotFoundError(f'No complete {n_seeds}-seed permutations in {null_dir}')
    return {mode: np.stack(v) for mode, v in draws.items()}, dropped


def _load_trees(feat):
    '''
    Observed and null alignments for every tree.

    Returns (observed, null, dropped): observed[tree] is [seed, fold, subject, biomarker],
    null[tree] is [permutation, subject, biomarker], and dropped[tree] names any
    permutations left out for having an incomplete set of training seeds.
    '''
    observed, null, dropped = {}, {}, {}
    by_null_dir = {}

    for tree, (grail_parts, null_parts, mode) in TREES.items():
        observed[tree], _ = _observed_alignments(output_dir(*grail_parts), tuple(feat))
        by_null_dir.setdefault(output_dir(*null_parts), []).append((tree, mode))

    for null_dir, entries in by_null_dir.items():
        # The null runs are the same training seeds as the observed ones, so the observed
        # seed count is what a complete permutation has to match.
        n_seeds = observed[entries[0][0]].shape[0]
        draws, skipped = _null_alignments(null_dir, feat, [m for _, m in entries], n_seeds)
        for tree, mode in entries:
            null[tree], dropped[tree] = draws[mode], skipped

    return observed, null, dropped


def _group_means(observed, null):
    '''
    Collapses both sides to one value per (tree, biomarker).

    The group-mean alignment averages over every model and every patient, identically on
    both sides, so the dependence between the fold models sits on both and cancels.
    '''
    return ({t: observed[t].mean(axis=(0, 1)).mean(0) for t in observed},
            {t: null[t].mean(1) for t in null})


def _sign_claims():
    '''
    The sign each tree's alignment must have, per category, from the synergy table.

    A category matches several rows of the table (they differ in the entries it leaves
    free), so an entry is a claim only where it is constant across those rows. Entries that
    vary, and entries that are 0, are not claims: 0 means "must not be significant", which a
    null distribution cannot confirm.
    '''
    synergies = pd.read_csv(require(add_project_root(SYNERGY_FILE)))
    claims = {}
    for category, rows in synergies.groupby('Biomarker_Category'):
        claims[category] = {tree: int(rows[tree].iloc[0])
                            for tree in TREES if rows[tree].nunique() == 1
                            and rows[tree].iloc[0] != 0}
    return claims


# Permutation-null statistics -------------------------------------------------------------

def _rank_p(observed, null, two_sided=True):
    '''
    Where an observed value falls in its null draws.

    Centred on the null mean rather than on zero, because the null is not assumed to be
    centred. Floors at 1/(n+1), which is the binding constraint on everything below.
    '''
    centre = null.mean()
    if two_sided:
        exceed = (np.abs(null - centre) >= abs(observed - centre)).sum()
    else:
        exceed = ((null - centre) >= (observed - centre)).sum()
    return (1 + exceed)/(1 + len(null))


def _null_z(observed, null):
    '''Observed value in units of its null spread.'''
    return float((observed - null.mean())/null.std(ddof=1))


def _leave_one_out_z(null):
    '''
    Each null draw standardised against the other draws, [permutation, biomarker].

    A draw must not contribute to the mean and SD it is judged against, or it is pulled
    towards zero and the null spread comes out too narrow.
    '''
    n = len(null)
    total, total_sq = null.sum(0), (null**2).sum(0)
    mean = (total - null)/(n - 1)
    var = (total_sq - null**2 - (n - 1)*mean**2)/(n - 2)
    return (null - mean)/np.sqrt(np.maximum(var, 1e-20))


def _profile_agreement(observed, null):
    '''
    Do two halves of the real ensemble agree with each other, and does a null ensemble
    agree with the real one?

    Both sides are the same object: a patients x biomarkers matrix of mean alignments,
    flattened and correlated. The observed side is split by training seed (every distinct
    half/half split of the seeds); the null side is one correlation per permutation, each
    against the full observed ensemble.
    '''
    n_seed = observed.shape[0]
    obs_profile = observed.mean(axis=(0, 1)).ravel()

    within = []
    for half in itertools.combinations(range(n_seed), n_seed//2):
        if 0 not in half:                       # each split counted once
            continue
        other = [s for s in range(n_seed) if s not in half]
        within.append(np.corrcoef(observed[list(half)].mean(axis=(0, 1)).ravel(),
                                  observed[other].mean(axis=(0, 1)).ravel())[0, 1])

    between = [np.corrcoef(obs_profile, null[p].ravel())[0, 1] for p in range(len(null))]
    return np.array(within), np.array(between)


def _per_biomarker(observed_group, null_group, feat, reported, claims):
    '''
    Each reported biomarker against the null of the tree that defines its category.

    Two-sided, because the category's sign was itself read off the observed alignments:
    testing one-sided in the direction the data chose would halve the p-value for free.
    Three columns of evidence per biomarker:

      rank_p       primary tree only, FDR-corrected within the reported set
      fwer_p       primary tree, but corrected over all candidates by the maximum |z| any
                   null draw reaches anywhere -- immune to the fact that the reported set
                   was itself chosen on these data
      conjunction  the largest rank p over every tree the synergy table constrains, i.e.
                   requiring the biomarker to beat the null in all of them at once
    '''
    max_z = {tree: np.abs(_leave_one_out_z(null_group[tree])).max(1) for tree in TREES}

    rows = []
    for biomarker, category in reported.items():
        i, tree = feat.index(biomarker), PRIMARY_TREE[category]
        z = _null_z(observed_group[tree][i], null_group[tree][:, i])
        components = {t: _rank_p(observed_group[t][i], null_group[t][:, i])
                      for t in claims[category]}
        rows.append({
            'biomarker': biomarker, 'category': category, 'primary_tree': tree,
            'observed': float(observed_group[tree][i]),
            'null_mean': float(null_group[tree][:, i].mean()),
            'null_sd': float(null_group[tree][:, i].std(ddof=1)),
            'z': z,
            'rank_p': _rank_p(observed_group[tree][i], null_group[tree][:, i]),
            'fwer_p': (1 + (max_z[tree] >= abs(z)).sum())/(1 + len(max_z[tree])),
            'conjunction_trees': '+'.join(components),
            'conjunction_p': max(components.values()),
            **{f'p_{t}': components.get(t, np.nan) for t in TREES}})

    table = pd.DataFrame(rows).sort_values('rank_p').reset_index(drop=True)
    table['fdr_q'] = fdrcorrection(table['rank_p'], alpha=0.05)[1]
    return table


@register('GRAIL_biomarkers', group='supp', subdir='SUPPLEMENTARY/GRAIL_biomarkers')
def grail_biomarkers(ctx, out):

    # The identified biomarkers and their marker symbols are shared by the alignment
    # scatter and the correlation panel, so that one legend serves both.
    categories, majority_cat, sorted_biomarkers = load_biomarker_categories(thresh=0.5)
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
    cohort, feat = _load_cohort(biomarker_dir)

    biomarker_values = cohort.copy()
    biomarker_values['Condition'] = ctx.conditions

    cat_corrs = _biomarker_correlations(biomarker_values, sorted_biomarkers, majority_cat)
    out.table('identified_biomarker_correlations', cat_corrs)

    _plot_biomarker_correlations(ctx, out, cat_corrs, marker_map)

    # e. Held-out correlations against training GRAIL means --------------------------------
    _corr_vs_grail_panel(ctx, out, identified, cohort, feat)

    # f. The permutation-null control -----------------------------------------------------
    # Skipped rather than fatal: the null arrays are the one input of this target that is
    # produced by a separate set of training runs.
    try:
        # Category order, not the panel's plotting order: it decides how biomarkers with
        # the same rank p are ordered in the null table and its panels.
        _grail_null_panels(out, feat,
                           {b: majority_cat[b] for b in categories.columns})
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
                label=f'{label} (n = {len(subset)}): r = {r:.4f}, p = {fmt_p(p)}')
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


# Held-out correlations against training GRAIL means ---------------------------------

def _fold_rho(grail_dir):
    '''{(seed, fold): test-fold rho} of the models behind a GRAIL run.'''
    rho = {}
    for seed_dir in sorted(glob.glob(os.path.join(grail_dir, 'seed_*'))):
        performances = pd.read_csv(os.path.join(seed_dir, 'fold_performances.csv'))
        for _, row in performances.iterrows():
            rho[(os.path.basename(seed_dir), int(row['fold']))] = float(row['rho'])
    return rho


def _corr_cols(X, y):
    '''Pearson correlation of every column of X with y.'''
    Xz = (X - X.mean(0))/(X.std(0) + 1e-12)
    yz = (y - y.mean())/(y.std() + 1e-12)
    return Xz.T @ yz/len(y)


def _corr_vec(a, b):
    '''Pearson correlation of two vectors, nan if either is constant.'''
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _out_of_fold_correlations(ctx, identified, cohort, feat):
    '''
    Per (seed, fold) model: how well the biomarkers' held-out outcome correlations line
    up with their training-patient GRAIL means, for the identified biomarkers and for the
    remaining candidates.
    '''
    X, y = cohort[feat].values, cohort['y'].values.astype(float)
    groups = np.array(['identified' if f in identified else 'other' for f in feat])

    grail_dir = output_dir('graphtrip', 'grail')
    align, seeds = _observed_alignments(grail_dir, tuple(feat))
    held = ctx.test_indices_dict
    rho = _fold_rho(grail_dir)

    rows, dropped = [], 0
    for s_idx, s in enumerate(seeds):
        for k in range(align.shape[1]):
            if MIN_RHO is not None and not rho.get((s, k), -np.inf) > MIN_RHO:
                dropped += 1
                continue
            train, test = np.where(held[s] != k)[0], np.where(held[s] == k)[0]
            if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
                dropped += 1
                continue

            xy_test = _corr_cols(X[test], y[test])       # held-out patients
            g_train = align[s_idx][k][train].mean(0)     # training patients
            rows.append({'seed': s, 'fold': k, 'n_train': len(train), 'n_test': len(test),
                         'rho': rho.get((s, k), np.nan),
                         **{f'{g}_r': _corr_vec(xy_test[groups == g], g_train[groups == g])
                            for g in ('identified', 'other')}})

    return pd.DataFrame(rows), int((groups == 'identified').sum()), dropped


def _paired_wilcoxon(df, a='identified_r', b='other_r'):
    '''One-sided Wilcoxon on the per-seed means, paired by training seed.'''
    per_seed = df[[a, b, 'seed']].dropna().groupby('seed')[[a, b]].mean()
    if len(per_seed) < 5 or np.allclose(per_seed[a], per_seed[b]):
        return np.nan
    return wilcoxon(per_seed[a], per_seed[b], alternative='greater').pvalue


def _corr_vs_grail_boxplot(df, pval, out, rng,
                           name='corr_vs_grail_reversed_boxplots'):
    '''
    Identified against the other candidates, one point per fold model, coloured by how
    well that model predicted its held-out patients.
    '''
    fig, ax = plt.subplots(figsize=(4.4, 5.4), constrained_layout=True)
    values = [df['identified_r'].dropna().values, df['other_r'].dropna().values]
    low = min(v.min() for v in values)
    high = max(v.max() for v in values)
    pad = 0.1*(high - low)

    box = ax.boxplot(values, labels=['identified', 'other'], widths=0.55,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', lw=1.5))
    for patch, colour, alpha in zip(box['boxes'], (DARK_RED, NEUTRAL), (0.25, 0.7)):
        patch.set_facecolor(colour)
        patch.set_alpha(alpha)

    for i, group in enumerate(('identified', 'other')):
        sub = df.dropna(subset=[f'{group}_r'])
        points = ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(sub)),
                            sub[f'{group}_r'].values, c=sub['rho'].values,
                            cmap=CMAP_DEFAULT, vmin=RHO_VMIN, vmax=RHO_VMAX, s=18,
                            edgecolor='0.3', linewidth=0.25, alpha=0.9, zorder=3)

    if not np.isnan(pval) and pval < 0.05:
        ax.text(1.5, high + 1.0*pad, '*', color=DARK_RED, fontsize=22,
                ha='center', va='center', fontweight='bold')

    ax.axhline(0, color='lightgray', ls='--', lw=1, zorder=0)
    ax.set_ylim(low - pad, high + 1.9*pad)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('corr across biomarkers\n(held-out correlations vs training GRAIL means)')
    ax.set_title(f'Out-of-fold direction, graphTRIP pooled\n'
                 f"paired Wilcoxon over {df['seed'].nunique()} training seeds: "
                 f'p = {fmt_p(pval)}', fontsize=9, linespacing=1.35)
    fig.colorbar(points, ax=ax, label="model's test-fold rho", fraction=0.06, pad=0.03)

    _save(fig, out, name)


def _corr_vs_grail_panel(ctx, out, identified, cohort, feat):
    '''Draws the out-of-fold correlation check and reports what it found.'''
    try:
        df, n_identified, dropped = _out_of_fold_correlations(ctx, identified, cohort, feat)
    except (MissingInput, FileNotFoundError, ValueError) as error:
        out.log(f'No out-of-fold correlation panel: {error}')
        out.log()
        return

    pval = _paired_wilcoxon(df)
    _corr_vs_grail_boxplot(df, pval, out, ctx.rng)
    out.table('corr_vs_grail_reversed_scores', df)

    out.log(f'Held-out biomarker-outcome correlations vs training GRAIL means, over '
            f'{len(df)} fold models ({dropped} dropped for test-fold rho <= {MIN_RHO} or '
            f'too few patients); {n_identified} identified biomarkers against the '
            f'remaining candidates.')
    for group in ('identified', 'other'):
        values = df[f'{group}_r'].dropna()
        out.log(f'  {group:11s} median r = {values.median():+.4f}, '
                f'mean = {values.mean():+.4f} +/- {values.std():.4f}')
    out.log(f'  paired Wilcoxon (one-sided, identified > other) over '
            f"{df['seed'].nunique()} training seeds: p = {fmt_p(pval)}")
    out.log()


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


def _grail_null_panels(out, feat, reported):
    '''
    Draws both permutation-null panels and writes the statistics behind them.

    The full-precision z, rank p and FDR q of every panel of grail_null_histograms go to
    grail_null_per_biomarker.csv, alongside the tree each biomarker was tested against
    and the two stricter corrections.
    '''
    claims = _sign_claims()
    observed, null, dropped = _load_trees(feat)
    observed_group, null_group = _group_means(observed, null)

    rows = []
    for tree in TREES:
        within, between = _profile_agreement(observed[tree], null[tree])
        rows += [{'tree': tree, 'comparison': 'observed_split_half', 'r': r} for r in within]
        rows += [{'tree': tree, 'comparison': 'observed_vs_null', 'r': r} for r in between]
    profile = pd.DataFrame(rows)
    _profile_histograms(profile, out)

    table = _per_biomarker(observed_group, null_group, feat, reported, claims)
    _null_histograms(table, null_group, feat, out)
    _null_summary(table, null_group, feat, out)

    out.table('grail_null_per_biomarker', table)
    out.table('grail_null_profile_agreement', profile)

    first = next(iter(TREES))
    n_draws = len(null_group[first])
    n_seeds, n_folds = observed[first].shape[:2]
    out.log(f'GRAIL permutation null: {n_draws} draws per tree, each the mean of one '
            f"permutation's {n_seeds} training seeds; observed from "
            f'{n_seeds} seeds x {n_folds} folds. '
            f'Rank p floor {1/(1 + n_draws):.4f}.')
    for tree, names in dropped.items():
        if names:
            out.log(f'  WARNING: {tree} dropped {len(names)} permutation(s) with an '
                    f'incomplete seed set: {", ".join(names)}.')
    for tree in TREES:
        within = profile.loc[(profile['tree'] == tree)
                             & (profile['comparison'] == 'observed_split_half'), 'r']
        between = profile.loc[(profile['tree'] == tree)
                              & (profile['comparison'] == 'observed_vs_null'), 'r']
        out.log(f'  {tree:13s} observed split-half r = {within.mean():.4f} +/- '
                f'{within.std():.4f}; null vs observed r = {between.mean():+.4f} +/- '
                f'{between.std():.4f}')
    out.log(f'  {(table["fdr_q"] < 0.05).sum()}/{len(table)} reported biomarkers beat '
            f'their null at FDR < 0.05; {(table["fwer_p"] < 0.05).sum()}/{len(table)} '
            f'also survive the max-statistic correction over all {len(feat)} candidates.')
    out.log_df('Reported biomarkers against their null',
               table[['biomarker', 'category', 'primary_tree', 'observed', 'null_mean',
                      'null_sd', 'z', 'rank_p', 'fdr_q', 'fwer_p', 'conjunction_trees',
                      'conjunction_p']].round(4))


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
                   linewidth=0.4, alpha=0.85, label=f'{label} ({values.mean():+.4f})')
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


def _null_summary(table, null_group, feat, out, name='grail_null_summary'):
    '''
    The whole per-biomarker figure on one axis, for when the ten histograms are too much.
    '''
    order = table.iloc[::-1]        # most significant at the top of the axis
    nulls, observed = [], []
    for _, row in order.iterrows():
        null = null_group[row['primary_tree']][:, feat.index(row['biomarker'])]
        nulls.append((null - null.mean())/null.std(ddof=1))
        observed.append(row['z'])   # the same standardisation, from the table

    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    box = ax.boxplot(nulls, vert=False, widths=0.6, whis=(0, 100), showfliers=False,
                     patch_artist=True, medianprops=dict(color=NEUTRAL2, lw=1.0))
    for patch in box['boxes']:
        patch.set_facecolor(NEUTRAL)
        patch.set_alpha(0.7)
        patch.set_edgecolor(NEUTRAL2)
    ax.scatter(observed, range(1, len(observed) + 1), color=DARK_RED, marker='D', s=26,
               zorder=3, label='observed')
    ax.plot([], [], 's', color=NEUTRAL, markeredgecolor=NEUTRAL2, markersize=7,
            label='null draws (box: IQR, whiskers: range)')

    ax.axvline(0, color=NEUTRAL2, lw=0.8, ls=':', zorder=0)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order['biomarker'], fontsize=7)
    for tick, q in zip(ax.get_yticklabels(), order['fdr_q']):
        if q < 0.05:
            tick.set_color(DARK_RED)
            tick.set_fontweight('bold')
    ax.set_xlabel('mean alignment, in units of the biomarker\'s own null spread',
                  fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    # Room for the strongest observed value, which sits well outside its null
    low = min(observed + [n.min() for n in nulls])
    high = max(observed + [n.max() for n in nulls])
    pad = 0.06*(high - low)
    ax.set_xlim(low - pad, high + pad)
    fig.legend(loc='outside lower center', ncol=2, fontsize=6.5, frameon=False,
               handlelength=1.2)

    # The tree each biomarker was tested against, and its q, on the opposite side
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(1, len(order) + 1))
    right.set_yticklabels([f"{row['primary_tree']}, q = {fmt_p(row['fdr_q'])}"
                           for _, row in order.iterrows()], fontsize=6.5)
    right.tick_params(length=0)
    right.spines[['top', 'right', 'left']].set_visible(False)

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
                     f"z = {row['z']:+.2f}, p = {fmt_p(row['rank_p'])}, "
                     f"q = {fmt_p(row['fdr_q'])}",
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
