"""
Supplementary: the permutation-null negative control for the reported GRAIL biomarkers.

The null models are retrained on outcomes shuffled across the whole cohort
(scripts/permutation_null.py), 100 permutations x 10 training seeds, and GRAIL is run on
their weights. They share the graphs, the architecture, the candidate features and the
cross-validation splits with the real models, and differ only in the outcome. Two
analyses:

    profile agreement   Does a null ensemble reproduce the real alignment profile at all?
                        Compared against how well two halves of the real ensemble agree
                        with each other, which is the ceiling the comparison could reach.
    per biomarker       Does each reported biomarker beat its own null? Each one is tested
                        against the tree whose claim defines its category, not against
                        graphTRIP in every case: the two E_response_P_resistance markers
                        are Medusa findings, for which a graphTRIP null is not a test.

This target reports the statistics; both panels are drawn by the GRAIL_biomarkers target
in supp_interpretability, alongside the biomarkers they control. The statistics live in
scripts/grail_model_null.py; this module only arranges them. See section 3 of
biomarker_selection_pipeline.md for the construction and its caveats.

Author: Hanna M. Tolle
Date: 2026-09-02
License: BSD 3-Clause
"""

import pandas as pd

from figure_making.registry import register
from scripts.grail_model_null import (
    TREES, load_candidates, load_all_trees, group_means, load_reported, load_sign_claims,
    profile_agreement, omnibus, per_biomarker, tree_sensitivity)


def _values(profile, tree, comparison):
    '''The correlations of one tree and one comparison, as an array.'''
    return profile.loc[(profile['tree'] == tree)
                       & (profile['comparison'] == comparison), 'r'].values


# Target -----------------------------------------------------------------------------

@register('grail_null', group='supp', subdir='SUPPLEMENTARY/grail_null')
def grail_null(ctx, out):
    '''
    Do the reported GRAIL biomarkers survive models that never saw the outcome?
    '''
    feat = load_candidates()
    reported, claims = load_reported(), load_sign_claims()
    observed, null, dropped = load_all_trees(feat)   # raises until the arrays land
    observed_group, null_group = group_means(observed, null)
    first = next(iter(TREES))
    n_seeds, n_folds = observed[first].shape[:2]
    n_draws = len(null_group[first])

    # A. Profile agreement -------------------------------------------------------------
    rows = []
    for tree in TREES:
        within, between = profile_agreement(observed[tree], null[tree])
        rows += [{'tree': tree, 'comparison': 'observed_split_half', 'r': r} for r in within]
        rows += [{'tree': tree, 'comparison': 'observed_vs_null', 'r': r} for r in between]
    profile = pd.DataFrame(rows)
    profile_summary = (profile.groupby(['tree', 'comparison'])['r']
                       .agg(n='size', mean='mean', sd='std', min='min', max='max')
                       .reset_index())

    # B. Omnibus over every sign the synergy table claims -------------------------------
    omni, _ = omnibus(observed_group, null_group, feat, reported, claims)

    # C. Per biomarker, against the null of its primary tree ----------------------------
    table = per_biomarker(observed_group, null_group, feat, reported, claims)
    sensitivity = tree_sensitivity(observed_group, null_group, feat)

    out.table('grail_null_profile_agreement', profile)
    out.table('grail_null_profile_summary', profile_summary)
    out.table('grail_null_omnibus', omni)
    out.table('grail_null_per_biomarker', table)
    out.table('grail_null_tree_sensitivity', sensitivity)

    # Report ---------------------------------------------------------------------------
    out.log(f'GRAIL permutation null: {n_draws} draws per tree, each the mean of one '
            f'permutation\'s {null[first].shape[1]} training seeds; observed from '
            f'{n_seeds} seeds x {n_folds} folds. Rank p floor {1/(1 + n_draws):.4f}.')
    for tree, names in dropped.items():
        if names:
            out.log(f'WARNING: {tree} dropped {len(names)} permutation(s) with an '
                    f'incomplete seed set: {", ".join(names)}.')
    out.log()

    out.log('A. Profile agreement -- the 42 x 69 alignment matrix, flattened and correlated.')
    for tree in TREES:
        within = _values(profile, tree, 'observed_split_half')
        between = _values(profile, tree, 'observed_vs_null')
        out.log(f'  {tree:13s} observed split-half r = {within.mean():.3f} +/- '
                f'{within.std():.3f} ({len(within)} splits); '
                f'null vs observed r = {between.mean():+.3f} +/- {between.std():.3f} '
                f'({len(between)} draws)')
    out.log('  Within one label assignment the profile reproduces; across label '
            'assignments it does not.')
    out.log('  NOTE: a parametric threshold does not apply here -- the 2898 cells are far '
            'from independent, and min_significant_r would call most null draws '
            'significant.')
    out.log()

    out.log(f'B. Omnibus over the {omni["n_claims"].iloc[0]} (biomarker x tree) signs the '
            f'synergy table claims for the {len(reported)} reported biomarkers.')
    out.log_df('', omni.round(4))

    out.log(f'C. Per biomarker, against the null of the tree that defines its category.')
    out.log(f'  {(table["fdr_q"] < 0.05).sum()}/{len(table)} beat their null at FDR < 0.05.')
    out.log(f'  {(table["fwer_p"] < 0.05).sum()}/{len(table)} also survive correction over '
            f'all {len(feat)} candidates by the max-statistic, which is the version immune '
            f'to the reported set having been chosen on these data.')
    out.log(f'  {(table["conjunction_p"] < 0.05).sum()}/{len(table)} beat the null in every '
            f'tree their category constrains; the Medusa psilocybin head caps this, its '
            f'null being wider than its own signal.')
    out.log()
    cols = ['biomarker', 'category', 'primary_tree', 'observed', 'null_mean', 'null_sd',
            'z', 'rank_p', 'fdr_q', 'fwer_p', 'conjunction_trees', 'conjunction_p']
    out.log_df('Reported biomarkers', table[cols].round(4))
    out.log_df('Tree sensitivity', sensitivity.drop(columns='fwer_survivors').round(4))
