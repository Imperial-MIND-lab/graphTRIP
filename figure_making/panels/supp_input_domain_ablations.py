"""
Supplementary: input-domain ablations of graphTRIP and of Medusa-graphTRIP.

Both families are presented side by side, so they share this target's output directory,
one subdirectory each: the same ablation ladder (clinical, FC and REACT node features,
held against the full model), the same ordering, and the same names for the retained
input domains.

Each family writes three tables. seed_performance.csv is the raw per-seed metrics every
other number is derived from; performance_summary.csv aggregates them two ways, which
must not be confused for one another; model_comparisons.csv holds every test. The
statistics report at the root is a reader's guide to those tables, not a second copy of
them.

Author: Hanna M. Tolle
Date: 2026-08-30
License: BSD 3-Clause
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from figure_making.common import (
    scatter_from_results, collect_seed_metric_table, collect_seed_metrics,
    compare_across_metrics, feature_ablation_panel, aggregate_prediction_results,
    attach_annotations, prediction_metrics, within_arm_metrics, fmt_p)
from figure_making.paths import output_dir
from figure_making.registry import register


GRAPHTRIP_FULL = ('graphtrip', 'weights')
MEDUSA_FULL = ('medusa_graphtrip', 'weights')

# Names of the ablated models in the figure, keyed by the results directory they live in.
ABLATION_NAMES = {
    'no_clinical_features': 'FC + REACT',
    'no_node_features': 'FC + clinical',
    'no_react_no_clinical': 'FC only',
}

# Metrics summarised in the tables. Rank test skips mse (redundant with rmse).
SUMMARY_METRICS = ['r', 'r2', 'mae', 'mse', 'rmse']

# (display name, results directory parts, input domains received, scatter name).
GRAPHTRIP_LADDER = [
    ('graphTRIP', GRAPHTRIP_FULL, ('Clinical', 'FC', 'REACT'), None),
    (ABLATION_NAMES['no_node_features'],
     ('ablation', 'feature_ablation', 'no_node_features'), ('Clinical', 'FC'),
     'scatter_fc_clinical'),
    (ABLATION_NAMES['no_clinical_features'],
     ('ablation', 'feature_ablation', 'no_clinical_features'), ('FC', 'REACT'),
     'scatter_fc_react'),
    (ABLATION_NAMES['no_react_no_clinical'],
     ('ablation', 'feature_ablation', 'no_react_no_clinical'), ('FC',),
     'scatter_fc_only'),
]

MEDUSA_LADDER = [
    ('Medusa-graphTRIP', MEDUSA_FULL, ('Clinical', 'FC', 'REACT'), None),
    (ABLATION_NAMES['no_node_features'],
     ('medusa_ablation', 'no_node_features'), ('Clinical', 'FC'), 'scatter_fc_clinical'),
    (ABLATION_NAMES['no_clinical_features'],
     ('medusa_ablation', 'no_clinical_features'), ('FC', 'REACT'), 'scatter_fc_react'),
    (ABLATION_NAMES['no_react_no_clinical'],
     ('medusa_ablation', 'no_react_no_clinical'), ('FC',), 'scatter_fc_only'),
]

# Clinical-only benchmarks (not included in the raincloud plots)
GRAPHTRIP_BENCHMARKS = [
    ('Clinical only (MLP)', ('ablation', 'feature_ablation', 'control_mlp_raw'),
     'scatter_clinical_mlp'),
    ('Clinical only (OLS)', ('ablation', 'feature_ablation', 'linreg_on_clinical_data'),
     'scatter_clinical_ols'),
]

# (contrast name, ablated graphTRIP model, ablated Medusa model).
INTERACTION_CONTRASTS = [
    ('REACT node features', ('ablation', 'feature_ablation', 'no_node_features'),
     ('medusa_ablation', 'no_node_features')),
    ('clinical features', ('ablation', 'feature_ablation', 'no_clinical_features'),
     ('medusa_ablation', 'no_clinical_features')),
]

COMPARISON_COLUMNS = ['metric', 'test', 'model_a', 'model_b', 'arm_a', 'arm_b',
                      'n', 'statistic', 'p', 'p_bh', 'significant']


# Per-family tables ----------------------------------------------------------------------

def _pooled_results(base_dir):
    '''Predictions averaged across seeds, with the clinical annotations attached.'''
    return attach_annotations(aggregate_prediction_results(
        results_file=os.path.join(base_dir, 'prediction_results.csv')))


def _seed_mean_rows(seed_df, model):
    '''Mean and sem of the per-seed metrics: the distributions the raincloud draws.'''
    values = seed_df[seed_df['model'] == model]
    return [{'model': model, 'metric': metric, 'estimator': 'seed_mean', 'arm': 'overall',
             'n': len(values), 'value': values[metric].mean(), 'sem': values[metric].sem(),
             'p': np.nan}
            for metric in SUMMARY_METRICS]


def _pooled_rows(model, results):
    '''
    Accuracy of the mean-across-seed predictions, overall and within each arm.

    A different estimator from the seed mean, not a more precise version of it: the
    correlation of the averaged predictions is not the average of the per-seed
    correlations, and the two must not be swapped for one another.
    '''
    overall = prediction_metrics(results, model)
    rows = [{'model': model, 'metric': metric, 'estimator': 'pooled_predictions',
             'arm': 'overall', 'n': overall['n'], 'value': value, 'sem': np.nan,
             'p': overall['p'] if metric == 'r' else np.nan}
            for metric, value in [('r', overall['r']), ('r2', overall['R2']),
                                  ('mae', overall['mae']), ('mse', overall['rmse'] ** 2),
                                  ('rmse', overall['rmse'])]]

    for _, arm in within_arm_metrics(results, model).iterrows():
        if arm['arm'] == 'difference':
            continue
        for metric in ('r', 'mae', 'bias'):
            rows.append({'model': model, 'metric': metric,
                         'estimator': 'pooled_predictions', 'arm': arm['arm'],
                         'n': arm['n'], 'value': arm[metric], 'sem': np.nan,
                         'p': arm['p'] if metric == 'r' else np.nan})
    return rows


def _arm_difference_rows(model, results):
    '''The Fisher z that a model's accuracy differs between the two treatment arms.'''
    arms = within_arm_metrics(results, model)
    difference = arms[arms['arm'] == 'difference']
    if difference.empty:
        return []
    row = difference.iloc[0]
    return [{'metric': 'r', 'test': 'fisher_z', 'model_a': model, 'model_b': model,
             'arm_a': 'Psilocybin', 'arm_b': 'Escitalopram', 'n': row['n'],
             'statistic': row['z'], 'p': row['p'], 'p_bh': np.nan,
             'significant': bool(row['p'] < 0.05)}]


def family_tables(ladder, benchmarks, out, subdir):
    '''
    Writes one family's three tables, its raincloud and its scatters.

    Returns:
    -------
        tuple: (seed_df, summary_df, comparisons_df)
    '''
    models = [(label, output_dir(*parts)) for label, parts, _, _ in ladder]
    models += [(label, output_dir(*parts)) for label, parts, _ in benchmarks]
    ladder_order = [label for label, _, _, _ in ladder]

    # Raw per-seed metrics: everything below is derived from this table alone.
    seed_df = collect_seed_metric_table(models)
    seed_df = seed_df[['model', 'seed'] + [c for c in seed_df.columns
                                           if c not in ('model', 'seed')]]
    out.table(f'{subdir}/seed_performance', seed_df)

    summary_rows, comparison_rows = [], []
    for model, base_dir in models:
        results = _pooled_results(base_dir)
        summary_rows += _seed_mean_rows(seed_df, model)
        summary_rows += _pooled_rows(model, results)
        comparison_rows += _arm_difference_rows(model, results)
    summary_df = pd.DataFrame(summary_rows)
    out.table(f'{subdir}/performance_summary', summary_df)

    comparisons_df = pd.concat(
        [compare_across_metrics(seed_df, ladder_order, ladder_order[0]),
         pd.DataFrame(comparison_rows)], ignore_index=True)
    comparisons_df = comparisons_df.reindex(columns=COMPARISON_COLUMNS).fillna(
        {'model_a': '', 'model_b': '', 'arm_a': '', 'arm_b': ''})
    out.table(f'{subdir}/model_comparisons', comparisons_df)

    # The panel reads the same per-seed r values the tests above were run on.
    distributions = {label: seed_df[seed_df['model'] == label]['r'].tolist()
                     for label in ladder_order}
    feature_ablation_panel(
        distributions, {label: domains for label, _, domains, _ in ladder},
        out, f'{subdir}/raincloud_feature_ablations')

    for label, parts, _, scatter in ladder:
        if scatter is not None:
            _scatter(parts, out, f'{subdir}/{scatter}', f'{ladder_order[0]}: {label}')
    for label, parts, scatter in benchmarks:
        _scatter(parts, out, f'{subdir}/{scatter}', label)

    return seed_df, summary_df, comparisons_df


def _scatter(parts, out, name, title):
    scatter_from_results(os.path.join(output_dir(*parts), 'prediction_results.csv'),
                         out, name, yerr='prediction_sem', title=title)


# Medusa versus graphTRIP ----------------------------------------------------------------

def interaction_tests(out):
    '''
    Tests whether an ablation costs more in one model family than in the other.

    The cost of an ablation is the per-seed drop in r from the full model. Both families
    are trained on the same cohort with the same seeds, so the two costs are paired and
    their difference is the interaction: a positive value means the domain matters more
    for Medusa, i.e. more for predicting differential than overall response.
    '''
    rows = []
    for label, graphtrip_parts, medusa_parts in INTERACTION_CONTRASTS:
        specs = [
            ('graphtrip_full', output_dir(*GRAPHTRIP_FULL), 'final_metrics.csv'),
            ('graphtrip_ablated', output_dir(*graphtrip_parts), 'final_metrics.csv'),
            ('medusa_full', output_dir(*MEDUSA_FULL), 'final_metrics.csv'),
            ('medusa_ablated', output_dir(*medusa_parts), 'final_metrics.csv'),
        ]
        r = collect_seed_metrics(specs).pivot(
            index='seed', columns='model', values='r').dropna()
        graphtrip_cost = r['graphtrip_full'] - r['graphtrip_ablated']
        medusa_cost = r['medusa_full'] - r['medusa_ablated']
        difference = medusa_cost - graphtrip_cost

        t_stat, p_ttest = stats.ttest_rel(medusa_cost, graphtrip_cost)
        p_wilcoxon = stats.wilcoxon(medusa_cost, graphtrip_cost).pvalue

        rows.append({
            'ablated_domain': label,
            'n_seeds': len(r),
            'graphtrip_cost': graphtrip_cost.mean(),
            'graphtrip_cost_sd': graphtrip_cost.std(ddof=1),
            'medusa_cost': medusa_cost.mean(),
            'medusa_cost_sd': medusa_cost.std(ddof=1),
            'interaction': difference.mean(),
            'interaction_sd': difference.std(ddof=1),
            't': t_stat,
            'p_ttest': p_ttest,
            'p_wilcoxon': p_wilcoxon,
        })

    table = pd.DataFrame(rows)
    out.table('medusa_vs_graphtrip_interaction', table)
    return table


# Statistics report ----------------------------------------------------------------------

GUIDE = '''Input-domain ablations of graphTRIP and Medusa-graphTRIP
Written by figure_making/panels/supp_input_domain_ablations.py

Files
    <family>/seed_performance.csv     Raw per-seed metrics, one row per model and seed.
                                      Every number elsewhere is derived from this file.
    <family>/performance_summary.csv  model x metric x estimator x arm.
                                      estimator=seed_mean: mean and sem over the seeds,
                                      the distributions the raincloud draws.
                                      estimator=pooled_predictions: the metric of the
                                      mean-across-seed predictions, over the patients.
                                      These are different estimators, not a coarse and a
                                      fine version of one: the correlation of the averaged
                                      predictions is not the average of the per-seed
                                      correlations, and the two must not be swapped.
    <family>/model_comparisons.csv    One row per test.
                                      friedman: does any rung of the ladder differ?
                                      wilcoxon: the full model against one ablation,
                                      paired over seeds. p_bh is Benjamini-Hochberg within
                                      across ablations.
                                      fisher_z: does one model's accuracy differ between
                                      the treatment arms?
    medusa_vs_graphtrip_interaction.csv
                                      Per-seed cost of an ablation (the drop in r from the
                                      full model), Medusa against graphTRIP. A positive
                                      interaction means the domain matters more for
                                      predicting differential than overall response.

Conventions
    Tests cover the ladder only (the full model and its three ablations); the clinical-only
    benchmarks appear in the performance tables but are not tested against anything.
    Lower is better for mae, mse and rmse; higher for r and r2.
    mse is summarised but not tested: it is a monotone transform of rmse, so the rank tests
    return identical p-values.
    The unit of replication in every seed-wise test is the training seed, not the patient.
'''


def _headlines(name, seed_df, comparisons_df, out):
    '''One line per ablation: what it costs, and on which metrics that cost is reliable.'''
    pairwise = comparisons_df[comparisons_df['test'] == 'wilcoxon']
    reference = pairwise['model_a'].iloc[0]
    reference_r = seed_df[seed_df['model'] == reference]['r'].mean()

    out.log(name)
    for model in pairwise['model_b'].unique():
        rows = pairwise[pairwise['model_b'] == model]
        significant = sorted(rows[rows['significant']]['metric'])
        r_row = rows[rows['metric'] == 'r'].iloc[0]
        verdict = (f'differs on {", ".join(significant)}' if significant
                   else 'no metric separates it from the full model')
        out.log(f'    {model}: r {seed_df[seed_df["model"] == model]["r"].mean():.3f} '
                f'against {reference_r:.3f} (p_bh = {fmt_p(r_row["p_bh"])} on r), '
                f'{verdict}.')
    out.log()


@register('input_domain_ablations', group='supp',
          subdir='SUPPLEMENTARY/input_domain_ablations')
def input_domain_ablations(ctx, out):
    '''Input-domain ablations of both model families, for the side-by-side panel.'''
    graphtrip_seeds, _, graphtrip_tests = family_tables(
        GRAPHTRIP_LADDER, GRAPHTRIP_BENCHMARKS, out, 'graphtrip')
    medusa_seeds, _, medusa_tests = family_tables(MEDUSA_LADDER, [], out, 'medusa')
    interaction = interaction_tests(out)

    out.log(GUIDE)
    out.log('Headlines')
    _headlines('graphTRIP', graphtrip_seeds, graphtrip_tests, out)
    _headlines('Medusa-graphTRIP', medusa_seeds, medusa_tests, out)
    for _, row in interaction.iterrows():
        direction = 'Medusa' if row['interaction'] > 0 else 'graphTRIP'
        out.log(f'    Removing {row["ablated_domain"]} costs {direction} more: '
                f'interaction {row["interaction"]:+.3f} r, '
                f'p = {fmt_p(row["p_wilcoxon"])} (Wilcoxon over seeds).')
