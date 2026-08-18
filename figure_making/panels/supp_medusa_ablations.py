"""
Supplementary: feature ablations of Medusa-graphTRIP.

Author: Hanna M. Tolle
Date: 2026-08-18
License: BSD 3-Clause
"""

import numpy as np
import pandas as pd
from scipy import stats

from figure_making.common import (
    scatter_from_results, collect_seed_metrics, feature_ablation_panel)
from figure_making.paths import output_dir
from figure_making.registry import register

import os


# Ordered as a nested ladder, read top to bottom.
MEDUSA_FEATURE_ABLATIONS = [
    ('medusa', ('medusa_graphtrip', 'weights'),
     ('Clinical', 'FC', 'REACT')),
    ('no_node_features', ('medusa_ablation', 'no_node_features'),
     ('Clinical', 'FC')),
    ('no_clinical_features', ('medusa_ablation', 'no_clinical_features'),
     ('FC', 'REACT')),
    ('no_react_no_clinical', ('medusa_ablation', 'no_react_no_clinical'),
     ('FC',)),
]

MEDUSA_SCATTERS = [
    (('medusa_ablation', 'no_node_features'), 'medusa_no_node_features_true_vs_pred',
     'Medusa, Trained without REACT Node Features'),
    (('medusa_ablation', 'no_clinical_features'), 'medusa_no_clinical_features_true_vs_pred',
     'Medusa, Trained without Clinical Features'),
    (('medusa_ablation', 'no_react_no_clinical'), 'medusa_no_react_no_clinical_true_vs_pred',
     'Medusa, Trained without REACT Node or Clinical Features'),
]

# (contrast name, ablated graphTRIP model, ablated Medusa model). Each pair is the same
# ablation applied to both families.
INTERACTION_CONTRASTS = [
    ('REACT node features', ('ablation', 'feature_ablation', 'no_node_features'),
     ('medusa_ablation', 'no_node_features')),
    ('clinical features', ('ablation', 'feature_ablation', 'no_clinical_features'),
     ('medusa_ablation', 'no_clinical_features')),
]

GRAPHTRIP_FULL = ('graphtrip', 'weights')
MEDUSA_FULL = ('medusa_graphtrip', 'weights')


def _seed_r(specs):
    '''Returns a {model: r} dataframe indexed by seed, so that models stay seed-aligned.'''
    metrics = collect_seed_metrics(specs)
    return metrics.pivot(index='seed', columns='model', values='r')


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
        r = _seed_r(specs).dropna()
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
    out.log('=== Medusa versus graphTRIP: does the ablation cost differ? ===')
    out.log('Positive interaction: the domain matters more for Medusa.')
    out.log_df('Interaction tests', table)
    out.table('medusa_vs_graphtrip_interaction', table)
    return table


@register('medusa_ablations', group='supp', subdir='SUPPLEMENTARY/medusa_ablations')
def medusa_ablations(ctx, out):
    '''Medusa feature ablations, and how they compare with graphTRIP's.'''
    for parts, name, title in MEDUSA_SCATTERS:
        scatter_from_results(
            os.path.join(output_dir(*parts), 'prediction_results.csv'),
            out, name, yerr='prediction_sem', title=title)

    out.log('=== Medusa feature ablations ===')
    feature_ablation_panel(
        [(label, output_dir(*parts), 'final_metrics.csv')
         for label, parts, _ in MEDUSA_FEATURE_ABLATIONS],
        {label: domains for label, _, domains in MEDUSA_FEATURE_ABLATIONS},
        out, 'raincloud_medusa_feature_ablations',
        num_subs=ctx.num_subs,
        reference_model='medusa',
        table_prefix='medusa_feature_ablation_')

    interaction_tests(out)
