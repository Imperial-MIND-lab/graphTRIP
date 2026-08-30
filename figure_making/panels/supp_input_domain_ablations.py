"""
Supplementary: input-domain ablations of graphTRIP and of Medusa-graphTRIP.

Both families are presented side by side, so they share this target's output directory:
the same ablation ladder (clinical, FC and REACT node features, held against the full
model), the same ordering, and the same names for the retained input domains. The
statistics of the two families are written to separate files, since a FigureOutput
holds one stats.txt per target.

Author: Hanna M. Tolle
Date: 2026-08-30
License: BSD 3-Clause
"""

import os

import pandas as pd
from scipy import stats

from figure_making.common import (
    scatter_from_results, collect_seed_metrics, feature_ablation_panel,
    report_prediction_metrics)
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

# (display name, results directory parts, input domains received), read top to bottom.
# The ladder is ordered identically for both families, so the two rainclouds align.
GRAPHTRIP_ABLATIONS = [
    ('graphTRIP', GRAPHTRIP_FULL, ('Clinical', 'FC', 'REACT')),
    (ABLATION_NAMES['no_node_features'],
     ('ablation', 'feature_ablation', 'no_node_features'), ('Clinical', 'FC')),
    (ABLATION_NAMES['no_clinical_features'],
     ('ablation', 'feature_ablation', 'no_clinical_features'), ('FC', 'REACT')),
    (ABLATION_NAMES['no_react_no_clinical'],
     ('ablation', 'feature_ablation', 'no_react_no_clinical'), ('FC',)),
]

MEDUSA_ABLATIONS = [
    ('Medusa-graphTRIP', MEDUSA_FULL, ('Clinical', 'FC', 'REACT')),
    (ABLATION_NAMES['no_node_features'],
     ('medusa_ablation', 'no_node_features'), ('Clinical', 'FC')),
    (ABLATION_NAMES['no_clinical_features'],
     ('medusa_ablation', 'no_clinical_features'), ('FC', 'REACT')),
    (ABLATION_NAMES['no_react_no_clinical'],
     ('medusa_ablation', 'no_react_no_clinical'), ('FC',)),
]

# (results directory parts, panel name, title). One scatter per ablated model; the full
# models are scattered in the main figures.
GRAPHTRIP_SCATTERS = [
    (parts, f'graphtrip_{parts[-1]}_true_vs_pred', f'graphTRIP: {name}')
    for name, parts, _ in GRAPHTRIP_ABLATIONS[1:]
] + [
    # Clinical-only benchmarks: not part of the ablation ladder, but their accuracy is
    # what the ablated models are worth comparing against.
    (('ablation', 'feature_ablation', 'control_mlp_raw'), 'control_mlp_true_vs_predicted',
     'MLP, Trained on Clinical Data'),
    (('ablation', 'feature_ablation', 'linreg_on_clinical_data'),
     'linreg_on_clinical_data_true_vs_pred', 'OLS Regression, Trained on Clinical Data'),
]

MEDUSA_SCATTERS = [
    (parts, f'medusa_{parts[-1]}_true_vs_pred', f'Medusa-graphTRIP: {name}')
    for name, parts, _ in MEDUSA_ABLATIONS[1:]
]

# (contrast name, ablated graphTRIP model, ablated Medusa model). Each pair is the same
# ablation applied to both families.
INTERACTION_CONTRASTS = [
    ('REACT node features', ('ablation', 'feature_ablation', 'no_node_features'),
     ('medusa_ablation', 'no_node_features')),
    ('clinical features', ('ablation', 'feature_ablation', 'no_clinical_features'),
     ('medusa_ablation', 'no_clinical_features')),
]


class Section:
    '''
    A FigureOutput that buffers its log lines instead of writing them to stats.txt.

    Panels and tables are delegated unchanged, so both families write into the target's
    single directory, while their statistics end up in <name>.txt rather than
    overwriting one another.
    '''

    def __init__(self, out, name):
        self._out = out
        self._name = name
        self._lines = []

    def __getattr__(self, attr):
        return getattr(self._out, attr)

    def log(self, msg=''):
        msg = str(msg)
        self._lines.append(msg)
        if self._out.cfg.verbose:
            print(msg)

    def log_df(self, title, df, index=False):
        self.log(title)
        self.log(df.to_string(index=index))
        self.log()

    def write(self):
        '''Writes the buffered lines as <name>.txt.'''
        if self._lines:
            self._out.text(self._name, '\n'.join(self._lines))


def _results_file(parts):
    return os.path.join(output_dir(*parts), 'prediction_results.csv')


def _scatters(specs, out):
    for parts, name, title in specs:
        scatter_from_results(_results_file(parts), out, name,
                             yerr='prediction_sem', title=title)


def _metrics(specs, out, name):
    '''Accuracy of each scattered model, so the numbers exist outside the panel titles.'''
    report_prediction_metrics(
        [(title, _results_file(parts)) for parts, _, title in specs], out, name=name)


def _raincloud(ablations, out, name, table_prefix):
    feature_ablation_panel(
        [(label, output_dir(*parts), 'final_metrics.csv')
         for label, parts, _ in ablations],
        {label: domains for label, _, domains in ablations},
        out, name,
        reference_model=ablations[0][0],
        table_prefix=table_prefix)


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


@register('input_domain_ablations', group='supp',
          subdir='SUPPLEMENTARY/input_domain_ablations')
def input_domain_ablations(ctx, out):
    '''Input-domain ablations of both model families, for the side-by-side panel.'''
    graphtrip = Section(out, 'graphtrip_stats')
    medusa = Section(out, 'medusa_stats')

    _scatters(GRAPHTRIP_SCATTERS, graphtrip)
    _metrics(GRAPHTRIP_SCATTERS, graphtrip, 'graphtrip_prediction_metrics')
    graphtrip.log('=== graphTRIP input-domain ablations ===')
    _raincloud(GRAPHTRIP_ABLATIONS, graphtrip, 'raincloud_feature_ablations_graphtrip',
               'graphtrip_feature_ablation_')

    _scatters(MEDUSA_SCATTERS, medusa)
    _metrics(MEDUSA_SCATTERS, medusa, 'medusa_prediction_metrics')
    medusa.log('=== Medusa-graphTRIP input-domain ablations ===')
    _raincloud(MEDUSA_ABLATIONS, medusa, 'raincloud_feature_ablations_medusa',
               'medusa_feature_ablation_')
    interaction_tests(medusa)

    graphtrip.write()
    medusa.write()
