"""
Fig. 2: Prediction performance of graphTRIP.

Panels:
- a. graphTRIP and the clinical-only MLP benchmark
- b. model ablations: graphTRIP against the dimensionality-reduction and linear-head
     benchmarks
- c. feature ablations: what each input domain contributes
- d-e. VGAE reconstruction performance
- f. permutation importance

The partial correlations moved to the graphtrip_partial_corrs supplementary target.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_importance_scores
from utils.plotting import NEUTRAL, permutation_importance_bar_chart

from figure_making.common import (
    scatter_from_results, model_comparison_panels, feature_ablation_panel,
    plot_reconstruction_panels)
from figure_making.paths import output_dir, require
from figure_making.registry import register


# The clinical-only MLP replaces the OLS model as the main-text benchmark: it is matched
# in flexibility to graphTRIP, and it is trained on raw scores, as graphTRIP is.
BENCHMARK_DIR = ('ablation', 'feature_ablation', 'control_mlp_raw')

# Ablations of the architecture, holding every input domain fixed.
MODEL_ABLATIONS = [
    ('graphtrip', ('graphtrip', 'weights')),
    ('pca_benchmark', ('ablation', 'pca_benchmark')),
    ('tsne_benchmark', ('ablation', 'tsne_benchmark')),
    ('vgae_linreg_head', ('ablation', 'vgae_linreg_head')),
]

# Ablations of the inputs, holding the architecture fixed. Ordered as a nested ladder,
# read top to bottom in the panel. Condition is retained wherever the model needs to know
# the treatment arm, and is not counted as a clinical input.
FEATURE_ABLATIONS = [
    ('graphtrip', ('graphtrip', 'weights'),
     ('Clinical', 'FC', 'REACT')),
    ('no_node_features', ('ablation', 'feature_ablation', 'no_node_features'),
     ('Clinical', 'FC')),
    ('no_clinical_features', ('ablation', 'feature_ablation', 'no_clinical_features'),
     ('FC', 'REACT')),
    ('no_react_no_clinical', ('ablation', 'feature_ablation', 'no_react_no_clinical'),
     ('FC',)),
    ('control_mlp', ('ablation', 'feature_ablation', 'control_mlp_raw'),
     ('Clinical',)),
    ('linreg_on_clinical_data', ('ablation', 'feature_ablation', 'linreg_on_clinical_data'),
     ('Clinical',)),
]


def _metrics_specs(models):
    return [(label, output_dir(*parts), 'final_metrics.csv') for label, parts in models]


@register('fig2', group='main', subdir='Fig.2')
def fig2_prediction_performance(ctx, out):
    weights_base_dir = ctx.weights_base_dir

    # a. graphTRIP and the clinical-only MLP benchmark -----------------------------------
    scatter_from_results(
        os.path.join(weights_base_dir, 'prediction_results.csv'),
        out, 'graphTRIP_true_vs_pred', yerr='prediction_sem')

    scatter_from_results(
        os.path.join(output_dir(*BENCHMARK_DIR), 'prediction_results.csv'),
        out, 'control_mlp_true_vs_pred', yerr='prediction_sem')

    # b. Model ablations -----------------------------------------------------------------
    out.log('=== Model ablations ===')
    model_comparison_panels(_metrics_specs(MODEL_ABLATIONS), out,
                            'raincloud_model_ablations',
                            num_subs=ctx.num_subs,
                            model_of_interest='graphtrip',
                            table_prefix='model_ablation_')

    # c. Feature ablations ---------------------------------------------------------------
    out.log('=== Feature ablations ===')
    feature_ablation_panel(
        [(label, output_dir(*parts), 'final_metrics.csv') for label, parts, _ in FEATURE_ABLATIONS],
        {label: domains for label, _, domains in FEATURE_ABLATIONS},
        out, 'raincloud_feature_ablations',
        num_subs=ctx.num_subs,
        reference_model='graphtrip',
        table_prefix='feature_ablation_')

    # d-e. VGAE reconstruction performance ------------------------------------------------
    plot_reconstruction_panels(ctx, out, ctx.core_reconstructions,
                               atlas=ctx.atlas,
                               rsn_mapping=ctx.rsn_mapping,
                               rsn_labels=ctx.rsn_names,
                               conditions=ctx.conditions,
                               brain_subdir='graphTRIP_reconstructions')

    # f. Permutation importance -----------------------------------------------------------
    importance_dir = require(output_dir('graphtrip', 'permutation_importance'))
    scores = aggregate_importance_scores(
        os.path.join(importance_dir, 'importance_scores_aggregated.csv'))
    scores = scores.sort_values(by='mean', ascending=False)
    permutation_importance_bar_chart(scores, yerr_column='se', color=NEUTRAL, alpha=0.8,
                                     save_path=out.fig('importance_scores_aggregated'))
