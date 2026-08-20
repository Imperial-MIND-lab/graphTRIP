"""
Fig. 2: Prediction performance of graphTRIP.

Panels:
- a. graphTRIP predictions
- b. model ablations: graphTRIP against the dimensionality-reduction and linear-head
     benchmarks
- c-d. VGAE reconstruction performance
- e. feature ablations, as scatters: graphTRIP without clinical inputs, and the
     clinical-only MLP benchmark
- f. permutation importance

The partial correlations moved to the graphtrip_partial_corrs supplementary target, and
the feature-ablation raincloud to the feature_ablation supplementary target.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from figure_making.common import (
    scatter_from_results, model_comparison_panels, report_prediction_metrics,
    plot_reconstruction_panels, importance_panel)
from figure_making.paths import output_dir, require
from figure_making.registry import register


# The clinical-only MLP replaces the OLS model as the main-text benchmark: it is matched
# in flexibility to graphTRIP, and it is trained on raw scores, as graphTRIP is.
BENCHMARK_DIR = ('ablation', 'feature_ablation', 'control_mlp_raw')

# The brain-only model of panel e
NO_CLINICAL_DIR = ('ablation', 'feature_ablation', 'no_clinical_features')

# Ablations of the architecture, holding every input domain fixed.
MODEL_ABLATIONS = [
    ('graphtrip', ('graphtrip', 'weights')),
    ('pca_benchmark', ('ablation', 'pca_benchmark')),
    ('tsne_benchmark', ('ablation', 'tsne_benchmark')),
    ('vgae_linreg_head', ('ablation', 'vgae_linreg_head')),
]


def _metrics_specs(models):
    return [(label, output_dir(*parts), 'final_metrics.csv') for label, parts in models]


@register('fig2', group='main', subdir='Fig.2')
def fig2_prediction_performance(ctx, out):
    weights_base_dir = ctx.weights_base_dir

    graphtrip_results = os.path.join(weights_base_dir, 'prediction_results.csv')
    no_clinical_results = os.path.join(output_dir(*NO_CLINICAL_DIR), 'prediction_results.csv')
    control_mlp_results = os.path.join(output_dir(*BENCHMARK_DIR), 'prediction_results.csv')

    # a. graphTRIP predictions -----------------------------------------------------------
    scatter_from_results(graphtrip_results, out, 'graphTRIP_true_vs_pred',
                         yerr='prediction_sem', arm_regression=True)

    # b. Model ablations -----------------------------------------------------------------
    out.log('=== Model ablations ===')
    model_comparison_panels(_metrics_specs(MODEL_ABLATIONS), out,
                            'raincloud_model_ablations',
                            num_subs=ctx.num_subs,
                            model_of_interest='graphtrip',
                            table_prefix='model_ablation_')

    # e. Feature ablations, as scatters ---------------------------------------------------
    scatter_from_results(no_clinical_results, out, 'no_clinical_features_true_vs_pred',
                         condition_study='psilodep2', yerr='prediction_sem',
                         arm_regression=True)

    scatter_from_results(control_mlp_results, out, 'control_mlp_true_vs_pred',
                         yerr='prediction_sem', arm_regression=True)

    # Accuracy of every model the Fig. 2 quoted in the text
    out.log('=== Prediction accuracy ===')
    report_prediction_metrics(
        [('graphtrip', graphtrip_results),
         ('no_clinical_features', no_clinical_results),
         ('control_mlp', control_mlp_results)],
        out)

    # c-d. VGAE reconstruction performance ------------------------------------------------
    plot_reconstruction_panels(ctx, out, ctx.core_reconstructions,
                               atlas=ctx.atlas,
                               rsn_mapping=ctx.rsn_mapping,
                               rsn_labels=ctx.rsn_names,
                               conditions=ctx.conditions,
                               brain_subdir='graphTRIP_reconstructions')

    # f. Permutation importance -----------------------------------------------------------
    importance_panel(require(output_dir('graphtrip', 'permutation_importance')),
                     weights_base_dir, out)
