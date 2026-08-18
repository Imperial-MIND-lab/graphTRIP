"""
Supplementary: model ablations and feature ablations.

Model ablations replace the VGAE (PCA, t-SNE), the MLP (linear prediction head,
post-hoc ridge regression on z, retrained MLP head on z), or the neuroimaging
features (clinical data only).
Feature ablations retrain graphTRIP and Medusa without clinical or without REACT
node features.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from figure_making.common import scatter_from_results, model_comparison_panels
from figure_making.paths import output_dir
from figure_making.registry import register


# (results directory parts, panel name, title, study for the condition column)
ABLATION_SCATTERS = [
    (('ablation', 'pca_benchmark'), 'pca_true_vs_predicted', 'PCA', None),
    (('ablation', 'tsne_benchmark'), 'tsne_true_vs_predicted', 't-SNE', None),
    (('ablation', 'vgae_linreg_head'), 'vgae_with_linreg_head_true_vs_predicted',
     'VGAE, trained with linear prediction head', None),
    (('graphtrip', 'linreg_on_z'), 'linreg_on_z_true_vs_predicted',
     'Frozen graphTRIP-VGAE, with post-hoc ridge regression on z', 'psilodep2'),
    (('graphtrip', 'retrain_mlp_on_z'), 'retrain_mlp_on_z_true_vs_predicted',
     'Frozen graphTRIP-VGAE, with retrained MLP head on z', 'psilodep2'),
    (('ablation', 'feature_ablation', 'control_mlp_raw'), 'control_mlp_true_vs_predicted',
     'MLP, Trained on Clinical Data', None),
    (('ablation', 'feature_ablation', 'linreg_on_clinical_data'), 'linreg_on_clinical_data_true_vs_pred',
     'OLS Regression, Trained on Clinical Data', None),
]

ABLATION_MODELS = [
    ('graphtrip', ('graphtrip', 'weights')),
    ('control_mlp', ('ablation', 'feature_ablation', 'control_mlp_raw')),
    ('vgae_linreg_head', ('ablation', 'vgae_linreg_head')),
    ('linreg_on_clinical_data', ('ablation', 'feature_ablation', 'linreg_on_clinical_data')),
    ('pca_benchmark', ('ablation', 'pca_benchmark')),
    ('tsne_benchmark', ('ablation', 'tsne_benchmark')),
    ('linreg_on_z', ('graphtrip', 'linreg_on_z')),
    ('retrain_mlp_on_z', ('graphtrip', 'retrain_mlp_on_z')),
]

GRAPHTRIP_FEATURE_ABLATIONS = [
    ('graphtrip', ('graphtrip', 'weights')),
    ('no_clinical_features', ('ablation', 'feature_ablation', 'no_clinical_features')),
    ('no_node_features', ('ablation', 'feature_ablation', 'no_node_features')),
]

MEDUSA_FEATURE_ABLATIONS = [
    ('medusa', ('medusa_graphtrip', 'weights')),
    ('no_clinical_features', ('medusa_ablation', 'no_clinical_features')),
    ('no_node_features', ('medusa_ablation', 'no_node_features')),
]


def _metrics_specs(models):
    return [(label, output_dir(*parts), 'final_metrics.csv') for label, parts in models]


@register('ablation_models', group='supp', subdir='SUPPLEMENTARY/ablation_models')
def ablation_models(ctx, out):
    # Ablate the VGAE, the MLP, and the neuroimaging features ---------------------------
    for parts, name, title, study in ABLATION_SCATTERS:
        scatter_from_results(os.path.join(output_dir(*parts), 'prediction_results.csv'),
                             out, name, condition_study=study,
                             yerr='prediction_sem', title=title)

    # Random seed sensitivity ------------------------------------------------------------
    model_comparison_panels(_metrics_specs(ABLATION_MODELS), out,
                            'raincloud_graphtrip_vs_benchmarks',
                            num_subs=ctx.num_subs,
                            model_of_interest='graphtrip')


@register('feature_ablation', group='supp', subdir='SUPPLEMENTARY/feature_ablation')
def feature_ablation(ctx, out):
    # a. graphTRIP feature ablation ------------------------------------------------------
    scatter_from_results(
        os.path.join(output_dir('ablation', 'feature_ablation', 'no_clinical_features'), 'prediction_results.csv'),
        out, 'no_clinical_features_true_vs_pred', yerr='prediction_sem',
        title='graphTRIP, Trained without Clinical Features')
    scatter_from_results(
        os.path.join(output_dir('ablation', 'feature_ablation', 'no_node_features'), 'prediction_results.csv'),
        out, 'no_node_features_true_vs_pred', yerr='prediction_sem',
        title='graphTRIP, Trained without REACT Node Features')

    out.log('=== graphTRIP feature ablation ===')
    model_comparison_panels(_metrics_specs(GRAPHTRIP_FEATURE_ABLATIONS), out,
                            'raincloud_graphtrip_feature_ablation',
                            num_subs=ctx.num_subs,
                            model_of_interest='graphtrip',
                            table_prefix='graphtrip_')

    # b. Medusa feature ablation ---------------------------------------------------------
    scatter_from_results(
        os.path.join(output_dir('medusa_ablation', 'no_clinical_features'),
                     'prediction_results.csv'),
        out, 'medusa_no_clinical_features_true_vs_pred', yerr='prediction_sem',
        title='Medusa, Trained without Clinical Features')
    scatter_from_results(
        os.path.join(output_dir('medusa_ablation', 'no_node_features'),
                     'prediction_results.csv'),
        out, 'medusa_no_node_features_true_vs_pred', yerr='prediction_sem',
        title='Medusa, Trained without REACT Node Features')

    # NOTE: the notebook wrote this raincloud and its two tables under the same names as
    # the graphTRIP block above, so the graphTRIP versions were silently overwritten. The
    # names below match the surviving reference files; the graphTRIP block writes to
    # distinct names so that both are kept.
    out.log('=== Medusa feature ablation ===')
    model_comparison_panels(_metrics_specs(MEDUSA_FEATURE_ABLATIONS), out,
                            'raincloud_graphtrip_vs_benchmarks',
                            num_subs=ctx.num_subs,
                            model_of_interest='medusa')
